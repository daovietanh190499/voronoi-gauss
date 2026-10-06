"""Inference speed of the interaction-point renderer: training kernel (foam.cu, warp per ray) vs the inference
kernel (foam_infer.cu) and its options (fp16 SH, skipped negligible density / color terms, baked colors, pixel
tiles), plus a stage ablation (exit faces / + density / full) with every ray's traversal fixed to the one of the
full renderer. Every 8th test view; times and image metrics averaged over views, max |diff| over views."""

import argparse
import json
import os

import numpy as np
import torch

from bench_fps import load_model
from bench_vs_radfoam import psnr, time_ms
from colmap import ColmapScene
from render import bake_colors, pack_foam, render_rays_foam, render_rays_foam_infer
from sites import nearest_sites

CONFIGS = {
	'lazy, fp32, no skipping': dict(),
	'lazy, fp32, no skipping, tiles': dict(tile=True),
	'lazy, fp16 SH': dict(sh_half=True),
	'lazy, eps_color 1e-3': dict(eps_color=1e-3),
	'lazy, eps_density 1e-3': dict(eps_density=1e-3),
	'lazy, fp16 + eps 1e-4 + tiles': dict(sh_half=True, eps_color=1e-4, eps_density=1e-4, tile=True),
	'fused, fp32, no skipping, tiles': dict(colors='fused', tile=True),
	'fused, fp16 + eps 1e-4 + tiles': dict(colors='fused', sh_half=True, eps_color=1e-4, eps_density=1e-4, tile=True),
	'fused, fp16 + eps 1e-3 + tiles': dict(colors='fused', sh_half=True, eps_color=1e-3, eps_density=1e-3, tile=True),
	'fused, fp16 + eps 1e-2 + tiles': dict(colors='fused', sh_half=True, eps_color=1e-2, eps_density=1e-2, tile=True),
	'fused, fp16 + eps 1e-4 + tiles + morton': dict(colors='fused', sh_half=True, eps_color=1e-4, eps_density=1e-4,
												  tile=True, reorder=True),
	'fused, fp32, no skipping, tiles + morton': dict(colors='fused', tile=True, reorder=True),
	'baked + eps 1e-3 + tiles (incl. bake)': dict(colors='baked', eps_color=1e-3, eps_density=1e-3, tile=True),
}


def tile_order(H, W, th=4, tw=8):
	"""Pixel permutation grouping 32 consecutive rays into th x tw tiles."""
	y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
	key = ((y // th) * ((W + tw - 1) // tw) + x // tw) * (th * tw) + (y % th) * tw + x % tw
	return torch.argsort(key.flatten()).cuda()


@torch.no_grad()
def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--exp', default='exp/bonsai_foam')
	ap.add_argument('--downsample', type=int, nargs='*', default=[8, 4])
	ap.add_argument('--t_thresh', type=float, default=1e-3)
	ap.add_argument('--configs', nargs='*', default=list(CONFIGS))
	ap.add_argument('--no_ablation', action='store_true')
	ap.add_argument('--reps', type=int, default=3)
	args = ap.parse_args()
	model, ck = load_model(os.path.join(args.exp, 'model.pth'))
	kw = dict(far_radius=ck['args']['far_radius'], t_thresh=args.t_thresh)
	packs = {}
	rows = []
	for ds in args.downsample:
		scene = ColmapScene(ck['args']['data'], 'test', ds, normalization=ck['normalization'])
		starts = nearest_sites(model.graph.points.cpu().numpy(), scene.centers.cpu().numpy())
		perm = tile_order(scene.H, scene.W)
		res, info = {}, {'samples': [], 'color_frac': []}
		for v in range(0, scene.n, 8):
			o, d, gt = scene.image_rays(v)
			st = torch.full((len(o),), int(starts[v]), dtype=torch.int32, device='cuda')
			ref = render_rays_foam(model, o, d, st, **kw)['rgb']
			runs = {'train kernel (warp/ray)': (lambda: render_rays_foam(model, o, d, st, **kw)['rgb'], None)}
			if 'base' not in packs:
				packs['base'] = pack_foam(model)
			base_pack = packs['base']
			full = render_rays_foam_infer(model, base_pack, o, d, st, **kw)
			cap = full['stats'][:, 0]
			info['samples'].append(float(cap.float().mean()))
			info['color_frac'].append(float(full['stats'][:, 1].sum() / (cap - 1).sum()))
			if not args.no_ablation:
				runs['ablation: exit faces only'] = (
					lambda: render_rays_foam_infer(model, base_pack, o, d, st, stage=0, cap=cap, **kw), 'nometric')
				runs['ablation: + density'] = (
					lambda: render_rays_foam_infer(model, base_pack, o, d, st, stage=1, cap=cap, **kw), 'nometric')
			for name in args.configs:
				cfg = dict(CONFIGS[name])
				tile = cfg.pop('tile', False)
				pk = (cfg.pop('sh_half', False), cfg.pop('eps_density', 0.0), cfg.pop('reorder', False))
				if pk not in packs:
					packs[pk] = pack_foam(model, sh_half=pk[0], eps_density=pk[1], reorder=pk[2])
				pack = packs[pk]
				oo, dd = (o[perm].contiguous(), d[perm].contiguous()) if tile else (o, d)

				def fn(pack=pack, cfg=cfg, oo=oo, dd=dd, tile=tile):
					if cfg.get('colors') == 'baked':
						bake_colors(model, pack, scene.centers[v])
					out = render_rays_foam_infer(model, pack, oo, dd, st, **cfg, **kw)
					if tile:
						rgb = torch.empty_like(out['rgb'])
						rgb[perm] = out['rgb']
						stats = torch.empty_like(out['stats'])
						stats[perm] = out['stats']
						out = {'rgb': rgb, 'stats': stats}
					return out
				runs[name] = (fn, None)
			for name, (fn, kind) in runs.items():
				out = fn()
				e = res.setdefault(name, {'ms': [], 'psnr': [], 'max_diff': [], 'mean_diff': [], 'dens': [], 'sh': []})
				e['ms'].append(time_ms(fn, 1, args.reps))
				if kind == 'nometric':
					continue
				img = out if torch.is_tensor(out) else out['rgb']
				e['psnr'].append(psnr(img.view(scene.H, scene.W, 3), gt))
				diff = (img - ref).abs()
				e['max_diff'].append(float(diff.max()))
				e['mean_diff'].append(float(diff.mean()))
				if isinstance(out, dict):
					s = out['stats'].double()
					e['dens'].append(float(s[:, 3].sum() / s[:, 0].sum()))
					e['sh'].append(float(s[:, 4].sum() / s[:, 1].sum().clamp_min(1)))
		print(f'\n{scene.W}x{scene.H}  T<{args.t_thresh:g}  samples/ray {np.mean(info["samples"]):.1f}  '
			  f'segments with color {100 * np.mean(info["color_frac"]):.1f}%')
		print(f'  {"renderer":34s} {"ms":>8s} {"FPS":>7s} {"x":>6s} {"PSNR":>7s} {"max|d|":>8s} {"mean|d|":>8s} '
			  f'{"dens/smp":>8s} {"SH/seg":>7s}')
		base = np.mean(res['train kernel (warp/ray)']['ms'])
		for name, e in res.items():
			ms = float(np.mean(e['ms']))
			row = {'res': f'{scene.W}x{scene.H}', 'renderer': name, 'ms': ms, 'fps': 1000 / ms, 'speedup': base / ms}
			for k in ('psnr', 'mean_diff', 'dens', 'sh'):
				row[k] = float(np.mean(e[k])) if e[k] else None
			row['max_diff'] = float(np.max(e['max_diff'])) if e['max_diff'] else None
			rows.append(row)
			f = lambda k, fmt: format(row[k], fmt) if row[k] is not None else '-'
			print(f'  {name:34s} {ms:8.2f} {1000 / ms:7.2f} {base / ms:6.2f} {f("psnr", ".3f"):>7s} '
				  f'{f("max_diff", ".1e"):>8s} {f("mean_diff", ".1e"):>8s} {f("dens", ".1f"):>8s} {f("sh", ".1f"):>7s}')
	os.makedirs('exp/fps_vs_radfoam', exist_ok=True)
	with open('exp/fps_vs_radfoam/infer.json', 'w') as fh:
		json.dump(rows, fh, indent=1)


if __name__ == '__main__':
	main()
