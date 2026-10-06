"""Test-set metrics (PSNR, SSIM, LPIPS-VGG) and inference FPS of a trained model with the inference kernel
(foam_infer.cu): 'fast' = fused colors, fp16 SH, negligible terms (< 1e-4) skipped, Z-order sites, 8x4 pixel
tiles; 'exact' = same samples and arithmetic as the training kernel. Metrics use the training t_thresh; FPS is
measured on every 8th test view at the training resolution and at 1/8 resolution, for T < training t_thresh and
T < 1e-3. Writes <exp>/eval.json."""

import argparse
import json
import os

import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity

from bench_fps import load_model
from bench_infer import tile_order
from bench_vs_radfoam import time_ms
from colmap import ColmapScene
from render import pack_foam, render_rays_foam, render_rays_foam_infer
from sites import nearest_sites

CONFIGS = {
	'fast': (dict(sh_half=True, eps_density=1e-4, reorder=True), dict(colors='fused', eps_color=1e-4), True),
	'exact': (dict(), dict(colors='exact'), False),
}


class Renderer:
	def __init__(self, model, scene, starts, far_radius):
		self.model, self.scene, self.starts, self.far_radius = model, scene, starts, far_radius
		self.perm = tile_order(scene.H, scene.W)
		self.packs = {name: pack_foam(model, **p) for name, (p, _, _) in CONFIGS.items()}

	def fn(self, name, v, t_thresh):
		o, d, gt = self.scene.image_rays(v)
		st = torch.full((len(o),), int(self.starts[v]), dtype=torch.int32, device='cuda')
		if name == 'training kernel':
			return (lambda: render_rays_foam(self.model, o, d, st, far_radius=self.far_radius,
											 t_thresh=t_thresh)['rgb']), gt
		_, kw, tile = CONFIGS[name]
		if tile:
			o, d = o[self.perm].contiguous(), d[self.perm].contiguous()

		def run():
			rgb = render_rays_foam_infer(self.model, self.packs[name], o, d, st, far_radius=self.far_radius,
										 t_thresh=t_thresh, **kw)['rgb']
			if tile:
				out = torch.empty_like(rgb)
				out[self.perm] = rgb
				rgb = out
			return rgb
		return run, gt


@torch.no_grad()
def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--exp', required=True)
	ap.add_argument('--reps', type=int, default=3)
	args = ap.parse_args()
	model, ck = load_model(os.path.join(args.exp, 'model.pth'))
	targs = ck['args']
	far_radius, t_train = targs['far_radius'], targs['t_thresh']
	lp = lpips.LPIPS(net='vgg', verbose=False).cuda()
	result = {'exp': args.exp, 'n_sites': model.n_sites, 'metrics': {}, 'fps': []}

	scene = ColmapScene(targs['data'], 'test', targs['downsample'], normalization=ck['normalization'])
	starts = nearest_sites(model.graph.points.cpu().numpy(), scene.centers.cpu().numpy())
	R = Renderer(model, scene, starts, far_radius)
	for name in ('fast', 'exact', 'training kernel'):
		ps, ss, ls, diffs = [], [], [], []
		for v in range(scene.n):
			fn, gt = R.fn(name, v, t_train)
			rgb = fn().view(scene.H, scene.W, 3).clamp(0, 1)
			ps.append(float(-10 * torch.log10(((rgb - gt) ** 2).mean())))
			ss.append(float(structural_similarity(rgb.cpu().numpy(), gt.cpu().numpy(), channel_axis=2,
												  data_range=1.0)))
			ls.append(float(lp(rgb.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1)))
		result['metrics'][name] = {'psnr': float(np.mean(ps)), 'ssim': float(np.mean(ss)), 'lpips_vgg': float(np.mean(ls)),
								   'n_views': scene.n, 'resolution': [scene.W, scene.H], 't_thresh': t_train}
		m = result['metrics'][name]
		print(f'{name:16s} test {scene.n} views {scene.W}x{scene.H}: PSNR {m["psnr"]:.3f}  SSIM {m["ssim"]:.4f}  '
			  f'LPIPS {m["lpips_vgg"]:.4f}', flush=True)

	for ds in sorted({targs['downsample'], 8}):
		if ds != targs['downsample']:
			scene = ColmapScene(targs['data'], 'test', ds, normalization=ck['normalization'])
			R = Renderer(model, scene, starts, far_radius)
		for t in sorted({t_train, 1e-3}):
			for name in ('fast', 'exact', 'training kernel'):
				views = range(0, scene.n, 8)
				ms = [time_ms(R.fn(name, v, t)[0], 1, args.reps if name != 'training kernel' else 1) for v in views]
				row = {'renderer': name, 'res': f'{scene.W}x{scene.H}', 't_thresh': t, 'ms': float(np.mean(ms)),
					   'fps': 1000 / float(np.mean(ms))}
				result['fps'].append(row)
				print(f'  {row["res"]:9s} T<{t:<7g} {name:16s} {row["ms"]:9.2f} ms  {row["fps"]:7.2f} FPS', flush=True)
	with open(os.path.join(args.exp, 'eval.json'), 'w') as fh:
		json.dump(result, fh, indent=1)


if __name__ == '__main__':
	main()
