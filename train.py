"""Trains the Voronoi-Gaussian field on a COLMAP scene (Mip-NeRF 360 layout) with a photometric loss."""

import argparse
import json
import math
import os
import resource
import time

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

import ext
from colmap import ColmapScene
from model import VoronoiGaussField
from render import render_image, render_rays, render_rays_foam, render_rays_uniform
from sites import delaunay_graph, nearest_sites, neighbor_spacing, scene_sites

LR = {'raw': 2e-2, 'scale': 5e-3, 'quat': 2e-3, 'sh_dc': 5e-3, 'sh_rest': 2.5e-4, 'bg': 1e-2}


def parse_args():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data', default='../radfoam/data/mipnerf360/bonsai')
	ap.add_argument('--out', default='exp/bonsai')
	ap.add_argument('--iters', type=int, default=20000)
	ap.add_argument('--batch', type=int, default=4096, help='rays per iteration')
	ap.add_argument('--downsample', type=int, default=4, help='train / test images_<d>')
	ap.add_argument('--far_radius', type=float, default=4.0, help='sites and samples beyond the unit ball, up to here')
	ap.add_argument('--sampler', choices=['nerf', 'uniform', 'foam'], default='foam',
					help='nerf: 64 stratified + 2 x 64 proposal samples; uniform: fixed steps, fused kernel; '
						 'foam: ray / Voronoi-face interaction points, fused kernel')
	ap.add_argument('--dt', type=float, default=1 / 512, help='uniform step inside the unit ball')
	ap.add_argument('--n_bg', type=int, default=64, help='uniform-disparity steps beyond the unit ball')
	ap.add_argument('--t_thresh', type=float, default=1e-4, help='early termination (uniform, foam)')
	ap.add_argument('--n_coarse', type=int, default=64)
	ap.add_argument('--n_fine', type=int, nargs=2, default=[64, 64])
	ap.add_argument('--proposal_mult', type=float, nargs=2, default=[4.0, 2.0])
	ap.add_argument('--init_alpha', type=float, default=0.1)
	ap.add_argument('--lr_mult', type=float, default=1.0)
	ap.add_argument('--sh_every', type=int, default=2000)
	ap.add_argument('--warmup', type=int, default=300)
	ap.add_argument('--log_every', type=int, default=100)
	ap.add_argument('--val_every', type=int, default=5000)
	ap.add_argument('--val_views', type=int, nargs='*', default=[0, 12, 24])
	ap.add_argument('--seed', type=int, default=0)
	return ap.parse_args()


def lr_factor(it, total, warmup, alpha=0.05):
	if it < warmup:
		return (it + 1) / warmup
	progress = (it - warmup) / max(1, total - warmup)
	return (math.cos(math.pi * progress) + 1) * 0.5 * (1 - alpha) + alpha


def to_png(x, path):
	Image.fromarray((x.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(path)


def host_peak_gb():
	return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def evaluate(model, scene, starts, i, kw):
	rgb, gt = render_image(model, scene, i, starts[i], chunk=32768, **kw)
	mse = ((rgb - gt) ** 2).mean()
	return rgb, gt, {'psnr': float(-10 * torch.log10(mse)),
					 'ssim': float(structural_similarity(rgb.cpu().numpy(), gt.cpu().numpy(), channel_axis=2,
														 data_range=1.0))}


def main():
	args = parse_args()
	torch.manual_seed(args.seed)
	os.makedirs(args.out, exist_ok=True)
	log_file = open(os.path.join(args.out, 'train.log'), 'a')

	def log(msg):
		print(msg, flush=True)
		log_file.write(msg + '\n')
		log_file.flush()

	log(f'== {time.strftime("%Y-%m-%d %H:%M:%S")} {json.dumps(vars(args))}')
	ext.extension()
	scene = ColmapScene(args.data, 'train', args.downsample)
	test_scene = ColmapScene(args.data, 'test', args.downsample, normalization=scene.normalization)
	log(f'scene {args.data}: {scene.n} train views, {test_scene.n} test views, {scene.W}x{scene.H}')

	t0 = time.time()
	pts, cols = scene_sites(scene, radius=args.far_radius)
	pts, keep, offsets, adj = delaunay_graph(pts)
	cols = cols[keep]
	spacing = neighbor_spacing(pts)
	deg = np.diff(offsets)
	log(f'sites: {len(pts)} of {len(scene.points)} sparse points | Delaunay {time.time() - t0:.1f} s, degree mean '
		f'{deg.mean():.1f} max {deg.max()} | spacing median {np.median(spacing):.4f}')
	train_starts = torch.tensor(nearest_sites(pts, scene.centers.cpu().numpy()), device='cuda')
	test_starts = nearest_sites(pts, test_scene.centers.cpu().numpy())

	dev = 'cuda'
	model = VoronoiGaussField(torch.tensor(pts, dtype=torch.float32, device=dev), torch.tensor(offsets, device=dev),
							  torch.tensor(adj, device=dev), torch.tensor(cols, device=dev),
							  torch.tensor(spacing, dtype=torch.float32, device=dev), init_alpha=args.init_alpha)
	n_params = sum(p.numel() for p in model.parameters())
	log(f'{n_params / 1e6:.2f} M parameters ({n_params // model.n_sites} per site)')

	if args.sampler == 'nerf':
		kw = dict(renderer=render_rays, n_coarse=args.n_coarse, n_fine=tuple(args.n_fine),
				  proposal_mult=tuple(args.proposal_mult), far_radius=args.far_radius)
	elif args.sampler == 'uniform':
		kw = dict(renderer=render_rays_uniform, dt=args.dt, n_bg=args.n_bg, far_radius=args.far_radius,
				  t_thresh=args.t_thresh)
	else:
		kw = dict(renderer=render_rays_foam, far_radius=args.far_radius, t_thresh=args.t_thresh)
	renderer = kw['renderer']
	train_kw = {k: v for k, v in kw.items() if k != 'renderer'}
	opt = torch.optim.Adam(model.param_groups({k: v * args.lr_mult for k, v in LR.items()}), eps=1e-15, fused=True)
	for g in opt.param_groups:
		g['base_lr'] = g['lr']
	stats, history, vals, mean = {}, [], [], {}
	t_start = t_last = time.time()
	for it in range(1, args.iters + 1):
		f = lr_factor(it, args.iters, args.warmup)
		for g in opt.param_groups:
			g['lr'] = g['base_lr'] * f
		if it % args.sh_every == 0 and model.sh_deg < model.max_sh_deg:
			model.sh_deg += 1

		o, d, rgb_gt, img = scene.random_rays(args.batch)
		out = renderer(model, o, d, train_starts[img], perturb=True, **train_kw)
		loss = ((out['rgb'] - rgb_gt) ** 2).mean()

		opt.zero_grad(set_to_none=True)
		loss.backward()
		opt.step()

		for name, v in (('loss', loss), ('acc', out['acc'].mean()), ('walk', out['walk_steps']),
						('samples', out.get('samples', 0.0)), ('active', out['active'])):
			stats[name] = stats.get(name, 0.0) + float(v.detach() if torch.is_tensor(v) else v)
		if it % args.log_every == 0:
			torch.cuda.synchronize()
			now = time.time()
			n = args.log_every
			row = {k: v / n for k, v in stats.items()}
			row.update(it=it, psnr=-10 * math.log10(row['loss']), sec_per_it=(now - t_last) / n,
					   elapsed=now - t_start, mem_gb=torch.cuda.max_memory_allocated() / 2**30, host_gb=host_peak_gb())
			history.append(row)
			log(f'it {it:6d} | mse {row["loss"]:.5f} psnr {row["psnr"]:.2f} | acc {row["acc"]:.3f} | sh {model.sh_deg} '
				f'| samples/ray {row["samples"]:.0f} | walk steps/sample {row["walk"]:.2f} | '
				f'color on {100 * row["active"]:.1f}% samples | '
				f'{1000 * row["sec_per_it"]:.0f} ms/it | GPU {row["mem_gb"]:.2f} GB, RAM {row["host_gb"]:.2f} GB')
			stats, t_last = {}, now
		if it % args.val_every == 0 or it == args.iters:
			vals = []
			for v in args.val_views:
				rgb, gt, m = evaluate(model, test_scene, test_starts, v, kw)
				to_png(torch.cat([gt, rgb], 1), os.path.join(args.out, f'val_{it:06d}_{v:03d}.png'))
				vals.append(m | {'view': v})
			mean = {k: float(np.mean([x[k] for x in vals])) for k in ('psnr', 'ssim')}
			log(f'  val @ {it}: PSNR {mean["psnr"]:.2f} SSIM {mean["ssim"]:.4f}')
			t_last = time.time()

	train_sec = time.time() - t_start
	log(f'training: {train_sec / 60:.1f} min ({1000 * train_sec / args.iters:.0f} ms/it incl. validation)')
	torch.save({'state': model.state_dict(), 'points': model.graph.points, 'offsets': model.graph.offsets,
				'adj': model.graph.adj, 'sh_deg': model.sh_deg, 'normalization': scene.normalization,
				'args': vars(args)}, os.path.join(args.out, 'model.pth'))

	t0 = time.time()
	per_view = []
	for v in range(test_scene.n):
		rgb, gt, m = evaluate(model, test_scene, test_starts, v, kw)
		per_view.append(m | {'view': v, 'name': test_scene.names[v]})
		if v % 6 == 0:
			to_png(torch.cat([gt, rgb], 1), os.path.join(args.out, f'test_{v:03d}.png'))
	test_eval = {'n_views': test_scene.n, 'resolution': [test_scene.W, test_scene.H],
				 'psnr': float(np.mean([x['psnr'] for x in per_view])),
				 'ssim': float(np.mean([x['ssim'] for x in per_view])),
				 'render_sec_per_view': (time.time() - t0) / test_scene.n, 'per_view': per_view}
	log(f'test set ({test_scene.n} views, {test_scene.W}x{test_scene.H}): PSNR {test_eval["psnr"]:.2f} dB, '
		f'SSIM {test_eval["ssim"]:.4f} ({test_eval["render_sec_per_view"]:.2f} s/view)')
	with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
		json.dump({'args': vars(args), 'train_minutes': train_sec / 60, 'n_sites': model.n_sites,
				   'params_million': n_params / 1e6, 'val': vals, 'val_mean': mean, 'history': history,
				   'test': test_eval}, fh, indent=1)


if __name__ == '__main__':
	main()
