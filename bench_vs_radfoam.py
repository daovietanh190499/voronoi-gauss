"""Inference FPS of the Voronoi-Gaussian field (interaction points) vs RadFoam (voronoi_utils renderer) under one
protocol: the same test views (every 8th test camera), resolutions and early-termination thresholds T < t.

The two code bases are built against different PyTorch installs, so each method runs in its own process:
  ../points-as-tori/.venv/bin/python bench_vs_radfoam.py --method vgf
  PYTHONPATH=/mnt/data/research:/mnt/data/research/voronoi_utils/cuda/_ext \
    python3 bench_vs_radfoam.py --method radfoam
  python3 bench_vs_radfoam.py --method report
"""

import argparse
import json
import os
import sys
import time
from argparse import Namespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'exp', 'fps_vs_radfoam')
DATA = '/mnt/data/research/radfoam/data/mipnerf360'


def time_ms(fn, warmup=1, reps=3):
	for _ in range(warmup):
		fn()
	torch.cuda.synchronize()
	a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
	a.record()
	for _ in range(reps):
		fn()
	b.record()
	torch.cuda.synchronize()
	return a.elapsed_time(b) / reps


def psnr(a, b):
	return float(-10 * torch.log10(((a.clamp(0, 1) - b) ** 2).mean().clamp_min(1e-10)))


@torch.no_grad()
def bench_vgf(args):
	sys.path.insert(0, HERE)
	from bench_fps import load_model
	from bench_infer import tile_order
	from colmap import ColmapScene
	from render import pack_foam, render_rays_foam, render_rays_foam_infer
	from sites import nearest_sites

	model, ck = load_model(os.path.join(HERE, args.vgf_exp, 'model.pth'))
	far_radius = ck['args']['far_radius']
	pack = pack_foam(model, sh_half=True, eps_density=1e-4, reorder=True)
	rows = []
	for ds in args.downsample:
		scene = ColmapScene(os.path.join(DATA, 'bonsai'), 'test', ds, normalization=ck['normalization'])
		starts = nearest_sites(model.graph.points.cpu().numpy(), scene.centers.cpu().numpy())
		perm = tile_order(scene.H, scene.W)
		for t in args.thresholds:
			for method in ('vgf', 'vgf-fast'):
				ms, ps, samples = [], [], []
				for v in range(0, scene.n, 8):
					o, d, gt = scene.image_rays(v)
					st = torch.full((len(o),), int(starts[v]), dtype=torch.int32, device='cuda')
					if method == 'vgf':
						run = lambda: render_rays_foam(model, o, d, st, far_radius=far_radius, t_thresh=t)
						out = run()
						rgb, n = out['rgb'], float(out['samples'])
					else:
						oo, dd = o[perm].contiguous(), d[perm].contiguous()
						run = lambda: render_rays_foam_infer(model, pack, oo, dd, st, far_radius=far_radius, t_thresh=t,
															 colors='fused', eps_color=1e-4)
						out = run()
						rgb = torch.empty_like(out['rgb'])
						rgb[perm] = out['rgb']
						n = float(out['stats'][:, 0].float().mean())
					ms.append(time_ms(run, args.warmup, args.reps))
					ps.append(psnr(rgb.view(scene.H, scene.W, 3), gt))
					samples.append(n)
				kernel = ('interaction points (fp32, warp/ray)' if method == 'vgf' else
						  'interaction points, inference kernel (fused colors, fp16 SH, eps 1e-4)')
				rows.append({'method': method, 'kernel': kernel, 'res': f'{scene.W}x{scene.H}', 'ds': ds, 't_thresh': t,
							 'ms': float(np.mean(ms)), 'fps': 1000 / float(np.mean(ms)), 'psnr': float(np.mean(ps)),
							 'steps_per_ray': float(np.mean(samples)), 'n_sites': model.n_sites})
				print(json.dumps(rows[-1]), flush=True)
	return rows


@torch.no_grad()
def bench_radfoam(args):
	sys.path.insert(0, '/mnt/data/research')
	from voronoi_utils.common.colmap_data import DataHandler
	from voronoi_utils.common.io_model_pt import load_radfoam_model_pt
	from voronoi_utils.common.starts import starting_points_from_rays
	from voronoi_utils.cuda.Voronoi import Voronoi
	from voronoi_utils.cuda.raytrace import build_adjacent_offsets, nearest_sites, raytrace, trace_benchmark
	sys.path.insert(0, '/mnt/data/research/voronoi_utils/benchmarks')
	from bench_demo_all import build_cameras

	dev = torch.device('cuda')
	ck = load_radfoam_model_pt(args.radfoam_ckpt, device=dev, activation_scale=1.0)
	v = Voronoi.from_radfoam_checkpoint(ck)
	adj4 = build_adjacent_offsets(v.xyz, v.adjacency)
	rows = []
	for ds in args.downsample:
		h = DataHandler(Namespace(dataset='colmap', data_path=DATA, scene='bonsai', downsample=[ds],
								  downsample_iterations=[0], patch_based=False), rays_per_batch=0, device=dev)
		h.reload(split='test', downsample=ds)
		W, H = int(h.img_wh[0]), int(h.img_wh[1])
		views = list(range(0, int(h.rays.shape[0]), 8))
		for t in args.thresholds:
			ms, ps, steps = [], [], []
			for vi in views:
				rays = h.rays[vi].reshape(-1, 6).contiguous()
				gt = h.rgbs[vi].reshape(H, W, 3)
				start = starting_points_from_rays(v.xyz, None, rays).reshape(-1)
				run = lambda: raytrace(v, rays, adjacency=v.adjacency, start_idx=start, adjacent_offsets=adj4,
									   weight_threshold=t, max_steps=1024, white_bg=False)
				out = run()
				ms.append(time_ms(run, args.warmup, args.reps))
				rgb = (out.rgba[..., :3] + (1 - out.rgba[..., 3:4])).reshape(H, W, 3)
				ps.append(psnr(rgb, gt))
				steps.append(float(out.num_intersections.float().mean()))
			row = {'method': 'radfoam', 'kernel': 'raytrace (fp32 attrs, thread/ray)', 'res': f'{W}x{H}', 'ds': ds,
				   't_thresh': t, 'ms': float(np.mean(ms)), 'fps': 1000 / float(np.mean(ms)),
				   'psnr': float(np.mean(ps)), 'steps_per_ray': float(np.mean(steps)), 'n_sites': int(v.num_sites)}
			cams, pos = build_cameras(h.c2ws, W, H, float(h.fy))
			start_pts = nearest_sites(v.xyz, pos.to(dev)).long()
			row['fps_native_camera_kernel'] = float(trace_benchmark(
				v, cams, start_pts, adjacency=v.adjacency, adjacent_offsets=adj4, weight_threshold=t, max_steps=1024,
				n_reps=5, warmup=True))
			rows.append(row)
			print(json.dumps(row), flush=True)
	return rows


def report():
	rows = []
	for m in ('vgf', 'radfoam'):
		with open(os.path.join(OUT, f'{m}.json')) as fh:
			rows += json.load(fh)
	print(f'{"method":9s} {"res":8s} {"T<":>6s} {"ms/frame":>9s} {"FPS":>7s} {"steps/ray":>9s} {"PSNR":>6s}  native FPS')
	for r in sorted(rows, key=lambda r: (r['ds'], r['t_thresh'], r['method'])):
		nat = f'{r["fps_native_camera_kernel"]:.1f}' if 'fps_native_camera_kernel' in r else '-'
		print(f'{r["method"]:9s} {r["res"]:8s} {r["t_thresh"]:6g} {r["ms"]:9.1f} {r["fps"]:7.2f} '
			  f'{r["steps_per_ray"]:9.1f} {r["psnr"]:6.2f}  {nat}')


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--method', choices=['vgf', 'radfoam', 'report'], required=True)
	ap.add_argument('--vgf_exp', default='exp/bonsai_foam')
	ap.add_argument('--radfoam_ckpt',
					default='/mnt/data/research/voronoi_utils/example/radfoam/output/bonsai_1k_retrain_20261006/model.pt')
	ap.add_argument('--downsample', type=int, nargs='*', default=[8, 4])
	ap.add_argument('--thresholds', type=float, nargs='*', default=[1e-3, 0.05])
	ap.add_argument('--warmup', type=int, default=1)
	ap.add_argument('--reps', type=int, default=3)
	args = ap.parse_args()
	if args.method == 'report':
		return report()
	os.makedirs(OUT, exist_ok=True)
	rows = bench_vgf(args) if args.method == 'vgf' else bench_radfoam(args)
	with open(os.path.join(OUT, f'{args.method}.json'), 'w') as fh:
		json.dump(rows, fh, indent=1)


if __name__ == '__main__':
	main()
