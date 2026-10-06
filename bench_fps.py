"""Inference speed of a trained Voronoi-Gaussian field: full-image FPS on the test views, with a per-stage
breakdown (CUDA events) and PSNR for several sample budgets."""

import argparse
import json
import os
import time

import numpy as np
import torch

import ext
import render
from colmap import ColmapScene
from model import VoronoiGaussField
from sites import nearest_sites


def load_model(path):
	ck = torch.load(path, map_location='cuda', weights_only=False)
	pts = ck['points']
	n = len(pts)
	model = VoronoiGaussField(pts, ck['offsets'], ck['adj'], torch.zeros(n, 3, device='cuda'),
							  torch.ones(n, device='cuda'))
	model.load_state_dict(ck['state'])
	model.sh_deg = ck['sh_deg']
	model.requires_grad_(False)
	return model, ck


class Timer:
	def __init__(self):
		self.events, self.totals = [], {}

	def mark(self, name):
		e = torch.cuda.Event(enable_timing=True)
		e.record()
		self.events.append((name, e))

	def flush(self):
		torch.cuda.synchronize()
		for (_, a), (name, b) in zip(self.events[:-1], self.events[1:]):
			self.totals[name] = self.totals.get(name, 0.0) + a.elapsed_time(b)
		self.events = []


def render_staged(model, o, d, start, n_coarse, n_fine, proposal_mult, far_radius, timer, near=0.02,
				  min_weight=1e-4):
	"""render.render_rays (perturb=False) split into timed stages."""
	B = len(o)
	timer.mark('start')
	bounds = render.ray_bounds(o, d, near, far_radius)
	s = torch.linspace(0, 1, n_coarse + 1, device=o.device)[:-1].expand(B, n_coarse) + 0.5 / n_coarse
	for n, mult in zip(n_fine, proposal_mult):
		w = render.proposal_weights(model, o, d, s, bounds, start, mult)
		s_new = render.resample(s, w, n, det=True)
		s, _ = torch.sort(torch.cat([s, s_new], 1), 1)
	timer.mark('proposals (walk + density + resample)')
	t = render.s_to_t(s, *bounds)
	site, _ = model.walk(o, d, t, start)
	timer.mark('final walk')
	S = t.shape[1]
	x = (o[:, None] + d[:, None] * t[..., None]).reshape(-1, 3)
	site = site.reshape(-1)
	sigma = ext.density(model.graph, model.raw, model.d_log_scale, model.d_quat, x, site).view(B, S)
	timer.mark('final density')
	w, t_final = render.composite(sigma, t, bounds[2])
	active = (w > min_weight).reshape(-1).nonzero().squeeze(1)
	timer.mark('composite + select')
	dirs = d[:, None].expand(B, S, 3).reshape(-1, 3)
	c = ext.color(model.graph, model.c_log_scale, model.c_quat, model.sh(), x[active], dirs[active], site[active],
				  model.sh_deg)
	timer.mark('color')
	rgb_s = torch.zeros(B * S, 3, device=o.device).index_put((active,), c).view(B, S, 3)
	rgb = (w[..., None] * rgb_s).sum(1) + t_final[:, None] * torch.sigmoid(model.bg_logit)
	timer.mark('blend')
	return rgb, len(active) / (B * S)


@torch.inference_mode()
def bench(model, scene, starts, views, cfg, chunk, far_radius, reps):
	rays = [scene.image_rays(i) for i in views]
	starts = [starts[i] for i in views]
	kw = dict(n_coarse=cfg[0], n_fine=(cfg[1], cfg[2]), proposal_mult=(4.0, 2.0), far_radius=far_radius)
	o, d, _ = rays[0]
	st = torch.full((len(o),), int(starts[0]), dtype=torch.int32, device='cuda')
	for j in range(0, len(o), chunk):  # warm-up
		render.render_rays(model, o[j:j + chunk], d[j:j + chunk], st[j:j + chunk], perturb=False, **kw)
	torch.cuda.synchronize()

	times, psnrs = [], []
	for _ in range(reps):
		for i, (o, d, gt) in enumerate(rays):
			st = torch.full((len(o),), int(starts[i]), dtype=torch.int32, device='cuda')
			torch.cuda.synchronize()
			t0 = time.perf_counter()
			out = [render.render_rays(model, o[j:j + chunk], d[j:j + chunk], st[j:j + chunk], perturb=False,
									  **kw)['rgb'] for j in range(0, len(o), chunk)]
			rgb = torch.cat(out).view(scene.H, scene.W, 3).clamp(0, 1)
			torch.cuda.synchronize()
			times.append(time.perf_counter() - t0)
			if len(psnrs) < len(rays):
				psnrs.append(float(-10 * torch.log10(((rgb - gt) ** 2).mean())))

	timer, active = Timer(), []
	for i, (o, d, _) in enumerate(rays):
		st = torch.full((len(o),), int(starts[i]), dtype=torch.int32, device='cuda')
		for j in range(0, len(o), chunk):
			_, a = render_staged(model, o[j:j + chunk], d[j:j + chunk], st[j:j + chunk], *kw.values(), timer)
			active.append(a)
			timer.flush()
	stages = {k: v / len(rays) for k, v in timer.totals.items()}
	ms = 1000 * float(np.median(times))
	return {'samples': sum(cfg), 'config': cfg, 'chunk': chunk, 'ms_per_frame': ms, 'fps': 1000 / ms,
			'psnr': float(np.mean(psnrs)), 'color_fraction': float(np.mean(active)), 'stage_ms': stages,
			'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2**30}


@torch.inference_mode()
def bench_fused(model, scene, starts, views, renderer, kw, chunk):
	"""FPS of a fused single-kernel renderer (uniform steps or interaction points)."""
	rays = [scene.image_rays(i) for i in views]
	o, d, _ = rays[0]
	st = torch.full((chunk,), int(starts[views[0]]), dtype=torch.int32, device='cuda')
	renderer(model, o[:chunk], d[:chunk], st[:len(o[:chunk])], **kw)  # warm-up
	times, psnrs, samples, active = [], [], [], []
	for v, (o, d, gt) in zip(views, rays):
		st = torch.full((len(o),), int(starts[v]), dtype=torch.int32, device='cuda')
		torch.cuda.synchronize()
		t0 = time.perf_counter()
		outs = [renderer(model, o[j:j + chunk], d[j:j + chunk], st[j:j + chunk], **kw)
				for j in range(0, len(o), chunk)]
		rgb = torch.cat([x['rgb'] for x in outs]).view(scene.H, scene.W, 3).clamp(0, 1)
		torch.cuda.synchronize()
		times.append(time.perf_counter() - t0)
		psnrs.append(float(-10 * torch.log10(((rgb - gt) ** 2).mean())))
		samples.append(float(np.mean([float(x['samples']) for x in outs])))
		active.append(float(np.mean([x['active'] for x in outs])))
	ms = 1000 * float(np.median(times))
	return {'t_thresh': kw['t_thresh'], 'chunk': chunk, 'ms_per_frame': ms, 'fps': 1000 / ms,
			'psnr': float(np.mean(psnrs)), 'samples_per_ray': float(np.mean(samples)),
			'color_fraction': float(np.mean(active)), 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2**30}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--exp', default='exp/bonsai')
	ap.add_argument('--chunk', type=int, nargs='*', default=[65536])
	ap.add_argument('--reps', type=int, default=1)
	ap.add_argument('--views', type=int, nargs='*', default=[0, 6, 12, 18, 24, 30])
	args = ap.parse_args()

	model, ck = load_model(os.path.join(args.exp, 'model.pth'))
	targs = ck['args']
	scene = ColmapScene(targs['data'], 'test', targs['downsample'], normalization=ck['normalization'])
	starts = nearest_sites(model.graph.points.cpu().numpy(), scene.centers.cpu().numpy())
	print(f'{model.n_sites} sites, {scene.n} test views at {scene.W}x{scene.H} ({scene.W * scene.H} rays), '
		  f'{torch.cuda.get_device_name()}', flush=True)

	results = []
	sampler = targs.get('sampler', 'nerf')
	if sampler in ('uniform', 'foam'):
		for t_thresh in (targs['t_thresh'], 1e-3, 1e-2):
			if sampler == 'uniform':
				renderer, label = render.render_rays_uniform, f'uniform dt {targs["dt"]:.5f}'
				kw = dict(dt=targs['dt'], n_bg=targs['n_bg'], far_radius=targs['far_radius'], t_thresh=t_thresh)
			else:
				renderer, label = render.render_rays_foam, 'interaction points'
				kw = dict(far_radius=targs['far_radius'], t_thresh=t_thresh)
			for chunk in args.chunk:
				torch.cuda.reset_peak_memory_stats()
				r = bench_fused(model, scene, starts, args.views, renderer, kw, chunk) | {'sampler': sampler}
				results.append(r)
				print(f'{label} stop T<{t_thresh:g} chunk {chunk}: {r["ms_per_frame"]:7.1f} '
					  f'ms/frame = {r["fps"]:5.2f} FPS | PSNR {r["psnr"]:.2f} | samples/ray {r["samples_per_ray"]:.0f} '
					  f'| color on {100 * r["color_fraction"]:.1f}% | GPU {r["peak_gpu_gb"]:.2f} GB', flush=True)
		with open(os.path.join(args.exp, 'fps.json'), 'w') as fh:
			json.dump(results, fh, indent=1)
		return
	for cfg in [(64, 64, 64), (48, 40, 40), (32, 32, 32), (24, 20, 20)]:
		for chunk in args.chunk:
			torch.cuda.reset_peak_memory_stats()
			r = bench(model, scene, starts, args.views, cfg, chunk, targs['far_radius'], args.reps)
			results.append(r)
			stages = ' | '.join(f'{k} {v:.1f}' for k, v in r['stage_ms'].items())
			print(f'{r["samples"]:3d} samples {cfg} chunk {chunk:6d}: {r["ms_per_frame"]:7.1f} ms/frame = '
				  f'{r["fps"]:5.2f} FPS | PSNR {r["psnr"]:.2f} | color on {100 * r["color_fraction"]:.1f}% | '
				  f'GPU {r["peak_gpu_gb"]:.2f} GB', flush=True)
			print(f'      stage ms/frame: {stages}', flush=True)
	with open(os.path.join(args.exp, 'fps.json'), 'w') as fh:
		json.dump(results, fh, indent=1)


if __name__ == '__main__':
	main()
