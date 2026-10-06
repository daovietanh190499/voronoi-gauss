"""Per-phase timing of one training step with interaction-point rendering (CUDA events, averaged), plus the top
CUDA kernels from torch.profiler."""

import argparse
import os

import torch

import ext
from bench_fps import load_model
from colmap import ColmapScene
from render import render_rays_foam
from sites import nearest_sites

LR = {'raw': 2e-2, 'scale': 5e-3, 'quat': 2e-3, 'sh_dc': 5e-3, 'sh_rest': 2.5e-4, 'bg': 1e-2}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--exp', default='exp/bonsai_foam')
	ap.add_argument('--batch', type=int, default=4096)
	ap.add_argument('--iters', type=int, default=50)
	args = ap.parse_args()

	model, ck = load_model(os.path.join(args.exp, 'model.pth'))
	model.requires_grad_(True)
	targs = ck['args']
	scene = ColmapScene(targs['data'], 'train', targs['downsample'], normalization=ck['normalization'])
	starts = torch.tensor(nearest_sites(model.graph.points.cpu().numpy(), scene.centers.cpu().numpy()), device='cuda')
	opt = torch.optim.Adam(model.param_groups(LR), eps=1e-15, fused=True)
	trainable = [n for n, p in model.named_parameters() if p.requires_grad]
	print(f'trainable: {trainable}; points require grad: {model.graph.points.requires_grad}')

	names = ['rays', 'conic + sh (torch)', 'foam forward kernel', 'loss', 'backward (all)', 'adam step', 'zero_grad']

	def step(timed):
		ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(names) + 1)]
		ev[0].record()
		o, d, rgb_gt, img = scene.random_rays(args.batch)
		ev[1].record()
		dA = ext.conic(model.d_log_scale, model.d_quat)
		cA = ext.conic(model.c_log_scale, model.c_quat)
		sh = model.sh()
		ev[2].record()
		near_t, _, far_t = __import__('render').ray_bounds(o, d, 0.02, targs['far_radius'])
		fg, T, _, stats = ext.foam(model.graph, model.raw, dA, cA, sh, model.sh_deg, o, d, starts[img], near_t, far_t)
		ev[3].record()
		rgb = fg + T[:, None] * torch.sigmoid(model.bg_logit)
		loss = ((rgb - rgb_gt) ** 2).mean()
		ev[4].record()
		loss.backward()
		ev[5].record()
		opt.step()
		ev[6].record()
		opt.zero_grad(set_to_none=True)
		ev[7].record()
		return ev, stats

	for _ in range(5):
		step(False)
	torch.cuda.synchronize()
	tot = [0.0] * len(names)
	samples = 0.0
	wall = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
	wall[0].record()
	for _ in range(args.iters):
		ev, stats = step(True)
		torch.cuda.synchronize()
		for k in range(len(names)):
			tot[k] += ev[k].elapsed_time(ev[k + 1])
		samples += stats[:, 0].float().mean().item()
	wall[1].record()
	torch.cuda.synchronize()
	print(f'batch {args.batch} rays, {samples / args.iters:.1f} samples/ray, sh_deg {model.sh_deg}')
	for k, n in enumerate(names):
		print(f'  {n:22s} {tot[k] / args.iters:7.2f} ms')
	print(f'  {"sum":22s} {sum(tot) / args.iters:7.2f} ms   (wall with sync {wall[0].elapsed_time(wall[1]) / args.iters:.2f} ms)')

	with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
		for _ in range(10):
			step(False)
		torch.cuda.synchronize()
	print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=12))


if __name__ == '__main__':
	main()
