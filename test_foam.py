"""Checks interaction-point rendering: the traversal (every segment lies in the Voronoi cell it is attributed to)
and the values / gradients against a float64 PyTorch reference that re-traces the rays independently
(thresholds disabled so every segment counts)."""

from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial import cKDTree

import ext
from render import pack_foam, ray_bounds, render_rays_foam, render_rays_foam_infer
from sites import delaunay_graph
from test_grad import neighborhood_mask, sh_basis
from test_march import mahalanobis6


def trace(pts, offsets, adj, o, d, near, far, cell):
	"""-> sample parameters t (S,) and the cell evaluating each sample (S,)."""
	t, cells = [near], [cell]
	while True:
		js = adj[offsets[cell]:offsets[cell + 1]]
		n = pts[js] - pts[cell]
		dn = n @ d
		ahead = dn > 0
		if not ahead.any():
			break
		tj = ((0.5 * (pts[js] + pts[cell]) - o) * n).sum(1)[ahead] / dn[ahead]
		k = np.argmin(tj)
		if tj[k] >= far:
			break
		cell = js[ahead][k]
		t.append(max(tj[k], t[-1]))
		cells.append(cell)
	t.append(far)
	cells.append(cells[-1])
	return np.array(t), np.array(cells)


def main():
	torch.manual_seed(0)
	rng = np.random.default_rng(2)
	pts, _, offsets, adj = delaunay_graph(rng.uniform(-0.9, 0.9, (300, 3)))
	n = len(pts)
	dev = 'cuda'
	graph = ext.Graph(torch.tensor(pts, device=dev), torch.tensor(offsets, device=dev), torch.tensor(adj, device=dev))

	raw = (torch.randn(n, device=dev) * 0.5 + 1.0).requires_grad_()
	ls = (torch.randn(n, 3, device=dev) * 0.3 - 1.8).requires_grad_()
	q = torch.randn(n, 4, device=dev).requires_grad_()
	cls = (torch.randn(n, 3, device=dev) * 0.3 - 1.8).requires_grad_()
	cq = torch.randn(n, 4, device=dev).requires_grad_()
	sh = (torch.randn(n, 16, 3, device=dev) * 0.3).requires_grad_()
	B = 64
	o = torch.tensor(rng.uniform(-0.5, 0.5, (B, 3)), dtype=torch.float32, device=dev)
	d = torch.nn.functional.normalize(torch.randn(B, 3, device=dev), dim=-1)
	tree = cKDTree(pts)
	start = torch.tensor(tree.query(o.cpu().numpy())[1], dtype=torch.int32, device=dev)
	near, _, far = ray_bounds(o, d, 0.02, 2.0)
	g_rgb, g_T = torch.randn(B, 3, device=dev), torch.randn(B, device=dev)

	def run(record_cap):
		dA, cA = ext.conic(ls, q), ext.conic(cls, cq)
		dA.retain_grad(), cA.retain_grad()
		fg, T, _, stats = ext.foam(graph, raw, dA, cA, sh, 3, o, d, start, near, far, t_thresh=0.0, w_thresh=-1.0,
								   record_cap=record_cap)
		((fg * g_rgb).sum() + (T * g_T).sum()).backward()
		grads = [raw.grad.clone(), dA.grad.clone(), cA.grad.clone(), sh.grad.clone()]
		for p in (raw, ls, q, cls, cq, sh):
			p.grad = None
		return fg.detach(), T.detach(), stats, grads, dA.detach(), cA.detach()

	runs = {cap: run(cap) for cap in (256, 0, 6)}
	fg, T, stats, _, dA, cA = runs[256]

	P = torch.tensor(pts, dtype=torch.float64)
	ref = [t.detach().cpu().double().requires_grad_() for t in (raw, dA, cA, sh)]
	rraw, rdA, rcA, rsh = ref
	fg_ref, T_ref, n_samples, bad_segments, n_segments = [], [], [], 0, 0
	o64, d64 = o.cpu().double().numpy(), d.cpu().double().numpy()
	for r in range(B):
		x0 = o64[r] + near[r].item() * d64[r]
		t, cells = trace(pts, offsets, adj, o64[r], d64[r], near[r].item(), far[r].item(), tree.query(x0)[1])
		n_samples.append(len(t))
		mids = o64[r] + 0.5 * (t[1:] + t[:-1])[:, None] * d64[r]
		long_enough = t[1:] - t[:-1] > 1e-6
		bad_segments += int((tree.query(mids)[1] != cells[:-1])[long_enough].sum())
		n_segments += int(long_enough.sum())

		X = torch.tensor(o64[r][None] + t[:, None] * d64[r][None])
		mask = neighborhood_mask(offsets, adj, torch.tensor(cells), n)
		sigma = (torch.exp(rraw)[None] * torch.exp(-0.5 * mahalanobis6(P, rdA, X)) * mask).sum(1)
		a = torch.softmax((-0.5 * mahalanobis6(P, rcA, X)).masked_fill(~mask, -float('inf')), 1)
		D = torch.tensor(d64[r])
		ck = (torch.einsum('j,njc->nc', sh_basis(D), rsh) + 0.5).clamp_min(0)
		C = a @ ck
		delta = torch.tensor(t[1:] - t[:-1])
		alpha = 1 - torch.exp(-0.5 * (sigma[1:] + sigma[:-1]) * delta)
		trans = torch.cumprod(torch.cat([torch.ones(1, dtype=torch.float64), 1 - alpha]), 0)
		w = alpha * trans[:-1]
		fg_ref.append((w[:, None] * 0.5 * (C[1:] + C[:-1])).sum(0))
		T_ref.append(trans[-1])
	fg_ref, T_ref = torch.stack(fg_ref), torch.stack(T_ref)
	((fg_ref * g_rgb.cpu().double()).sum() + (T_ref * g_T.cpu().double()).sum()).backward()

	def rel(a, b):
		return ((a.detach().cpu().double() - b.detach()).norm() / b.detach().norm().clamp_min(1e-30)).item()

	same = (stats[:, 0].cpu().numpy() == np.array(n_samples)).mean()
	print(f'samples/ray {stats[:, 0].float().mean().item():.1f} (reference {np.mean(n_samples):.1f}, same count on '
		  f'{100 * same:.0f}% of rays) | segments outside their cell: {bad_segments} / {n_segments} | '
		  f'mean T {T.mean().item():.3f}')
	print(f'rgb rel err {rel(fg, fg_ref):.2e}   T rel err {rel(T, T_ref):.2e}')
	for cap, label in ((256, 'replay all rays'), (0, 're-trace all rays'), (6, 'replay <= 6 samples, re-trace rest')):
		grads = runs[cap][3]
		errs = '  '.join(f'{name} {rel(gc, r.grad):.2e}' for name, gc, r in
						 zip(['raw', 'density conic', 'color conic', 'sh'], grads, ref))
		print(f'grad rel err ({label:34s}): {errs}')

	# inference kernel (foam_infer.cu) against the training kernel, exact options and Z-order renumbering
	model = SimpleNamespace(graph=graph, raw=raw.detach(), d_log_scale=ls.detach(), d_quat=q.detach(),
							c_log_scale=cls.detach(), c_quat=cq.detach(), sh=lambda: sh.detach(), sh_deg=3,
							bg_logit=torch.zeros(3, device=dev))
	kw = dict(far_radius=2.0, t_thresh=0.0, w_thresh=-1.0)
	with torch.no_grad():
		train = render_rays_foam(model, o, d, start, **kw)['rgb']
		for colors in ('exact', 'fused'):
			for reorder in (False, True):
				out = render_rays_foam_infer(model, pack_foam(model, reorder=reorder), o, d, start, colors=colors,
											 **kw)
				print(f'inference kernel, {colors} colors, reorder {reorder!s:5s}: max |rgb - training kernel| '
					  f'{(out["rgb"] - train).abs().max().item():.1e}')


if __name__ == '__main__':
	main()
