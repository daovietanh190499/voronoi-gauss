"""Checks the fused uniform-step march (values and gradients) against a float64 PyTorch reference that uses the
same samples, exact nearest neighbors and NeRF compositing (thresholds disabled so every sample counts)."""

import numpy as np
import torch
from scipy.spatial import cKDTree

import ext
from render import ray_bounds
from sites import delaunay_graph
from test_grad import neighborhood_mask, sh_basis


def samples(near, mid, far, dt, n_bg):
	"""Midpoints and lengths of the march steps of one ray (same formulas as march.cu)."""
	n_in = int(np.ceil((mid - near) / dt)) if mid > near else 0
	t0 = near + dt * np.arange(n_in)
	t1 = np.minimum(t0 + dt, mid)
	u = np.arange(n_bg + 1) / n_bg
	tb = 1.0 / (1.0 / mid + u * (1.0 / far - 1.0 / mid))
	t = np.concatenate([0.5 * (t0 + t1), 1.0 / (1.0 / mid + 0.5 * (u[:-1] + u[1:]) * (1.0 / far - 1.0 / mid))])
	return t, np.concatenate([t1 - t0, tb[1:] - tb[:-1]])


def mahalanobis6(points, A, x):
	v = x[:, None] - points[None]
	a = A[None]
	return (a[..., 0] * v[..., 0] ** 2 + a[..., 3] * v[..., 1] ** 2 + a[..., 5] * v[..., 2] ** 2 +
			2 * (a[..., 1] * v[..., 0] * v[..., 1] + a[..., 2] * v[..., 0] * v[..., 2] + a[..., 4] * v[..., 1] * v[..., 2]))


def main():
	torch.manual_seed(0)
	rng = np.random.default_rng(1)
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
	dA, cA = ext.conic(ls, q, 0.01), ext.conic(cls, cq, 0.01)
	dA.retain_grad(), cA.retain_grad()

	B, dt, n_bg = 64, 0.05, 8
	o = torch.tensor(rng.uniform(-0.5, 0.5, (B, 3)), dtype=torch.float32, device=dev)
	d = torch.nn.functional.normalize(torch.randn(B, 3, device=dev), dim=-1)
	start = torch.tensor(cKDTree(pts).query(o.cpu().numpy())[1], dtype=torch.int32, device=dev)
	near, mid, far = ray_bounds(o, d, 0.02, 2.0)
	fg, T, _, stats = ext.march(graph, raw, dA, cA, sh, 3, o, d, start, near, mid, far, dt, n_bg, t_thresh=0.0,
								w_thresh=-1.0)
	g_rgb, g_T = torch.randn(B, 3, device=dev), torch.randn(B, device=dev)
	((fg * g_rgb).sum() + (T * g_T).sum()).backward()

	P = torch.tensor(pts, dtype=torch.float64)
	ref = [t.detach().cpu().double().requires_grad_() for t in (raw, dA, cA, sh)]
	rraw, rdA, rcA, rsh = ref
	fg_ref, T_ref = [], []
	tree = cKDTree(pts)
	for r in range(B):
		t, delta = samples(near[r].item(), mid[r].item(), far[r].item(), dt, n_bg)
		O, D = o[r].cpu().double(), d[r].cpu().double()
		X = O[None] + torch.tensor(t)[:, None] * D[None]
		site = torch.tensor(tree.query(X.numpy())[1])
		mask = neighborhood_mask(offsets, adj, site, n)
		sigma = (torch.exp(rraw)[None] * torch.exp(-0.5 * mahalanobis6(P, rdA, X)) * mask).sum(1)
		alpha = 1 - torch.exp(-sigma * torch.tensor(delta))
		trans = torch.cumprod(torch.cat([torch.ones(1, dtype=torch.float64), 1 - alpha]), 0)
		w = alpha * trans[:-1]
		a = torch.softmax((-0.5 * mahalanobis6(P, rcA, X)).masked_fill(~mask, -float('inf')), 1)
		ck = (torch.einsum('j,njc->nc', sh_basis(D), rsh) + 0.5).clamp_min(0)
		fg_ref.append((w[:, None] * (a @ ck)).sum(0))
		T_ref.append(trans[-1])
	fg_ref, T_ref = torch.stack(fg_ref), torch.stack(T_ref)
	((fg_ref * g_rgb.cpu().double()).sum() + (T_ref * g_T.cpu().double()).sum()).backward()

	def rel(a, b):
		return ((a.detach().cpu().double() - b.detach()).norm() / b.detach().norm().clamp_min(1e-30)).item()

	print(f'samples/ray {stats[:, 0].float().mean().item():.1f}, walk steps/sample '
		  f'{(stats[:, 2].sum() / stats[:, 0].sum()).item():.2f}, mean T {T.mean().item():.3f}')
	print(f'rgb rel err {rel(fg, fg_ref):.2e}   T rel err {rel(T, T_ref):.2e}')
	for name, gc, r in zip(['raw', 'density conic', 'color conic', 'sh'], [raw.grad, dA.grad, cA.grad, sh.grad], ref):
		print(f'grad {name:14s} rel err {rel(gc, r.grad):.2e}')


if __name__ == '__main__':
	main()
