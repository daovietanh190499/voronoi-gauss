"""Checks the CUDA walk against exact nearest neighbors and the density / color kernels (values and gradients)
against a float64 PyTorch reference on a small random point cloud."""

import numpy as np
import torch
from scipy.spatial import cKDTree

import ext
from sites import delaunay_graph


def quat_rot(q):
	q = q / q.norm(dim=-1, keepdim=True)
	w, x, y, z = q.unbind(-1)
	return torch.stack([
		1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
		2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
		2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(*q.shape[:-1], 3, 3)


def sh_basis(d):
	x, y, z = d.unbind(-1)
	xx, yy, zz = x * x, y * y, z * z
	return torch.stack([
		torch.full_like(x, 0.28209479177387814),
		-0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x,
		1.0925484305920792 * x * y, -1.0925484305920792 * y * z, 0.31539156525252005 * (2 * zz - xx - yy),
		-1.0925484305920792 * x * z, 0.5462742152960396 * (xx - yy),
		-0.5900435899266435 * y * (3 * xx - yy), 2.890611442640554 * x * y * z,
		-0.4570457994644658 * y * (4 * zz - xx - yy), 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
		-0.4570457994644658 * x * (4 * zz - xx - yy), 1.445305721320277 * z * (xx - yy),
		-0.5900435899266435 * x * (xx - 3 * yy)], -1)


def neighborhood_mask(offsets, adj, site, n):
	"""(m, N) bool: k in {site} U adj(site)."""
	mask = torch.zeros(len(site), n, dtype=torch.bool)
	for r, i in enumerate(site.tolist()):
		mask[r, i] = True
		mask[r, adj[offsets[i]:offsets[i + 1]]] = True
	return mask


def mahalanobis(points, log_scale, quat, x):
	R = quat_rot(quat)  # (N, 3, 3)
	v = x[:, None] - points[None]  # (m, N, 3)
	y = torch.einsum('nab,mna->mnb', R, v)
	return (y * y / torch.exp(2 * log_scale)[None]).sum(-1)


def main():
	torch.manual_seed(0)
	rng = np.random.default_rng(0)
	n, m = 400, 3000
	pts = rng.uniform(-1, 1, (n, 3))
	pts, _, offsets, adj = delaunay_graph(pts)
	n = len(pts)
	dev = 'cuda'
	graph = ext.Graph(torch.tensor(pts, device=dev), torch.tensor(offsets, device=dev), torch.tensor(adj, device=dev))

	# walk: rays from random origins, sorted samples
	B, S = 500, 40
	o = torch.tensor(rng.uniform(-1.2, 1.2, (B, 3)), dtype=torch.float32, device=dev)
	d = torch.nn.functional.normalize(torch.randn(B, 3, device=dev), dim=-1)
	z = torch.sort(torch.rand(B, S, device=dev) * 2.5, 1)[0]
	start = torch.tensor(cKDTree(pts).query(o.cpu().numpy())[1], dtype=torch.int32, device=dev)
	site, steps = graph.walk(o, d, z, start)
	x = (o[:, None] + d[:, None] * z[..., None]).reshape(-1, 3)
	dist_walk = np.linalg.norm(x.cpu().numpy() - pts[site.reshape(-1).cpu().numpy()], axis=1)
	dist_nn, _ = cKDTree(pts).query(x.cpu().numpy())
	print(f'walk: {np.mean(dist_walk <= dist_nn + 1e-6) * 100:.3f}% exact NN, '
		  f'max excess {np.max(dist_walk - dist_nn):.2e}, steps/sample {steps.float().mean().item() / S:.2f}')

	# fields
	x = torch.tensor(rng.uniform(-1, 1, (m, 3)), dtype=torch.float32, device=dev)
	site = torch.tensor(cKDTree(pts).query(x.cpu().numpy())[1], dtype=torch.int32, device=dev)
	raw = (torch.randn(n, device=dev) * 0.5).requires_grad_()
	ls = (torch.randn(n, 3, device=dev) * 0.3 - 1.2).requires_grad_()
	q = torch.randn(n, 4, device=dev).requires_grad_()
	cls = (torch.randn(n, 3, device=dev) * 0.3 - 1.2).requires_grad_()
	cq = torch.randn(n, 4, device=dev).requires_grad_()
	sh = (torch.randn(n, 16, 3, device=dev) * 0.3).requires_grad_()
	dirs = torch.randn(m, 3, device=dev)
	g_sigma = torch.randn(m, device=dev)
	g_rgb = torch.randn(m, 3, device=dev)

	sigma = ext.density(graph, raw, ls, q, x, site)
	rgb = ext.color(graph, cls, cq, sh, x, dirs, site, 3)
	((sigma * g_sigma).sum() + (rgb * g_rgb).sum()).backward()
	cuda_grads = [p.grad.clone() for p in (raw, ls, q, cls, cq, sh)]

	P = torch.tensor(pts, dtype=torch.float64)
	ref = [p.detach().cpu().double().requires_grad_() for p in (raw, ls, q, cls, cq, sh)]
	rraw, rls, rq, rcls, rcq, rsh = ref
	X = x.cpu().double()
	mask = neighborhood_mask(offsets, adj, site.cpu(), n)
	md = mahalanobis(P, rls, rq, X)
	sigma_ref = (torch.exp(rraw)[None] * torch.exp(-0.5 * md) * mask).sum(1)
	mc = mahalanobis(P, rcls, rcq, X).masked_fill(~mask, float('inf'))
	a = torch.softmax(-0.5 * mc, 1)
	D = torch.nn.functional.normalize(dirs.cpu().double(), dim=-1)
	ck = (torch.einsum('mj,njc->mnc', sh_basis(D), rsh) + 0.5).clamp_min(0)
	rgb_ref = (a[..., None] * ck).sum(1)
	((sigma_ref * g_sigma.cpu().double()).sum() + (rgb_ref * g_rgb.cpu().double()).sum()).backward()

	def rel(a, b):
		return ((a.cpu().double() - b).norm() / b.norm().clamp_min(1e-30)).item()

	print(f'sigma rel err {rel(sigma.detach(), sigma_ref.detach()):.2e}   rgb rel err {rel(rgb.detach(), rgb_ref.detach()):.2e}')
	for name, gc, r in zip(['raw', 'd_log_scale', 'd_quat', 'c_log_scale', 'c_quat', 'sh'], cuda_grads, ref):
		print(f'grad {name:12s} rel err {rel(gc, r.grad):.2e}')


if __name__ == '__main__':
	main()
