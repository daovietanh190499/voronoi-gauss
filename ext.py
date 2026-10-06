"""JIT-built CUDA extension and autograd wrappers for the Voronoi-neighborhood density and color fields."""

import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_EXTS = {}


def _load(name, source):
	if name not in _EXTS:
		from torch.utils.cpp_extension import load

		major, minor = torch.cuda.get_device_capability()
		os.environ.setdefault('TORCH_CUDA_ARCH_LIST', f'{major}.{minor}')
		build = os.path.join(HERE, 'build', name)
		os.makedirs(build, exist_ok=True)
		_EXTS[name] = load(
			name=name,
			sources=[os.path.join(HERE, 'csrc', source)],
			extra_cuda_cflags=['-O3', '-lineinfo'],
			extra_cflags=['-O3'],
			build_directory=build,
			verbose=False,
		)
	return _EXTS[name]


def extension():
	return _load('vgf_ext', 'vgf.cu')


def march_extension():
	return _load('vgf_march', 'march.cu')


def foam_extension():
	return _load('vgf_foam', 'foam.cu')


def foam_infer_extension():
	return _load('vgf_foam_infer', 'foam_infer.cu')


class Graph:
	"""Fixed sites and their Delaunay adjacency (CSR, int32) on the GPU."""

	def __init__(self, points, offsets, adj):
		self.points = points.float().contiguous()
		self.offsets = offsets.int().contiguous()
		self.adj = adj.int().contiguous()

	def walk(self, o, d, z, start):
		"""Nearest site of every sample o + z d; z (B, S) sorted along each ray, start (B,) a site near o."""
		site, steps = extension().walk(self.points, self.offsets, self.adj, o.float().contiguous(),
									   d.float().contiguous(), z.float().contiguous(), start.int().contiguous())
		return site, steps


class _Density(torch.autograd.Function):
	@staticmethod
	def forward(ctx, raw, log_scale, quat, x, site, graph):
		sigma = extension().density_forward(graph.points, graph.offsets, graph.adj, raw, log_scale, quat, x, site, 1.0)
		ctx.save_for_backward(raw, log_scale, quat, x, site)
		ctx.graph = graph
		return sigma

	@staticmethod
	def backward(ctx, grad_sigma):
		raw, log_scale, quat, x, site = ctx.saved_tensors
		g = ctx.graph
		g_raw, g_ls, g_quat = extension().density_backward(g.points, g.offsets, g.adj, raw, log_scale, quat, x, site,
														   grad_sigma.contiguous())
		return g_raw, g_ls, g_quat, None, None, None


def density(graph, raw, log_scale, quat, x, site, scale_mult=1.0):
	"""sigma(x) = sum over the Voronoi neighborhood of site of exp(raw_k) G_k(x). scale_mult widens the
	Gaussians (proposal passes, no gradient)."""
	x = x.detach().float().contiguous()
	site = site.int().contiguous()
	params = [p.contiguous() for p in (raw, log_scale, quat)]
	if scale_mult != 1.0 or not torch.is_grad_enabled() or not any(p.requires_grad for p in params):
		return extension().density_forward(graph.points, graph.offsets, graph.adj, *[p.detach() for p in params], x,
										   site, float(scale_mult))
	return _Density.apply(*params, x, site, graph)


class _Color(torch.autograd.Function):
	@staticmethod
	def forward(ctx, log_scale, quat, sh, x, dirs, site, graph, sh_deg):
		rgb = extension().color_forward(graph.points, graph.offsets, graph.adj, log_scale, quat, sh, sh_deg, x, dirs,
										site)
		ctx.save_for_backward(log_scale, quat, sh, x, dirs, site, rgb)
		ctx.graph, ctx.sh_deg = graph, sh_deg
		return rgb

	@staticmethod
	def backward(ctx, grad_rgb):
		log_scale, quat, sh, x, dirs, site, rgb = ctx.saved_tensors
		g = ctx.graph
		g_ls, g_quat, g_sh = extension().color_backward(g.points, g.offsets, g.adj, log_scale, quat, sh, ctx.sh_deg,
														x, dirs, site, rgb, grad_rgb.contiguous())
		return g_ls, g_quat, g_sh, None, None, None, None, None


def color(graph, log_scale, quat, sh, x, dirs, site, sh_deg=3):
	"""C(x, d) = sum_k H_k(x) c_k(d) / sum_k H_k(x) over the Voronoi neighborhood of site; sh (N, 16, 3)."""
	x = x.detach().float().contiguous()
	dirs = dirs.detach().float().contiguous()
	site = site.int().contiguous()
	params = [p.contiguous() for p in (log_scale, quat, sh)]
	if not torch.is_grad_enabled() or not any(p.requires_grad for p in params):
		return extension().color_forward(graph.points, graph.offsets, graph.adj, *[p.detach() for p in params],
										 int(sh_deg), x, dirs, site)
	return _Color.apply(*params, x, dirs, site, graph, int(sh_deg))


class _Conic(torch.autograd.Function):
	@staticmethod
	def forward(ctx, log_scale, quat, floor):
		ctx.save_for_backward(log_scale, quat)
		ctx.floor = floor
		return extension().conic_forward(log_scale, quat, floor)

	@staticmethod
	def backward(ctx, gA):
		log_scale, quat = ctx.saved_tensors
		g_ls, g_quat = extension().conic_backward(log_scale, quat, ctx.floor, gA.contiguous())
		return g_ls, g_quat, None


def conic(log_scale, quat, floor=0.0):
	"""(N, 6) inverse covariances (xx, xy, xz, yy, yz, zz) of R diag(s^2 + floor^2) R^T (fused CUDA kernels)."""
	return _Conic.apply(log_scale.contiguous(), quat.contiguous(), float(floor))


def conic_reference(log_scale, quat, floor=0.0):
	"""PyTorch version of conic (for tests)."""
	q = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
	w, x, y, z = q.unbind(-1)
	R = torch.stack([
		1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
		2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
		2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)
	inv_var = 1.0 / (torch.exp(2 * log_scale.clamp(-20, 20)) + floor * floor)
	A = torch.einsum('nak,nk,nbk->nab', R, inv_var, R)
	return torch.stack([A[:, 0, 0], A[:, 0, 1], A[:, 0, 2], A[:, 1, 1], A[:, 1, 2], A[:, 2, 2]], -1).contiguous()


class _March(torch.autograd.Function):
	@staticmethod
	def forward(ctx, raw, dA, cA, sh, graph, rays, cfg):
		rgb, T, depth, stats = march_extension().march_forward(graph.points, graph.offsets, graph.adj, raw, dA, cA,
															   sh, cfg[0], *rays, *cfg[1:])
		ctx.save_for_backward(raw, dA, cA, sh, rgb, T, *rays)
		ctx.graph, ctx.cfg = graph, cfg
		ctx.mark_non_differentiable(depth, stats)
		return rgb, T, depth, stats

	@staticmethod
	def backward(ctx, g_rgb, g_T, _g_depth, _g_stats):
		raw, dA, cA, sh, rgb, T, *rays = ctx.saved_tensors
		g, cfg = ctx.graph, ctx.cfg
		g_raw, g_dA, g_cA, g_sh = march_extension().march_backward(
			g.points, g.offsets, g.adj, raw, dA, cA, sh, cfg[0], *rays, *cfg[1:], rgb, T, g_rgb.contiguous(),
			g_T.contiguous())
		return g_raw, g_dA, g_cA, g_sh, None, None, None


def march(graph, raw, dA, cA, sh, sh_deg, o, d, start, near, mid, far, dt, n_bg, t_thresh=1e-4, w_thresh=1e-4):
	"""Uniform-step fused rendering -> foreground rgb (B, 3) (without background), final transmittance (B,),
	depth (B,), per-ray stats (B, 3) int: samples marched, samples with color, walk steps."""
	rays = tuple(t.float().contiguous() for t in (o, d)) + (start.int().contiguous(),) + \
		tuple(t.float().contiguous() for t in (near, mid, far))
	cfg = (int(sh_deg), float(dt), int(n_bg), float(t_thresh), float(w_thresh))
	params = [p.contiguous() for p in (raw, dA, cA, sh)]
	if not torch.is_grad_enabled() or not any(p.requires_grad for p in params):
		return march_extension().march_forward(graph.points, graph.offsets, graph.adj,
											   *[p.detach() for p in params], cfg[0], *rays, *cfg[1:])
	return _March.apply(*params, graph, rays, cfg)


class _Foam(torch.autograd.Function):
	@staticmethod
	def forward(ctx, raw, dA, cA, sh, graph, rays, cfg, record_cap):
		rgb, T, depth, stats, rec = foam_extension().foam_forward(graph.points, graph.offsets, graph.adj, raw, dA, cA,
																  sh, cfg[0], *rays, *cfg[1:], record_cap)
		ctx.save_for_backward(raw, dA, cA, sh, rgb, T, rec, stats, *rays)
		ctx.graph, ctx.cfg = graph, cfg
		ctx.mark_non_differentiable(depth, stats)
		return rgb, T, depth, stats

	@staticmethod
	def backward(ctx, g_rgb, g_T, _g_depth, _g_stats):
		raw, dA, cA, sh, rgb, T, rec, stats, *rays = ctx.saved_tensors
		g, cfg = ctx.graph, ctx.cfg
		g_raw, g_dA, g_cA, g_sh = foam_extension().foam_backward(
			g.points, g.offsets, g.adj, raw, dA, cA, sh, cfg[0], *rays, *cfg[1:], rgb, T, g_rgb.contiguous(),
			g_T.contiguous(), rec, stats)
		return g_raw, g_dA, g_cA, g_sh, None, None, None, None


def foam(graph, raw, dA, cA, sh, sh_deg, o, d, start, near, far, t_thresh=1e-4, w_thresh=1e-4, record_cap=256):
	"""Rendering at the interaction points of the rays with the Voronoi faces -> foreground rgb (B, 3), final
	transmittance (B,), depth (B,), per-ray stats (B, 3) int: samples, segments with color, walk steps to x(near).
	With gradients, up to record_cap samples per ray are kept for the backward pass (0: re-trace every ray)."""
	rays = tuple(t.float().contiguous() for t in (o, d)) + (start.int().contiguous(),) + \
		tuple(t.float().contiguous() for t in (near, far))
	cfg = (int(sh_deg), float(t_thresh), float(w_thresh))
	params = [p.contiguous() for p in (raw, dA, cA, sh)]
	if not torch.is_grad_enabled() or not any(p.requires_grad for p in params):
		return foam_extension().foam_forward(graph.points, graph.offsets, graph.adj, *[p.detach() for p in params],
											 cfg[0], *rays, *cfg[1:], 0)[:4]
	return _Foam.apply(*params, graph, rays, cfg, int(record_cap))
