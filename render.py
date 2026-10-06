"""Rendering of the Voronoi-Gaussian field: NeRF-style sampling (render_rays), uniform steps (render_rays_uniform)
or the ray / Voronoi-face interaction points (render_rays_foam).

For render_rays, samples live in a normalized ray coordinate s in [0, 1]: the first half is linear in t inside the unit ball,
the second half linear in disparity from the unit sphere out to radius `far_radius`. 64 stratified samples
are followed by two proposal passes (64 + 64 samples drawn from the weights of widened, gradient-free
density Gaussians); the 192 merged samples are walked to their nearest sites and rendered with gradients."""

import math

import torch

import ext


def sphere_exit(o, d, radius):
	b = (o * d).sum(-1)
	c = (o * o).sum(-1) - radius * radius
	return -b + (b * b - c).clamp_min(0).sqrt()


def ray_bounds(o, d, near, far_radius):
	t_mid = sphere_exit(o, d, 1.0).clamp_min(near + 1e-3)
	t_far = torch.maximum(sphere_exit(o, d, far_radius), t_mid + 1e-3)
	return torch.full_like(t_mid, near), t_mid, t_far


def s_to_t(s, t_near, t_mid, t_far):
	t_near, t_mid, t_far = t_near[:, None], t_mid[:, None], t_far[:, None]
	lin = t_near + (t_mid - t_near) * (2 * s).clamp(max=1)
	u = (2 * s - 1).clamp(min=0)
	disp = 1.0 / (1.0 / t_mid + u * (1.0 / t_far - 1.0 / t_mid))
	return torch.where(s < 0.5, lin, disp)


def sample_pdf(bins, weights, n, det=True):
	weights = weights + 1e-5
	pdf = weights / weights.sum(-1, keepdim=True)
	cdf = torch.cat([torch.zeros_like(pdf[:, :1]), torch.cumsum(pdf, -1)], -1)
	if det:
		u = torch.linspace(0.5 / n, 1 - 0.5 / n, n, device=bins.device).expand(len(bins), n).contiguous()
	else:
		u = torch.rand(len(bins), n, device=bins.device)
	inds = torch.searchsorted(cdf, u, right=True)
	below = (inds - 1).clamp_min(0)
	above = inds.clamp_max(cdf.shape[-1] - 1)
	cdf_b, cdf_a = cdf.gather(1, below), cdf.gather(1, above)
	bin_b, bin_a = bins.gather(1, below), bins.gather(1, above)
	denom = torch.where(cdf_a - cdf_b < 1e-5, torch.ones_like(cdf_a), cdf_a - cdf_b)
	return bin_b + (u - cdf_b) / denom * (bin_a - bin_b)


def composite(sigma, t, t_far):
	delta = torch.cat([t[:, 1:] - t[:, :-1], (t_far[:, None] - t[:, -1:]).clamp_min(0)], 1)
	alpha = 1.0 - torch.exp(-sigma * delta)
	trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], 1), 1)
	return alpha * trans[:, :-1], trans[:, -1]


def proposal_weights(model, o, d, s, bounds, start, scale_mult):
	t = s_to_t(s, *bounds)
	site, _ = model.walk(o, d, t, start)
	x = (o[:, None] + d[:, None] * t[..., None]).reshape(-1, 3)
	with torch.no_grad():
		sigma = ext.density(model.graph, model.raw, model.d_log_scale, model.d_quat, x, site.reshape(-1),
							scale_mult).view_as(t)
	w, _ = composite(sigma, t, bounds[2])
	return w


def resample(s, w, n, det):
	mid = 0.5 * (s[:, 1:] + s[:, :-1])
	bins = torch.cat([torch.zeros_like(s[:, :1]), mid, torch.ones_like(s[:, :1])], 1)
	return sample_pdf(bins, w, n, det)


def render_rays(model, o, d, start, n_coarse=64, n_fine=(64, 64), proposal_mult=(4.0, 2.0), near=0.02,
				far_radius=4.0, perturb=True, min_weight=1e-4):
	"""o, d (B, 3) rays, start (B,) site containing each ray origin -> dict with rgb (B, 3), acc, depth, stats."""
	B = len(o)
	bounds = ray_bounds(o, d, near, far_radius)
	s = torch.linspace(0, 1, n_coarse + 1, device=o.device)[:-1].expand(B, n_coarse)
	s = s + (torch.rand_like(s) if perturb else 0.5) / n_coarse
	with torch.no_grad():
		for n, mult in zip(n_fine, proposal_mult):
			w = proposal_weights(model, o, d, s, bounds, start, mult)
			s_new = resample(s, w, n, det=not perturb)
			s, _ = torch.sort(torch.cat([s, s_new], 1), 1)
		t = s_to_t(s, *bounds)
		site, steps = model.walk(o, d, t, start)
	S = t.shape[1]
	x = (o[:, None] + d[:, None] * t[..., None]).reshape(-1, 3)
	site = site.reshape(-1)
	sigma = ext.density(model.graph, model.raw, model.d_log_scale, model.d_quat, x, site).view(B, S)
	w, t_final = composite(sigma, t, bounds[2])

	active = (w.detach() > min_weight).reshape(-1).nonzero().squeeze(1)
	dirs = d[:, None].expand(B, S, 3).reshape(-1, 3)
	c = ext.color(model.graph, model.c_log_scale, model.c_quat, model.sh(), x[active], dirs[active], site[active],
				  model.sh_deg)
	rgb_s = torch.zeros(B * S, 3, device=o.device, dtype=c.dtype).index_put((active,), c).view(B, S, 3)
	bg = torch.sigmoid(model.bg_logit)
	rgb = (w[..., None] * rgb_s).sum(1) + t_final[:, None] * bg
	depth = (w * t).sum(1)
	return {'rgb': rgb, 'acc': 1.0 - t_final, 'depth': depth, 'walk_steps': steps.float().mean() / S,
			'active': len(active) / (B * S)}


def render_rays_uniform(model, o, d, start, dt=1 / 512, n_bg=64, near=0.02, far_radius=4.0, t_thresh=1e-4,
						w_thresh=1e-4, scale_floor=None, perturb=False):
	"""Plenoxels-style sampling: midpoints of fixed steps dt inside the unit ball, n_bg steps uniform in disparity
	beyond it, early termination at T < t_thresh, all in one fused kernel. scale_floor (default dt / 2) is added
	in quadrature to every Gaussian scale so no Gaussian is thinner than the step can resolve."""
	near_t, mid_t, far_t = ray_bounds(o, d, near, far_radius)
	floor = 0.5 * dt if scale_floor is None else scale_floor
	dA = ext.conic(model.d_log_scale, model.d_quat, floor)
	cA = ext.conic(model.c_log_scale, model.c_quat, floor)
	fg, T, depth, stats = ext.march(model.graph, model.raw, dA, cA, model.sh(), model.sh_deg, o, d, start, near_t,
									mid_t, far_t, dt, n_bg, t_thresh, w_thresh)
	rgb = fg + T[:, None] * torch.sigmoid(model.bg_logit)
	stats = stats.float()
	return {'rgb': rgb, 'acc': 1.0 - T, 'depth': depth, 'samples': stats[:, 0].mean(),
			'active': (stats[:, 1].sum() / stats[:, 0].sum()).item(),
			'walk_steps': stats[:, 2].sum() / stats[:, 0].sum()}


def render_rays_foam(model, o, d, start, near=0.02, far_radius=4.0, t_thresh=1e-4, w_thresh=1e-4, scale_floor=0.0,
					 perturb=False):
	"""Samples only at the interaction points of each ray with the Voronoi faces of the sites (plus x(near) and
	x(far)); segments between them are composited with the trapezoid rule in one fused kernel."""
	near_t, _, far_t = ray_bounds(o, d, near, far_radius)
	dA = ext.conic(model.d_log_scale, model.d_quat, scale_floor)
	cA = ext.conic(model.c_log_scale, model.c_quat, scale_floor)
	fg, T, depth, stats = ext.foam(model.graph, model.raw, dA, cA, model.sh(), model.sh_deg, o, d, start, near_t,
								   far_t, t_thresh, w_thresh)
	rgb = fg + T[:, None] * torch.sigmoid(model.bg_logit)
	stats = stats.float()
	segments = (stats[:, 0] - 1).sum().clamp_min(1)
	return {'rgb': rgb, 'acc': 1.0 - T, 'depth': depth, 'samples': stats[:, 0].mean(),
			'active': (stats[:, 1].sum() / segments).item(), 'walk_steps': stats[:, 2].mean()}


def sh_basis(d, deg):
	"""Real SH basis (B, 16) of unit directions d (B, 3), same constants and signs as the CUDA kernels."""
	x, y, z = d.unbind(-1)
	b = torch.zeros(len(d), 16, device=d.device, dtype=d.dtype)
	b[:, 0] = 0.28209479177387814
	if deg >= 1:
		b[:, 1], b[:, 2], b[:, 3] = -0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x
	if deg >= 2:
		xx, yy, zz = x * x, y * y, z * z
		b[:, 4] = 1.0925484305920792 * x * y
		b[:, 5] = -1.0925484305920792 * y * z
		b[:, 6] = 0.31539156525252005 * (2 * zz - xx - yy)
		b[:, 7] = -1.0925484305920792 * x * z
		b[:, 8] = 0.5462742152960396 * (xx - yy)
	if deg >= 3:
		b[:, 9] = -0.5900435899266435 * y * (3 * xx - yy)
		b[:, 10] = 2.890611442640554 * x * y * z
		b[:, 11] = -0.4570457994644658 * y * (4 * zz - xx - yy)
		b[:, 12] = 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy)
		b[:, 13] = -0.4570457994644658 * x * (4 * zz - xx - yy)
		b[:, 14] = 1.445305721320277 * z * (xx - yy)
		b[:, 15] = -0.5900435899266435 * x * (xx - 3 * yy)
	return b


def morton_order(points, bits=10):
	"""Permutation sorting the points along a Z-order curve of their bounding box."""
	lo, hi = points.amin(0), points.amax(0)
	q = ((points - lo) / (hi - lo).clamp_min(1e-12) * (2 ** bits - 1)).long()
	code = torch.zeros(len(points), dtype=torch.long, device=points.device)
	for b in range(bits):
		for a in range(3):
			code |= ((q[:, a] >> b) & 1) << (3 * b + a)
	return torch.argsort(code)


@torch.no_grad()
def pack_foam(model, scale_floor=0.0, eps_density=0.0, sh_half=False, reorder=False):
	"""Records of the inference renderer (see foam_infer.cu): self-first neighborhoods (offsets (N + 1), nbr (E))
	with one float4 per edge = (p_j, half2(r2_j, lam_j)), r2 the squared distance beyond which
	rho_j G_j < eps_density (rho_j exp(-|v|^2 / (2 s_max^2)) bounds it; inf when eps_density = 0) and lam the
	smallest eigenvalue of the color conic, both rounded to the safe side; attr (N, 20) = (density conic,
	min(raw, 30), color conic, baked rgb); sh (N, 16, 3) in fp32 or fp16. reorder renumbers the sites along a
	Z-order curve (perm: new -> old site, inv: old -> new) so that the gathers of nearby sites share cache lines."""
	g = model.graph
	n = len(g.points)
	raw = model.raw.clamp(max=30.0)
	if eps_density > 0:
		s_max2 = (torch.exp(2 * model.d_log_scale.clamp(-20, 20)) + scale_floor ** 2).amax(-1)
		r2 = 2 * s_max2 * (raw - math.log(eps_density)).clamp_min(0) * (1 + 2 ** -9)
	else:
		r2 = torch.full_like(raw, float('inf'))
	lam = 1.0 / (torch.exp(2 * model.c_log_scale.clamp(-20, 20)) + scale_floor ** 2).amax(-1)
	lam = (lam * (1 - 2 ** -9)).clamp(max=65000.0)
	cut = torch.stack([r2, lam], -1).half().view(torch.float32)[:, 0]
	off = g.offsets.long()
	deg = off[1:] - off[:-1]
	offsets = off + torch.arange(n + 1, device=off.device)
	row = torch.repeat_interleave(torch.arange(n, device=off.device), deg)
	nbr = torch.empty(len(g.adj) + n, dtype=torch.long, device=off.device)
	nbr[offsets[:-1]] = torch.arange(n, device=off.device)
	nbr[torch.arange(len(g.adj), device=off.device) + row + 1] = g.adj.long()
	edge = torch.cat([g.points[nbr], cut[nbr, None]], 1).contiguous()
	points = torch.cat([g.points, torch.zeros_like(raw)[:, None]], 1).contiguous()
	dA = ext.conic(model.d_log_scale, model.d_quat, scale_floor)
	cA = ext.conic(model.c_log_scale, model.c_quat, scale_floor)
	z = torch.zeros_like(raw)[:, None]
	attr = torch.cat([dA, raw[:, None], z, cA, z, z, z, z, z, z], 1).contiguous()
	sh = model.sh().contiguous()
	perm = inv = None
	if reorder:
		perm = morton_order(g.points)
		inv = torch.empty_like(perm)
		inv[perm] = torch.arange(n, device=perm.device)
		size = (deg + 1)[perm]
		new_off = torch.cat([torch.zeros_like(size[:1]), torch.cumsum(size, 0)])
		cell = torch.repeat_interleave(torch.arange(n, device=perm.device), size)
		old = offsets[perm][cell] + torch.arange(len(nbr), device=perm.device) - new_off[cell]
		offsets, nbr, edge = new_off, inv[nbr[old]], edge[old].contiguous()
		points, attr, sh = points[perm].contiguous(), attr[perm].contiguous(), sh[perm].contiguous()
	return {'points': points, 'offsets': offsets.int().contiguous(), 'nbr': nbr.int().contiguous(), 'edge': edge,
			'attr': attr, 'sh': sh.half() if sh_half else sh, 'sh32': sh, 'perm': perm, 'inv': inv,
			'orig': (perm if reorder else torch.empty(0, device=raw.device)).int().contiguous(), 'baked_center': None}


@torch.no_grad()
def bake_colors(model, pack, center):
	"""Site colors max(SH_k(d_k) + 1/2, 0) for the view direction d_k from the camera center to site k."""
	points = model.graph.points if pack['perm'] is None else model.graph.points[pack['perm']]
	dirs = torch.nn.functional.normalize(points - center.view(1, 3), dim=-1)
	rgb = torch.einsum('nj,njc->nc', sh_basis(dirs, model.sh_deg), pack['sh32']).add_(0.5).clamp_min_(0.0)
	pack['attr'][:, 16:19] = rgb
	pack['baked_center'] = center


@torch.no_grad()
def render_rays_foam_infer(model, pack, o, d, start, near=0.02, far_radius=4.0, t_thresh=1e-4, w_thresh=1e-4,
						   colors='exact', eps_color=0.0, stage=2, cap=None):
	"""Inference version of render_rays_foam (same samples and quadrature). colors='exact' evaluates the SH per
	ray after the density pass for the segments with weight > w_thresh, 'fused' blends them in the density pass of
	every sample (more SH reads, no divergent second pass), both skipping the neighbors whose normalized color
	weight is below eps_color; 'baked' uses the per-view site colors of bake_colors. Per-ray stats (B, 5):
	samples, segments with color, walk steps, density Gaussians evaluated, SH evaluated."""
	near_t, _, far_t = ray_bounds(o, d, near, far_radius)
	cap = torch.empty(0, dtype=torch.int32, device=o.device) if cap is None else cap.int().contiguous()
	if pack['inv'] is not None:
		start = pack['inv'][start.long()]
	color_cut = 2 * math.log(1 / eps_color) if eps_color > 0 else float('inf')
	fg, T, depth, stats = ext.foam_infer_extension().foam_infer(
		pack['points'], pack['offsets'], pack['nbr'], pack['edge'], pack['attr'], pack['sh'], model.sh_deg,
		o.float().contiguous(), d.float().contiguous(), start.int().contiguous(), near_t.contiguous(),
		far_t.contiguous(), float(t_thresh), float(w_thresh), color_cut, {'exact': 0, 'baked': 1, 'fused': 2}[colors],
		int(stage), cap, pack['orig'])
	rgb = fg + T[:, None] * torch.sigmoid(model.bg_logit)
	return {'rgb': rgb, 'acc': 1.0 - T, 'depth': depth, 'stats': stats}


@torch.no_grad()
def render_image(model, scene, i, start, chunk=8192, renderer=render_rays, **kw):
	o, d, gt = scene.image_rays(i)
	start = torch.full((len(o),), int(start), dtype=torch.int32, device=o.device)
	out = [renderer(model, o[j:j + chunk], d[j:j + chunk], start[j:j + chunk], perturb=False, **kw)['rgb']
		   for j in range(0, len(o), chunk)]
	return torch.cat(out).view(scene.H, scene.W, 3).clamp(0, 1), gt
