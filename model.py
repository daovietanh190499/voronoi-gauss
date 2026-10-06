"""Per-site parameters: density Gaussian (rho, scale, rotation), color Gaussian (scale, rotation) and SH color."""

import math

import torch
import torch.nn as nn

from ext import Graph

C0 = 0.28209479177387814


def identity_quat(n, device):
	q = torch.zeros(n, 4, device=device)
	q[:, 0] = 1.0
	return q


class VoronoiGaussField(nn.Module):
	"""points (N, 3) fixed sites; offsets/adj their Delaunay adjacency; colors (N, 3) in [0, 1]; spacing (N,)
	mean distance to the nearest sites, used as the initial isotropic scale of both Gaussians."""

	def __init__(self, points, offsets, adj, colors, spacing, init_alpha=0.1, sh_deg=3):
		super().__init__()
		device = points.device
		n = len(points)
		self.graph = Graph(points, offsets, adj)
		self.max_sh_deg, self.sh_deg = sh_deg, 0
		s = spacing.clamp_min(1e-4)
		# a ray through the center of a lone Gaussian then has opacity init_alpha: rho s sqrt(2 pi) = -ln(1 - a)
		rho = -math.log(1.0 - init_alpha) / (s * math.sqrt(2.0 * math.pi))
		self.raw = nn.Parameter(rho.log())
		self.d_log_scale = nn.Parameter(s.log()[:, None].repeat(1, 3))
		self.d_quat = nn.Parameter(identity_quat(n, device))
		self.c_log_scale = nn.Parameter(s.log()[:, None].repeat(1, 3))
		self.c_quat = nn.Parameter(identity_quat(n, device))
		self.sh_dc = nn.Parameter(((colors.clamp(0, 1) - 0.5) / C0)[:, None])
		self.sh_rest = nn.Parameter(torch.zeros(n, 15, 3, device=device))
		self.bg_logit = nn.Parameter(torch.zeros(3, device=device))

	@property
	def n_sites(self):
		return len(self.raw)

	def sh(self):
		return torch.cat([self.sh_dc, self.sh_rest], 1)

	def walk(self, o, d, z, start):
		return self.graph.walk(o, d, z, start)

	def param_groups(self, lr):
		return [
			{'params': [self.raw], 'lr': lr['raw'], 'name': 'raw'},
			{'params': [self.d_log_scale, self.c_log_scale], 'lr': lr['scale'], 'name': 'scale'},
			{'params': [self.d_quat, self.c_quat], 'lr': lr['quat'], 'name': 'quat'},
			{'params': [self.sh_dc], 'lr': lr['sh_dc'], 'name': 'sh_dc'},
			{'params': [self.sh_rest], 'lr': lr['sh_rest'], 'name': 'sh_rest'},
			{'params': [self.bg_logit], 'lr': lr['bg'], 'name': 'bg'},
		]
