"""COLMAP scenes (Mip-NeRF 360 / Deep Blending layout: sparse/0/*.bin, images_<d>/), normalized into the unit ball."""

import os
import struct

import numpy as np
import torch
from PIL import Image

CAMERA_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12}  # SIMPLE_PINHOLE PINHOLE SIMPLE_RADIAL RADIAL OPENCV ...


def qvec_to_rot(q):
	w, x, y, z = q
	return np.array([
		[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
		[2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
		[2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
	])


def read_cameras(path):
	cams = {}
	with open(path, 'rb') as fh:
		for _ in range(struct.unpack('<Q', fh.read(8))[0]):
			cid, model, w, h = struct.unpack('<iiQQ', fh.read(24))
			params = struct.unpack(f'<{CAMERA_PARAMS[model]}d', fh.read(8 * CAMERA_PARAMS[model]))
			cams[cid] = (model, w, h, params)
	return cams


def read_images(path):
	images = {}
	with open(path, 'rb') as fh:
		for _ in range(struct.unpack('<Q', fh.read(8))[0]):
			iid = struct.unpack('<i', fh.read(4))[0]
			q = struct.unpack('<4d', fh.read(32))
			t = struct.unpack('<3d', fh.read(24))
			cid = struct.unpack('<i', fh.read(4))[0]
			name = b''
			while (c := fh.read(1)) != b'\0':
				name += c
			n2d = struct.unpack('<Q', fh.read(8))[0]
			fh.seek(24 * n2d, 1)
			images[iid] = (name.decode(), qvec_to_rot(q), np.array(t), cid)
	return images


def read_points(path):
	xyz, rgb, err, tracks = [], [], [], []
	with open(path, 'rb') as fh:
		for _ in range(struct.unpack('<Q', fh.read(8))[0]):
			fh.read(8)
			xyz.append(struct.unpack('<3d', fh.read(24)))
			rgb.append(struct.unpack('<3B', fh.read(3)))
			err.append(struct.unpack('<d', fh.read(8))[0])
			n = struct.unpack('<Q', fh.read(8))[0]
			tracks.append(np.frombuffer(fh.read(8 * n), dtype=np.int32)[0::2].copy())
	return np.array(xyz), np.array(rgb, np.float32) / 255.0, np.array(err), tracks


def intrinsics(model, params):
	if model in (0, 2, 3):  # f cx cy (+ radial terms, ignored: the images are undistorted)
		f, cx, cy = params[:3]
		return f, f, cx, cy
	return params[:4]


class ColmapScene:
	"""Images, cameras and sparse points of one split. World coordinates are mapped to the unit ball:
	x_n = (x - center) / scale, with center/scale from the sparse points (so that `keep_quantile` of them lie
	within radius `radius`)."""

	def __init__(self, root, split='train', downsample=4, device='cuda', test_every=8, radius=0.95,
				 keep_quantile=0.98, normalization=None):
		self.root, self.device = root, device
		sparse = os.path.join(root, 'sparse', '0')
		cams = read_cameras(os.path.join(sparse, 'cameras.bin'))
		images = read_images(os.path.join(sparse, 'images.bin'))
		order = sorted(images, key=lambda i: images[i][0])
		idx = np.arange(len(order))
		keep = idx % test_every != 0 if split == 'train' else idx % test_every == 0
		self.image_ids = [order[i] for i in idx[keep]]
		self.names = [images[i][0] for i in self.image_ids]
		self.n = len(self.names)

		img_dir = os.path.join(root, 'images' if downsample == 1 else f'images_{downsample}')
		rgb = [np.asarray(Image.open(os.path.join(img_dir, n)).convert('RGB')) for n in self.names]
		self.H, self.W = rgb[0].shape[:2]
		self.images = torch.from_numpy(np.stack(rgb)).to(device)
		del rgb

		xyz, colors, err, tracks = read_points(os.path.join(sparse, 'points3D.bin'))
		if normalization is None:
			center = np.median(xyz, 0)
			scale = np.quantile(np.linalg.norm(xyz - center, axis=1), keep_quantile) / radius
		else:
			center, scale = normalization
		self.normalization = (center, scale)
		# normalized -> world, in the same 4x4 form as NeuS' scale_mat
		self.scale_mat = np.diag([scale, scale, scale, 1.0])
		self.scale_mat[:3, 3] = center

		Ks, c2ws = [], []
		for iid in self.image_ids:
			_, R, t, cid = images[iid]
			model, w, h, params = cams[cid]
			fx, fy, cx, cy = intrinsics(model, params)
			sx, sy = self.W / w, self.H / h
			Ks.append(np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]]))
			c2w = np.eye(4)
			c2w[:3, :3] = R.T
			c2w[:3, 3] = (-R.T @ t - center) / scale
			c2ws.append(c2w)
		self.K = torch.tensor(np.stack(Ks), dtype=torch.float32, device=device)
		self.K_inv = torch.linalg.inv(self.K)
		self.c2w = torch.tensor(np.stack(c2ws), dtype=torch.float32, device=device)
		self.centers = self.c2w[:, :3, 3]

		# sparse points in normalized coordinates, with the normalized centers of the cameras observing them
		id_to_slot = {iid: s for s, iid in enumerate(order)}
		all_centers = np.stack([(-images[i][1].T @ images[i][2] - center) / scale for i in order])
		self.points = (xyz - center) / scale
		self.point_colors = colors
		self.point_error = err
		self.point_tracks = [np.array([id_to_slot[i] for i in tr if i in id_to_slot]) for tr in tracks]
		self.all_centers = all_centers

	def rays(self, img, px, py):
		p = torch.stack([px + 0.5, py + 0.5, torch.ones_like(px)], -1).float()
		d = torch.einsum('nij,nj->ni', self.K_inv[img], p)
		d = torch.einsum('nij,nj->ni', self.c2w[img, :3, :3], d)
		d = d / d.norm(dim=-1, keepdim=True)
		return self.centers[img], d

	def random_rays(self, batch):
		"""-> o, d, rgb, image index of every ray."""
		img = torch.randint(0, self.n, (batch,), device=self.device)
		px = torch.randint(0, self.W, (batch,), device=self.device)
		py = torch.randint(0, self.H, (batch,), device=self.device)
		o, d = self.rays(img, px, py)
		return o, d, self.images[img, py, px].float() / 255.0, img

	def image_rays(self, i):
		py, px = torch.meshgrid(torch.arange(self.H, device=self.device).float(),
								torch.arange(self.W, device=self.device).float(), indexing='ij')
		img = torch.full_like(px, i, dtype=torch.long)
		o, d = self.rays(img.reshape(-1), px.reshape(-1), py.reshape(-1))
		return o, d, self.images[i].float() / 255.0
