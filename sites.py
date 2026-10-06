"""Sites of the field: filtered COLMAP points, their Delaunay (Voronoi-neighbor) adjacency, and the site
containing each camera (k-d tree query, one per camera; the ray samples are then located by walking)."""

import numpy as np
from scipy.spatial import Delaunay, cKDTree


def scene_sites(scene, max_error=2.0, min_track=3, radius=0.98, sor_k=8, sor_std=2.5):
	"""Sparse points inside the ball with a small reprojection error and >= min_track views, without
	statistical outliers (mean distance to the sor_k nearest points) and without duplicates."""
	p = scene.points
	track_len = np.array([len(t) for t in scene.point_tracks])
	keep = (np.linalg.norm(p, axis=1) < radius) & (scene.point_error < max_error) & (track_len >= min_track)
	p, c = p[keep], scene.point_colors[keep]
	d, _ = cKDTree(p).query(p, sor_k + 1)
	mean_d = d[:, 1:].mean(1)
	keep = mean_d < mean_d.mean() + sor_std * mean_d.std()
	p, c = p[keep], c[keep]
	_, first = np.unique(np.round(p, 7), axis=0, return_index=True)
	first = np.sort(first)
	return p[first], c[first]


def delaunay_graph(points):
	"""-> (points kept, index of the kept points, CSR offsets (N + 1), adjacency (M)), int32. Points left out
	of the triangulation by Qhull (coplanar / nearly coincident) are dropped and the triangulation rebuilt."""
	idx = np.arange(len(points))
	for _ in range(3):
		tri = Delaunay(points[idx])
		offsets, adj = tri.vertex_neighbor_vertices
		alone = np.diff(offsets) == 0
		if not alone.any():
			return points[idx], idx, offsets.astype(np.int32), adj.astype(np.int32)
		idx = idx[~alone]
	raise RuntimeError('Delaunay triangulation keeps dropping points')


def nearest_sites(points, queries):
	return cKDTree(points).query(queries)[1].astype(np.int32)


def neighbor_spacing(points, k=3):
	"""Mean distance to the k nearest other sites."""
	d, _ = cKDTree(points).query(points, k + 1)
	return d[:, 1:].mean(1)
