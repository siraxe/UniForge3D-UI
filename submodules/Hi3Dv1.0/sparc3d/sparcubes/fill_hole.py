"""Boundary-loop detection and triangulation of holes in extracted meshes."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from sparc3d.structures.point import Mesh

try:
    import mapbox_earcut
except ImportError:  # the numpy ear-clipping fallback still works
    mapbox_earcut = None


def fill_mesh_holes(mesh: Mesh, length_thresh: int = 10000) -> Mesh:
    """Close every boundary loop of a triangle mesh.

    Boundary half-edges are found on the GPU, then each loop is triangulated on the
    CPU. ``mapbox_earcut`` triangulates the loop in its best-fit plane, with the
    quadratic numpy ear-clipper kept as a fallback. Loops longer than
    ``length_thresh`` are left open.

    Args:
        mesh: Mesh to repair. Vertices are reused unchanged; only faces are added.
        length_thresh: Maximum boundary-loop length to fill.

    Returns:
        New mesh with the fill triangles appended.
    """
    vertices = mesh.vertices
    faces = mesh.faces
    device = vertices.device

    # A directed half-edge is on the boundary when its undirected edge occurs once.
    v0, v1, v2 = faces.unbind(dim=1)
    directed = torch.cat(
        [
            torch.stack([v0, v1], dim=1),
            torch.stack([v1, v2], dim=1),
            torch.stack([v2, v0], dim=1),
        ],
        dim=0,
    )
    undirected = torch.stack(
        [
            torch.minimum(directed[:, 0], directed[:, 1]),
            torch.maximum(directed[:, 0], directed[:, 1]),
        ],
        dim=1,
    )
    _, inv_idx, counts = torch.unique(
        undirected, return_inverse=True, return_counts=True, dim=0
    )
    boundary_directed = directed[counts[inv_idx] == 1]

    loops = _boundary_loops(boundary_directed.cpu().numpy().astype(int))

    if loops:
        # One host copy for all loops. Gathering each loop's vertices from the
        # CUDA tensor (and syncing) separately dominated extraction at 1536,
        # where the isosurface can leave hundreds of thousands of tiny loops.
        source = vertices.detach().cpu().numpy()
        new_faces = [
            _triangulate_loop(source, loop)
            for loop in loops
            if 3 <= len(loop) <= length_thresh
        ]
    else:
        new_faces = []
    if new_faces:
        appended = torch.from_numpy(np.vstack(new_faces)).long().to(device)
        faces = torch.cat([faces, appended], dim=0)

    return Mesh(vertices=vertices, faces=faces)


def _triangulate_loop(vertices: np.ndarray, boundary: List[int]) -> np.ndarray:
    """Triangulate one boundary loop, wound opposite to its boundary edges.

    The fill follows the same orientation as the old ear-clipper so the added
    triangles agree with the surrounding surface.
    """
    indices = np.asarray(boundary, dtype=np.int64)
    if len(indices) == 3:
        return indices[[2, 1, 0]]
    points = vertices[indices]
    triangles = _earcut(points)
    if triangles is None:
        return _ear_clip(points, indices)
    return indices[triangles]


def _earcut(points: np.ndarray) -> np.ndarray | None:
    """Triangulate a 3D loop in its best-fit plane with mapbox_earcut.

    Returns local triangle indices, or ``None`` when the loop is degenerate and
    the caller should fall back to ear clipping.
    """
    if mapbox_earcut is None or len(points) < 4:
        return None
    centered = points.astype(np.float64, copy=False)
    centered = centered - centered.mean(axis=0, keepdims=True)
    radius = float(np.max(np.linalg.norm(centered, axis=1)))
    if not np.isfinite(radius) or radius <= 0.0:
        return None

    # Newell normal: robust for non-planar loops and cheap for long ones.
    normal = np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= radius * radius * 1e-12:
        return None
    normal = normal / norm

    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(normal @ helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(helper, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    coords = np.stack([centered @ u, centered @ v], axis=1)

    x, y = coords[:, 0], coords[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if not np.isfinite(area2) or abs(area2) <= radius * radius * 1e-12:
        return None

    try:
        triangles = mapbox_earcut.triangulate_float64(
            np.ascontiguousarray(coords),
            np.array([len(coords)], dtype=np.uint32),
        ).reshape(-1, 3)
    except Exception:  # noqa: BLE001 - fall back to the numpy path
        return None
    if triangles.size == 0:
        return None
    # mapbox_earcut emits counter-clockwise triangles in the projected frame.
    # The fill convention is opposite to the boundary loop order, so flip when
    # the loop itself runs counter-clockwise.
    if area2 > 0.0:
        triangles = triangles[:, [0, 2, 1]]
    return triangles


def _boundary_loops(boundary: np.ndarray) -> List[List[int]]:
    """Chain boundary half-edges ``[B, 2]`` into closed vertex loops.

    Half-edges are consumed at most once, so loops that share a vertex are still
    separated instead of merging into one traversal.
    """
    outgoing: dict[int, List[int]] = {}
    for idx, (start, _) in enumerate(boundary):
        outgoing.setdefault(start, []).append(idx)
    visited = [False] * len(boundary)

    loops: List[List[int]] = []
    for i in range(len(boundary)):
        if visited[i]:
            continue
        visited[i] = True
        start_v = boundary[i][0]
        loop = [start_v, boundary[i][1]]

        while loop[-1] != start_v:
            for next_idx in outgoing.get(loop[-1], []):
                if not visited[next_idx]:
                    visited[next_idx] = True
                    loop.append(boundary[next_idx][1])
                    break
            else:
                break  # dead end: not a closed loop

        if loop[0] == loop[-1] and len(loop) > 3:
            loop.pop()  # drop the duplicated start vertex
            loops.append(loop)
    return loops


def _ear_clip(points: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Triangulate one boundary loop, repeatedly clipping its sharpest corner."""
    boundline = np.asarray(points, dtype=np.float64)
    boundline_idx = np.asarray(boundary, dtype=np.int64)

    faces_hole = []
    while True:
        if len(boundline_idx) == 3:
            faces_hole.append(boundline_idx[[2, 1, 0]])
            break

        next_pts = np.roll(boundline, -1, axis=0)
        T = next_pts - boundline
        T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-6)
        T_prev = np.roll(T, 1, axis=0)
        N = np.cross(-T_prev, T)

        # Signed interior angle at each corner, relative to the loop centroid.
        center = boundline.mean(axis=0, keepdims=True)
        P = next_pts - center
        Q = boundline - center
        sin_sign = np.sign((np.cross(P, Q) * N).sum(axis=1, keepdims=True))
        sin_val = np.linalg.norm(N, axis=1, keepdims=True) * sin_sign
        cos_val = -(T_prev * T).sum(axis=1, keepdims=True)
        A = np.arctan2(sin_val, cos_val).reshape(-1)

        # Clip the sharpest corner: 1/A ranks the smallest positive angle first and
        # every reflex corner last. Scoring exp(1/A) instead ranks corners the same
        # way, but a folded corner has exactly antiparallel edges, hence A of exactly
        # 0, and dividing by it is fatal rather than merely a warning wherever
        # floating-point traps are unmasked. Such a corner is clipped first, which is
        # where the ranking wants it. Ties among comparably sharp corners can break
        # differently than under exp, which rounds large scores together.
        collapsed = A == 0.0
        score = np.where(collapsed, np.inf, 1.0 / np.where(collapsed, 1.0, A))
        ear = int(np.argmax(score))

        i0 = (ear + 1) % len(boundline_idx)
        i2 = (ear - 1) % len(boundline_idx)
        faces_hole.append(boundline_idx[[i0, ear, i2]])

        boundline = np.delete(boundline, ear, axis=0)
        boundline_idx = np.delete(boundline_idx, ear)

    return np.vstack(faces_hole)
