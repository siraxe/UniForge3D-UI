"""Parallel Numba kernels for :mod:`polish`.

Mirror the NumPy reference in ``api.py`` (same CSR neighbor order, same
passes) with per-vertex loops in parallel.  Imported lazily by
``polish.polish``; without numba the NumPy path is used unchanged.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _umbrella3(values, indptr, nbr, deg):
    n = values.shape[0]
    out = np.empty_like(values)
    for i in prange(n):
        d = deg[i]
        if d > 0:
            sx = 0.0
            sy = 0.0
            sz = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                j = nbr[k]
                sx += values[j, 0]
                sy += values[j, 1]
                sz += values[j, 2]
            half_inv = 0.5 / d
            out[i, 0] = 0.5 * values[i, 0] + sx * half_inv
            out[i, 1] = 0.5 * values[i, 1] + sy * half_inv
            out[i, 2] = 0.5 * values[i, 2] + sz * half_inv
        else:
            out[i, 0] = values[i, 0]
            out[i, 1] = values[i, 1]
            out[i, 2] = values[i, 2]
    return out


@njit(parallel=True, cache=True)
def _umbrella1(values, indptr, nbr, deg):
    n = values.shape[0]
    out = np.empty_like(values)
    for i in prange(n):
        d = deg[i]
        if d > 0:
            s = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                s += values[nbr[k]]
            out[i] = 0.5 * values[i] + s * (0.5 / d)
        else:
            out[i] = values[i]
    return out


@njit(parallel=True, cache=True)
def _normalize3(values):
    n = values.shape[0]
    out = np.empty_like(values)
    for i in prange(n):
        x = values[i, 0]
        y = values[i, 1]
        z = values[i, 2]
        length = sqrt(x * x + y * y + z * z)
        if length == 0.0:
            length = 1.0
        out[i, 0] = x / length
        out[i, 1] = y / length
        out[i, 2] = z / length
    return out


@njit(parallel=True, cache=True)
def _face_normals(vertices, faces):
    count = faces.shape[0]
    out = np.empty((count, 3))
    for t in prange(count):
        a = faces[t, 0]
        b = faces[t, 1]
        c = faces[t, 2]
        ux = vertices[b, 0] - vertices[a, 0]
        uy = vertices[b, 1] - vertices[a, 1]
        uz = vertices[b, 2] - vertices[a, 2]
        wx = vertices[c, 0] - vertices[a, 0]
        wy = vertices[c, 1] - vertices[a, 1]
        wz = vertices[c, 2] - vertices[a, 2]
        out[t, 0] = uy * wz - uz * wy
        out[t, 1] = uz * wx - ux * wz
        out[t, 2] = ux * wy - uy * wx
    return out


@njit(parallel=True, cache=True)
def _vertex_normals(face_normals, face_indptr, vertex_faces):
    n = face_indptr.shape[0] - 1
    out = np.zeros((n, 3))
    for i in prange(n):
        x = 0.0
        y = 0.0
        z = 0.0
        for k in range(face_indptr[i], face_indptr[i + 1]):
            t = vertex_faces[k]
            x += face_normals[t, 0]
            y += face_normals[t, 1]
            z += face_normals[t, 2]
        length = sqrt(x * x + y * y + z * z)
        if length != 0.0:
            out[i, 0] = x / length
            out[i, 1] = y / length
            out[i, 2] = z / length
    return out


@njit(parallel=True, cache=True)
def _tangential(vertices, normals, indptr, nbr, deg):
    n = vertices.shape[0]
    out = np.empty_like(vertices)
    for i in prange(n):
        d = deg[i]
        if d > 0:
            sx = 0.0
            sy = 0.0
            sz = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                j = nbr[k]
                sx += vertices[j, 0]
                sy += vertices[j, 1]
                sz += vertices[j, 2]
            inv = 1.0 / (d + 1)
            ax = (vertices[i, 0] + sx) * inv
            ay = (vertices[i, 1] + sy) * inv
            az = (vertices[i, 2] + sz) * inv

            tx = 0.0
            ty = 0.0
            tz = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                j = nbr[k]
                dx = ax - vertices[j, 0]
                dy = ay - vertices[j, 1]
                dz = az - vertices[j, 2]
                nx = normals[j, 0]
                ny = normals[j, 1]
                nz = normals[j, 2]
                dot = dx * nx + dy * ny + dz * nz
                tx -= dot * nx
                ty -= dot * ny
                tz -= dot * nz
            half_inv = 0.5 / d
            out[i, 0] = ax + tx * half_inv
            out[i, 1] = ay + ty * half_inv
            out[i, 2] = az + tz * half_inv
        else:
            out[i, 0] = vertices[i, 0]
            out[i, 1] = vertices[i, 1]
            out[i, 2] = vertices[i, 2]
    return out


def _face_adjacency(faces, vertex_count: int):
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    count = len(faces)
    flat = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
    vertex_face_order = np.argsort(flat, kind="stable")
    vertex_faces = (vertex_face_order % count).astype(np.int64)
    counts = np.bincount(flat, minlength=vertex_count)
    indptr = np.zeros(vertex_count + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, vertex_faces


def polish(vertices, faces, iterations: int, passes: int,
           preserve_peaks: bool, adj,
           normal_iterations: int = 5) -> np.ndarray:
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int64)
    indptr, nbr, deg, _rows = adj
    face_indptr, vertex_faces = _face_adjacency(f, len(v))

    if preserve_peaks:
        base = v.copy()
        face_normals = _face_normals(base, f)
        n0 = _vertex_normals(face_normals, face_indptr, vertex_faces)
        smoothed = base
        for _ in range(2 * iterations):
            smoothed = _umbrella3(smoothed, indptr, nbr, deg)
        disp = smoothed - base
        dist = np.linalg.norm(disp, axis=1)
        weight = np.zeros(len(v), dtype=np.float64)
        nonzero = dist > 0.0
        weight[nonzero] = 0.5 + 0.5 * np.einsum(
            "ij,ij->i", disp[nonzero] / dist[nonzero, None], n0[nonzero])
        for _ in range(iterations):
            weight = _umbrella1(weight, indptr, nbr, deg)
        v = base + weight[:, None] * (smoothed - base)
    else:
        for _ in range(iterations):
            v = _umbrella3(v, indptr, nbr, deg)

    for _ in range(passes):
        p0 = v
        face_normals = _face_normals(p0, f)
        normals = _vertex_normals(face_normals, face_indptr, vertex_faces)
        for _ in range(max(1, int(normal_iterations))):
            normals = _umbrella3(normals, indptr, nbr, deg)
        normals = _normalize3(normals)
        v = _tangential(p0, normals, indptr, nbr, deg)
    return v
