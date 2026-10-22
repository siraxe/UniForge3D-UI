"""Local quadric-fitting curvature: principal curvatures/directions."""
from __future__ import annotations

import numpy as np

from ._backend import accelerator
from .mesh import TriMesh


def _canonical_d1_sign(d1, vi) -> None:
    """Flip ``d1[vi]`` so its largest-|component| (first on ties) is >= 0."""
    c = d1[vi]
    i = int(np.argmax(np.abs(c)))
    if c[i] < 0:
        d1[vi] = -c


def compute_vertex_curvature(mesh: TriMesh, radius_scale: float = 2.0,
                             progress=None):
    """Return (k1, k2, d1) per vertex: signed principal curvatures and the
    unit direction of max |curvature|.

    ``progress`` is called with a fraction in [0, 1] as the vertex loop
    advances."""
    V, VN = mesh.V, mesh.vertex_normals
    n = mesh.n_vertices

    ref = np.where(np.abs(VN[:, 2:3]) < 0.9,
                   np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    tu = np.cross(VN, ref)
    tu /= np.maximum(np.linalg.norm(tu, axis=1, keepdims=True), 1e-300)
    tv = np.cross(VN, tu)

    k1 = np.zeros(n)
    k2 = np.zeros(n)
    d1 = np.zeros_like(V)

    el = mesh.edges[:, 0]
    eh = mesh.edges[:, 1]
    nbr = np.concatenate([np.stack([el, eh], axis=1),
                          np.stack([eh, el], axis=1)])
    order = np.argsort(nbr[:, 0], kind="stable")
    nbr = nbr[order]
    starts = np.zeros(n + 1, dtype=np.int64)
    starts[1:] = np.cumsum(np.bincount(nbr[:, 0], minlength=n))

    # Below this size the Python loop finishes before the Numba call
    # overhead (cache load + chunk dispatch) is paid back.
    accel = accelerator()
    if accel is not None and n >= 12000:
        return accel.vertex_curvature(
            V, VN, tu, tv, starts,
            np.ascontiguousarray(nbr[:, 1], dtype=np.int64),
            progress=progress)

    for vi in range(n):
        if progress is not None and vi % 512 == 0:
            progress(vi / max(1, n))
        s, e = starts[vi], starts[vi + 1]
        idx = nbr[s:e, 1]
        if len(idx) < 5:
            continue
        P = V[idx] - V[vi]
        x = P @ tu[vi]
        y = P @ tv[vi]
        z = P @ VN[vi]
        r2 = x * x + y * y
        w = 1.0 / (r2 + 1e-30)
        A = np.stack([x * x, x * y, y * y, x, y], axis=1) * w[:, None]
        b = z * w
        try:
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        a2, ab, c2 = coef[0], coef[1], coef[2]
        H = np.array([[2 * a2, ab], [ab, 2 * c2]])
        evals, evecs = np.linalg.eigh(H)
        k1[vi], k2[vi] = evals
        pick = 1 if abs(evals[1]) >= abs(evals[0]) else 0
        dloc = evecs[:, pick]
        d1[vi] = dloc[0] * tu[vi] + dloc[1] * tv[vi]
        # Canonical sign: the largest-|component| (first on ties) is made
        # positive.  Removes the LAPACK eigenvector sign ambiguity so every
        # backend (python / numba / rust) produces identical directions.
        _canonical_d1_sign(d1, vi)
    return k1, k2, d1


def face_trust_and_direction(mesh: TriMesh, k1, k2, d1):
    V, F = mesh.V, mesh.F
    fn = mesh.face_normals
    ref = np.where(np.abs(fn[:, 2:3]) < 0.9,
                   np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    u = np.cross(fn, ref)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-300)
    v = np.cross(fn, u)

    dir_face = np.zeros((len(F), 3))
    aniso_v = np.abs(k1 - k2)
    for k in range(3):
        vi = F[:, k]
        dir_face += d1[vi] * aniso_v[vi][:, None]
    knorm = np.linalg.norm(dir_face, axis=1)
    ok = knorm > 1e-12
    dir_face[ok] /= knorm[ok, None]
    dir_face[~ok] = u[~ok]

    a = np.arctan2(np.sum(dir_face * v, axis=1), np.sum(dir_face * u, axis=1))

    num = np.zeros(len(F))
    den = np.zeros(len(F))
    for k in range(3):
        vi = F[:, k]
        num += aniso_v[vi]
        den += np.abs(k1[vi]) + np.abs(k2[vi])
    trust = np.clip(num / (den + 1e-30), 0.0, 1.0)
    return a, trust, u, v
