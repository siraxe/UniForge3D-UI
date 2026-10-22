"""Cross-field smoothing."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh import TriMesh


def _edge_frame_angles(mesh: TriMesh, u, v):
    F = mesh.F
    fL, fR = mesh.edge_faces()
    inner = fR >= 0
    e_lo, e_hi = mesh.edges[inner, 0], mesh.edges[inner, 1]
    fl, fr = fL[inner], fR[inner]

    d = mesh.V[e_hi] - mesh.V[e_lo]

    def angle_in_frame(fidx):
        du = np.sum(d * u[fidx], axis=1)
        dv = np.sum(d * v[fidx], axis=1)
        return np.arctan2(dv, du)

    a_l = angle_in_frame(fl)
    a_r = angle_in_frame(fr)
    phi = a_l - a_r
    return fl, fr, 4.0 * phi, inner


def smooth_cross_field(mesh: TriMesh, alpha, trust, u, v,
                       lambda_data=1.0, lambda_anchor=1.0, rounds=3):
    nf = mesh.n_faces
    fl, fr, m4, inner = _edge_frame_angles(mesh, u, v)
    w_e = np.ones(len(fl))
    R = np.exp(1j * m4)

    k = np.zeros(len(fl), dtype=np.int64)
    u_prev = np.ones(nf, dtype=np.complex128)

    for rnd in range(max(1, rounds)):
        Rk = R * np.exp(1j * k * (np.pi / 2.0))

        r = np.arange(len(fl))
        rows = [r, r]
        cols = [fl, fr]
        vals = [w_e * np.ones(len(fl)), -w_e * Rk]
        rowcount = len(fl)

        d_w = np.sqrt(lambda_data * trust)
        act = np.flatnonzero(d_w > 1e-6)
        rd = rowcount + np.arange(len(act))
        rows.append(rd)
        cols.append(act)
        vals.append(d_w[act].astype(np.complex128))
        data_rhs = d_w[act] * np.exp(1j * 4.0 * alpha[act])
        rowcount += len(act)

        ra = rowcount + np.arange(nf)
        rows.append(ra)
        cols.append(np.arange(nf))
        vals.append(np.full(nf, lambda_anchor, dtype=np.complex128))
        anchor_rhs = lambda_anchor * u_prev
        rowcount += nf

        A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(rowcount, nf)).tocsr()
        b = np.zeros(A.shape[0], dtype=np.complex128)
        b[len(fl):len(fl) + len(act)] = data_rhs
        b[rowcount - nf:] = anchor_rhs

        H = (A.conj().T @ A).tocsc()
        rhs = A.conj().T @ b
        eps = 1e-8 * (float(H.diagonal().real.mean()) + 1e-30)
        Hs = (H + sp.identity(nf, format="csc") * eps).astype(np.complex128)
        x, info = spla.cg(Hs, rhs, maxiter=20000, rtol=1e-12)
        if info != 0 or np.isnan(x).any():
            u_field = spla.spsolve(Hs, rhs)
        else:
            u_field = x

        mag = np.abs(u_field)
        mag[mag < 1e-300] = 1.0
        u_field = u_field / mag

        ratio = u_field[fl] / (R * u_field[fr])
        k = np.mod(np.round(np.angle(ratio) / (np.pi / 2.0)), 4).astype(np.int64)

    theta = np.mod(np.angle(u_field) / 4.0, np.pi / 2.0)
    transitions = k
    return theta, transitions, u_field


def edge_field_residual(mesh: TriMesh, theta, u, v, transitions=None):
    fL, fR = mesh.edge_faces()
    inner = fR >= 0
    fl, fr = fL[inner], fR[inner]
    e_lo, e_hi = mesh.edges[inner, 0], mesh.edges[inner, 1]
    d = mesh.V[e_hi] - mesh.V[e_lo]

    def ang(fidx):
        return np.arctan2(np.sum(d * v[fidx], axis=1),
                          np.sum(d * u[fidx], axis=1))

    phi = ang(fl) - ang(fr)
    k = 0 if transitions is None else transitions[inner]
    r = np.mod(theta[fr] - theta[fl] - phi - k * np.pi / 2.0 + np.pi / 4.0,
               np.pi / 2.0) - np.pi / 4.0
    out = np.zeros(mesh.edge_face_count.sum(), dtype=float)
    return r


def vertex_singularity_indices(mesh: TriMesh, theta, u, v):
    fL, fR = mesh.edge_faces()
    inner = fR >= 0
    fl, fr = fL[inner], fR[inner]
    e_lo, e_hi = mesh.edges[inner, 0], mesh.edges[inner, 1]
    d = mesh.V[e_hi] - mesh.V[e_lo]

    def ang(fidx):
        return np.arctan2(np.sum(d * v[fidx], axis=1),
                          np.sum(d * u[fidx], axis=1))

    phi = ang(fl) - ang(fr)
    jump = np.mod(theta[fr] - theta[fl] - phi + np.pi / 4.0,
                  np.pi / 2.0) - np.pi / 4.0

    idx = np.zeros(mesh.n_vertices)

    V = mesh.V
    fl_all = fL[inner]
    lo, hi = mesh.edges[inner, 0], mesh.edges[inner, 1]
    F = mesh.F
    fv = F[fl_all]
    is_lo = fv == lo[:, None]
    is_hi = fv == hi[:, None]
    w = fv[~(is_lo | is_hi)].reshape(-1)

    n_fl = mesh.face_normals[fl_all]

    def _sign(vtx, other):
        a = V[other] - V[vtx]
        b = V[w] - V[vtx]
        return np.sign(np.sum(np.cross(a, b) * n_fl, axis=1))

    s_lo = _sign(lo, hi)
    s_hi = _sign(hi, lo)
    np.add.at(idx, lo, s_lo * jump)
    np.add.at(idx, hi, s_hi * jump)

    V, F = mesh.V, mesh.F
    tri = V[F]
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 1]
    l0 = np.linalg.norm(e0, axis=1)
    l1 = np.linalg.norm(e1, axis=1)
    l2 = np.linalg.norm(e2, axis=1)
    ang0 = np.arccos(np.clip(np.sum(e0 * e1, axis=1) / (l0 * l1), -1, 1))
    ang1 = np.arccos(np.clip(np.sum(-e0 * e2, axis=1) / (l0 * l2), -1, 1))
    ang2 = np.arccos(np.clip(np.sum(e1 * e2, axis=1) / (l1 * l2), -1, 1))
    np.add.at(idx, F[:, 0], -ang0)
    np.add.at(idx, F[:, 1], -ang1)
    np.add.at(idx, F[:, 2], -ang2)
    idx += 2.0 * np.pi

    k4 = np.mod(np.round(idx / (np.pi / 2.0)).astype(np.int64), 4)
    return k4
