"""Seamless global UV solve on the cut mesh."""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .cut import build_cut_mesh, unwrap_field_branches
from .mesh import TriMesh

_DIRECT_LIMIT = 200000


def _direct_solve(H, b):
    try:
        Hs = ((H + H.conj().T) * 0.5).tocsc()
        lu = spla.splu(Hs, permc_spec="MMD_AT_PLUS_A",
                       diag_pivot_thresh=0.0,
                       options=dict(SymmetricMode=True))
        x = lu.solve(b)
        if np.all(np.isfinite(x)):
            return x
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spla.spsolve(H, b)


def _normal_solve(rr, cc, vv, rhs, ncols, pin, maxiter=20000):
    mask = ~pin[cc]
    rr, cc, vv = rr[mask], cc[mask], vv[mask]
    free = np.flatnonzero(~pin)
    red = np.full(ncols, -1, dtype=int)
    red[free] = np.arange(len(free))
    A = sp.coo_matrix((vv, (rr, red[cc])),
                      shape=(int(rr.max()) + 1, len(free))).tocsr()
    H = (A.conj().T @ A).tocsc()
    b = A.conj().T @ rhs
    if len(free) <= _DIRECT_LIMIT:
        # Direct sparse solve: for 2D-mesh sized systems SuperLU beats the
        # many CG iterations a stiff seam penalty needs.
        x = _direct_solve(H, b)
    else:
        d = H.diagonal().real
        M = sp.diags(1.0 / np.maximum(d, 1e-30))
        x, info = spla.cg(H, b, rtol=1e-11, maxiter=maxiter, M=M)
        if info != 0 or not np.all(np.isfinite(x)):
            x = _direct_solve(H, b)
    U = np.zeros(ncols, dtype=np.complex128)
    U[free] = x
    resid = np.linalg.norm(A @ x - rhs) / (np.linalg.norm(rhs) + 1e-30)
    return U, float(resid)

def _arap_rows(mesh, corner, thg, h, frame_u, frame_v):
    F = mesh.F
    nF = mesh.n_faces
    p0 = mesh.V[F[:, 0]]
    d1 = mesh.V[F[:, 1]] - p0
    d2 = mesh.V[F[:, 2]] - p0
    x1 = np.einsum("ij,ij->i", d1, frame_u)
    y1 = np.einsum("ij,ij->i", d1, frame_v)
    x2 = np.einsum("ij,ij->i", d2, frame_u)
    y2 = np.einsum("ij,ij->i", d2, frame_v)
    m = np.exp(-1j * thg) / h
    det = x1 * y2 - y1 * x2
    w = np.sqrt(np.maximum(0.5 * det, 1e-300))
    inv = 1.0 / det
    i00, i10 = y2 * inv, -y1 * inv
    i01, i11 = -x2 * inv, x1 * inv
    rows = np.arange(nF) * 2
    rr = np.empty(6 * nF, dtype=int)
    rr[0::6] = rr[1::6] = rr[2::6] = rows
    rr[3::6] = rr[4::6] = rr[5::6] = rows + 1
    cc = np.concatenate([corner, corner], axis=1).reshape(-1)
    vv = np.empty(6 * nF, dtype=np.complex128)
    vv[0::6] = -w * (i00 + i10)
    vv[1::6] = w * i00
    vv[2::6] = w * i10
    vv[3::6] = -w * (i01 + i11)
    vv[4::6] = w * i01
    vv[5::6] = w * i11
    rhs = np.empty(2 * nF, dtype=np.complex128)
    rhs[0::2] = w * m
    rhs[1::2] = w * m * 1j
    return rr, cc, vv, rhs, w, det

def _seam_stats(U, pairs, ks, h):
    if len(pairs) == 0:
        z = np.zeros(0)
        return z, z, z, z
    Rk = 1j ** ks
    ta = U[[p[4] for p in pairs]] - Rk * U[[p[2] for p in pairs]]
    tb = U[[p[3] for p in pairs]] - Rk * U[[p[1] for p in pairs]]
    t = 0.5 * (ta + tb)
    twist = np.abs(ta - tb)
    dL = U[[p[2] for p in pairs]] - U[[p[1] for p in pairs]]
    dR = U[[p[4] for p in pairs]] - U[[p[3] for p in pairs]]
    kk = np.where(np.abs(dL) > 1e-12, dL, 1.0)
    rot = np.mod(np.angle(dR / kk) - ks * (np.pi / 2.0) + np.pi,
                 2 * np.pi) - np.pi
    scale = np.abs(dR) / np.maximum(np.abs(dL), 1e-30)
    return rot, scale, t, twist

def _uv_metrics(U, corner, det, h):
    Uc = U[corner]
    a2 = np.imag(np.conj(Uc[:, 1] - Uc[:, 0]) * (Uc[:, 2] - Uc[:, 0]))
    flips = int(np.sum(a2 <= 0))
    ratio = a2 / np.maximum(0.5 * det, 1e-30) * h * h
    return flips, ratio

def solve_uv(mesh, theta, frame_u, frame_v, cut_edges=None,
             target_quads=1000, edge_length=None, seam_weight=100.0,
             maxiter=20000, seam_k=None):
    """Seamless least-squares UV solve on the cut mesh.

    ARAP per face to the 4-RoSy frame, seam edge rotations i^k.
    Returns UVs, cut-mesh corner map and seam diagnostics.
    """
    cut_edges = set() if cut_edges is None else set(int(e) for e in cut_edges)
    cm = build_cut_mesh(mesh, cut_edges)
    thg = unwrap_field_branches(mesh, theta, frame_u, frame_v, cut_edges)
    corner = cm["corner_uv"]
    n_uv = cm["n_uv"]
    nF = mesh.n_faces
    total_area = float(mesh.face_areas.sum())
    h = float(edge_length) if edge_length else np.sqrt(
        total_area / max(1.0, float(target_quads)))

    rr, cc, vv, rhs, w, det = _arap_rows(
        mesh, corner, thg, h, frame_u, frame_v)

    pin = np.zeros(n_uv, dtype=bool)
    for p in range(cm["n_patches"]):
        ids = np.flatnonzero(cm["uv_patch"] == p)
        if len(ids):
            pin[ids[0]] = True

    U, resid0 = _normal_solve(rr, cc, vv, rhs, n_uv, pin, maxiter)

    pairs = cm["cut_pairs"]
    ks = np.zeros(len(pairs), dtype=int)
    resid = resid0
    if len(pairs):
        if seam_k is not None:
            if isinstance(seam_k, dict):
                ks = np.array([int(seam_k.get(int(p[0]), 0))
                               for p in pairs], dtype=int)
            else:
                ks = np.asarray(seam_k, dtype=int).copy()
            ks %= 4
        else:
            dL = U[[p[2] for p in pairs]] - U[[p[1] for p in pairs]]
            dR = U[[p[4] for p in pairs]] - U[[p[3] for p in pairs]]
            ok = (np.abs(dL) > 1e-12) & (np.abs(dR) > 1e-12)
            ang = np.angle(dR / np.where(ok, dL, 1.0))
            ks = np.mod(np.round(ang / (np.pi / 2.0)).astype(int), 4)
            ks[~ok] = 0
        ws = seam_weight * float(np.mean(w))
        Rk = 1j ** ks
        sr = 2 * nF + np.arange(len(pairs))
        srr = np.concatenate([sr, sr, sr, sr])
        scc = np.concatenate([[p[4] for p in pairs], [p[3] for p in pairs],
                              [p[2] for p in pairs], [p[1] for p in pairs]])
        svv = np.concatenate([np.full(len(pairs), ws, dtype=np.complex128),
                              np.full(len(pairs), -ws, dtype=np.complex128),
                              -ws * Rk, ws * Rk])
        srhs = np.zeros(len(pairs), dtype=np.complex128)
        U, resid = _normal_solve(np.concatenate([rr, srr]),
                                 np.concatenate([cc, scc]),
                                 np.concatenate([vv, svv]),
                                 np.concatenate([rhs, srhs]),
                                 n_uv, pin, maxiter)
    rot_err, scale, t, twist = _seam_stats(U, pairs, ks, h)
    flips, area_ratio = _uv_metrics(U, corner, det, h)

    return {"uv": np.stack([U.real, U.imag], axis=1),
            "u_complex": U, "corner_uv": corner, "n_uv": n_uv,
            "patch_of_face": cm["patch_of_face"],
            "n_patches": cm["n_patches"], "theta_global": thg, "h": h,
            "edge_length": h, "cut_pairs": pairs,
            "seam_k": ks, "seam_rot_err": rot_err, "seam_scale": scale,
            "seam_t": t, "seam_twist": twist,
            "arap_resid": resid0, "resid": resid,
            "flips": flips, "area_ratio": area_ratio,
            "pins": pin, "frame_u": frame_u, "frame_v": frame_v}


