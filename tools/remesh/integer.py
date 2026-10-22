"""Integer seam offsets and the fold-free integer UV layout."""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize

from ._backend import accelerator
from .mesh import TriMesh
from .param import _arap_rows, _normal_solve, _seam_stats, _uv_metrics


class RemeshCancelled(Exception):
    """Raised inside the pipeline when the UI asks to stop."""


def seam_fan_cycles(mesh, uvres):
    """Closed corner-group cycles around cut vertices."""
    pairs = uvres["cut_pairs"]
    k_of = {int(p[0]): int(k) for p, k in zip(pairs, uvres["seam_k"])}
    t_of = {int(p[0]): complex(t) for p, t in zip(pairs, uvres["seam_t"])}
    inc_at = defaultdict(list)
    for p in pairs:
        e = int(p[0])
        a, b = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
        inc_at[a].append((e, int(p[1]), int(p[3])))
        inc_at[b].append((e, int(p[2]), int(p[4])))
    cycles = []
    for v, inc in inc_at.items():
        adj = defaultdict(list)
        for e, ca, cb in inc:
            adj[ca].append((e, 1, cb))
            adj[cb].append((e, -1, ca))
        visited = set()
        for u0 in list(adj):
            for e0, s0, w0 in list(adj[u0]):
                if (u0, e0, s0) in visited:
                    continue
                A = 1.0 + 0j
                B = defaultdict(complex)
                u, e, s, w = u0, e0, s0, w0
                closed = False
                for _ in range(len(inc) + 2):
                    visited.add((u, e, s))
                    visited.add((w, e, -s))
                    R = 1j ** k_of[e]
                    if s == 1:
                        A *= R
                        for kk in list(B):
                            B[kk] *= R
                        B[e] += 1.0
                    else:
                        Ri = np.conj(R)
                        A *= Ri
                        for kk in list(B):
                            B[kk] *= Ri
                        B[e] -= Ri
                    u = w
                    if u == u0:
                        closed = True
                        break
                    nxt = None
                    for t in adj[u]:
                        if (u, t[0], t[1]) not in visited:
                            nxt = t
                            break
                    if nxt is None:
                        break
                    e, s, w = nxt
                if closed:
                    cycles.append((v, u0, A, dict(B)))
    return cycles, t_of

def seam_nullspace(uvres, tol=1e-6):
    """Homogeneous seam dofs: corner -> (root, unit rotation, pinned)."""
    pairs = uvres["cut_pairs"]
    n_uv = int(uvres["corner_uv"].max()) + 1
    parent = np.arange(n_uv)
    rel = np.ones(n_uv, dtype=np.complex128)
    pinned = np.zeros(n_uv, dtype=bool)

    def find(x):
        nodes, rs = [], []
        y = x
        while parent[y] != y:
            nodes.append(y)
            rs.append(rel[y])
            y = parent[y]
        root = y
        acc = 1.0 + 0j
        for k in range(len(nodes) - 1, -1, -1):
            acc *= rs[k]
            parent[nodes[k]] = root
            rel[nodes[k]] = acc
        return root, (rel[x] if nodes else 1.0 + 0j)

    for p, k in zip(pairs, uvres["seam_k"]):
        R = 1j ** int(k)
        for u, v in ((int(p[1]), int(p[3])), (int(p[2]), int(p[4]))):
            ru, au = find(u)
            rv, av = find(v)
            Rrel = R * au / av
            if ru == rv:
                if abs(Rrel - 1.0) > tol:
                    pinned[ru] = True
            else:
                parent[rv] = ru
                rel[rv] = Rrel
                pinned[ru] |= pinned[rv]
    roots = np.empty(n_uv, dtype=int)
    rots = np.empty(n_uv, dtype=np.complex128)
    for x in range(n_uv):
        roots[x], rots[x] = find(x)
    return roots, rots, pinned[roots]

def seam_constraint_matrix(uvres):
    pairs = uvres["cut_pairs"]
    Rk = 1j ** uvres["seam_k"]
    n_uv = int(uvres["corner_uv"].max()) + 1
    rows, cols, vals = [], [], []
    for i, p in enumerate(pairs):
        rows += [2 * i, 2 * i, 2 * i + 1, 2 * i + 1]
        cols += [int(p[3]), int(p[1]), int(p[4]), int(p[2])]
        vals += [1.0, -Rk[i], 1.0, -Rk[i]]
    return sp.coo_matrix((vals, (rows, cols)),
                         shape=(2 * len(pairs), n_uv)).tocsr()

def _gauss_unit(c):
    return 1j ** (int(round(np.angle(c) / (np.pi / 2.0))) % 4)

def integer_lattice_offsets(mesh, uvres, weights=None, order="reverse",
                            tol=1e-6, sweeps=100, snap_skip=()):
    """Globally consistent Gaussian-integer seam offsets.

    Every identity-rotation corner cycle is a row over Gaussian-unit
    coefficients.  Row-echelon elimination over those units keeps all
    coefficients integral; pivot offsets follow exactly from the free
    ones, so every cycle closes with zero residual.
    """
    pairs = uvres["cut_pairs"]
    E = len(pairs)
    if E == 0:
        return {"n": np.zeros(0, complex), "dev": np.zeros(0), "viol": 0.0,
                "sweeps": 0, "F": 0, "t": np.zeros(0)}
    cycles, t_of = seam_fan_cycles(mesh, uvres)
    eid = [int(p[0]) for p in pairs]
    eidx = {e: i for i, e in enumerate(eid)}
    t = np.array([t_of[e] for e in eid], dtype=np.complex128)
    cons = []
    for v, u0, A, B in cycles:
        if abs(A - 1.0) > tol:
            continue
        cons.append({eidx[e]: _gauss_unit(c) for e, c in B.items()})

    pivot_row, creation = {}, []
    for co in cons:
        r = dict(co)
        work = list(r)
        while work:
            p = work.pop()
            if p not in r or p not in pivot_row:
                continue
            times = r[p]
            for e, c in pivot_row[p].items():
                nv = r.get(e, 0) - times * c
                if abs(nv) < 1e-12:
                    r.pop(e, None)
                else:
                    r[e] = nv
                    if e not in work:
                        work.append(e)
            r.pop(p, None)
        if not r:
            continue
        p = min(r)
        cp = r[p]
        r = {e: c * np.conj(cp) for e, c in r.items()}
        pivot_row[p] = r
        creation.append(p)
    free = [e for e in range(E) if e not in pivot_row]
    fidx = {e: j for j, e in enumerate(free)}
    F = len(free)

    exprs = {}
    for p in reversed(creation):
        terms = {}
        for e, c in pivot_row[p].items():
            if e == p:
                continue
            w = -c
            if e in exprs:
                for j, a in exprs[e].items():
                    terms[j] = terms.get(j, 0) + w * a
            else:
                j = fidx[e]
                terms[j] = terms.get(j, 0) + w
        exprs[p] = terms

    rows, cols, vals = [], [], []
    for e, j in fidx.items():
        rows.append(e)
        cols.append(j)
        vals.append(1.0 + 0j)
    for p, terms in exprs.items():
        for j, a in terms.items():
            rows.append(p)
            cols.append(j)
            vals.append(a)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(E, F)).tocsc()
    w = np.ones(E) if weights is None else np.asarray(weights, float)
    # Sparse normal equations: H is F x F and sparse, but on production
    # meshes F can be tens of thousands, so never densify it.
    H = (A.conj().T @ sp.diags(w) @ A).tocsc()
    b = np.asarray(A.conj().T @ (w * t)).ravel()
    if F:
        Hreg = (H + sp.identity(F, format="csc", dtype=np.complex128) * 1e-9)
        try:
            X = spla.spsolve(Hreg, b)
            if not np.all(np.isfinite(X)):
                raise RuntimeError("spsolve produced non-finite values")
        except Exception:
            X = spla.lsqr(Hreg, b, atol=1e-10, btol=1e-10,
                          iter_lim=max(1000, 4 * F))[0]
    else:
        X = np.zeros(0, dtype=np.complex128)
    X = np.round(X.real) + 1j * np.round(X.imag)
    res = np.asarray(A @ X).ravel() - t
    g = np.ascontiguousarray(np.asarray(A.conj().T @ (w * res)).ravel(),
                             dtype=np.complex128)

    Hcsc = H.tocsc()
    h_indptr = np.ascontiguousarray(Hcsc.indptr, dtype=np.int64)
    h_indices = np.ascontiguousarray(Hcsc.indices, dtype=np.int64)
    h_data = np.ascontiguousarray(Hcsc.data, dtype=np.complex128)
    diag = np.ascontiguousarray(Hcsc.diagonal().real, dtype=np.float64)

    def _sweep_order():
        if isinstance(order, (int, np.integer)):
            return np.random.default_rng(int(order)).permutation(F)
        if isinstance(order, (list, tuple, np.ndarray)):
            return np.asarray(order, dtype=np.int64).ravel()
        if order == "reverse":
            return np.arange(F - 1, -1, -1, dtype=np.int64)
        return np.arange(F, dtype=np.int64)

    accel = accelerator()
    sweeps_run = 0
    for _ in range(sweeps):
        js = _sweep_order()
        if accel is not None:
            changed = accel.integer_sweeps(h_indptr, h_indices, h_data,
                                           diag, g, X,
                                           np.ascontiguousarray(
                                               js, dtype=np.int64))
        else:
            changed = False
            for j in js:
                best, bd = 0j, 0.0
                for d in (1.0, -1.0, 1j, -1j):
                    delta = 2.0 * float(np.real(np.conj(d) * g[j])) + \
                        float(diag[j])
                    if delta < bd - 1e-12:
                        bd, best = delta, d
                if best != 0:
                    X[j] += best
                    lo, hi = h_indptr[j], h_indptr[j + 1]
                    g[h_indices[lo:hi]] += h_data[lo:hi] * best
                    changed = True
        sweeps_run += 1
        if not changed:
            break

    n = np.asarray(A @ X).ravel()
    pole_infos = _fan_poles(mesh, uvres, tol)
    n, X, snap = _snap_pole_offsets(pole_infos, A, X, n, t, tol,
                                    skip=snap_skip)
    viol = max((abs(sum(c * n[e] for e, c in co.items()))
                for co in cons), default=0.0)
    return {"n": n, "dev": np.abs(n - t), "viol": viol,
            "sweeps": sweeps_run,
            "F": F, "t": t, "X": X, "A": A, "free": np.array(free, dtype=int),
            "snap": snap, "poles": snap.get("poles", []),
            "pole_valence": snap.get("pole_valence", [])}

def _fan_poles(mesh, uvres, tol=1e-6):
    """Full vertex-fan holonomy z -> A z + N of the seam transitions."""
    pairs = uvres.get("cut_pairs")
    if pairs is None or len(pairs) == 0:
        return []
    eid = [int(p[0]) for p in pairs]
    eidx = {e: i for i, e in enumerate(eid)}
    k_of = {int(p[0]): int(k) for p, k in zip(pairs, uvres["seam_k"])}
    fL, fR = mesh.edge_faces()
    Fm = mesh.F
    fid = mesh.face_edge_ids
    verts = set()
    for e in eid:
        verts.add(int(mesh.edges[e, 0]))
        verts.add(int(mesh.edges[e, 1]))
    out = []
    for v in sorted(verts):
        f0 = int(mesh.vertex_faces(v)[0])
        f = f0
        edges = []
        while True:
            k = int(np.flatnonzero(Fm[f] == v)[0])
            e = int(fid[f, k])
            edges.append((f, e))
            g = int(fR[e]) if int(fL[e]) == f else int(fL[e])
            if g < 0 or g == f0 or len(edges) > 4096:
                break
            f = g
        if g < 0:
            continue
        A = 1.0 + 0j
        co = {}
        for f, e in edges:
            if e not in k_of:
                continue
            Rs = 1j ** k_of[e]
            sc = 1.0 + 0j
            if int(fL[e]) != f:
                Rs = np.conj(Rs)
                sc = -Rs
            for kk in list(co):
                co[kk] *= Rs
            co[eidx[e]] = co.get(eidx[e], 0j) + sc
            A *= Rs
        if abs(A - 1.0) <= tol:
            continue
        if min(abs(A - 1j), abs(A + 1j)) > tol and abs(A + 1.0) > tol:
            continue
        co = {i: _gauss_unit(c) for i, c in co.items() if abs(c) > 1e-9}
        out.append((int(v), A, co))
    return out

def close_cone_skip(mesh, uvres, singular_vertices, factor=0.3):
    """Regular-vertex cones closer than factor*h to another pole."""
    poles = _fan_poles(mesh, uvres)
    h = float(uvres.get("h", 0.0) or 0.0)
    if not poles or h <= 0:
        return set()
    sing = set(int(v) for v in singular_vertices)
    P = np.array([mesh.V[int(v)] for v, _, _ in poles])
    skip = set()
    for i, (v, _, _) in enumerate(poles):
        if int(v) in sing:
            continue
        d = np.linalg.norm(P - P[i], axis=1)
        d[i] = np.inf
        if d.min() < factor * h:
            skip.add(int(v))
    return skip

def _snap_pole_offsets(pole_infos, A, X, n, t, tol=1e-6, skip=()):
    """Move free offset parities so +-90 poles land on integer UV."""
    infos = []
    for v, Arot, co in pole_infos:
        N = sum(c * n[i] for i, c in co.items())
        infos.append((int(v), Arot, co, N))
    val = {int(v): (5 if abs(A - 1j) <= tol else 3)
           for v, A, co, N in infos}
    if not infos:
        return n, X, {"snapped": 0, "parity_ok": True, "poles": [],
                      "pole_valence": []}
    skip = set(int(v) for v in skip)
    cons = [inf for inf in infos if inf[0] not in skip]
    meta = {"poles": [v for v, *_ in cons],
            "pole_valence": [val[v] for v, *_ in cons],
            "skipped": sorted(skip)}
    if not cons:
        return n, X, dict(meta, snapped=0, parity_ok=True)
    rows, rhs = [], []
    for v, Arot, co, N in cons:
        if abs(Arot + 1.0) <= tol:
            rows.append((co, "re"))
            rhs.append(int(round(N.real)) % 2)
            rows.append((co, "im"))
            rhs.append(int(round(N.imag)) % 2)
        else:
            rows.append((co, "sum"))
            rhs.append(int(round(N.real + N.imag)) % 2)
    nf = A.shape[1]
    col = {}
    for j in range(nf):
        col[(j, 0)] = len(col)
        col[(j, 1)] = len(col)
    M = np.zeros((len(rows), len(col)), dtype=np.int8)
    Acsr = A.tocsr()
    a_indptr = Acsr.indptr
    a_indices = Acsr.indices
    a_data = Acsr.data
    # iterate the sparse lattice rows directly - densifying them costs
    # O(F) float64 per touched seam and explodes RAM on large meshes
    for ri, (co, typ) in enumerate(rows):
        for i, c in co.items():
            for k in range(int(a_indptr[i]), int(a_indptr[i + 1])):
                z = c * a_data[k]
                if abs(z) < 0.5:
                    continue
                j = int(a_indices[k])
                if typ == "re":
                    v0, v1 = z.real, -z.imag
                elif typ == "im":
                    v0, v1 = z.imag, z.real
                else:
                    v0 = v1 = z.real + z.imag
                M[ri, col[(j, 0)]] ^= int(round(v0)) & 1
                M[ri, col[(j, 1)]] ^= int(round(v1)) & 1
    sol = _gf2_solve(M, np.array(rhs, dtype=np.int8))
    if sol is None:
        return n, X, dict(meta, snapped=0, parity_ok=False)
    X2 = X.copy()
    for j in range(nf):
        d = 0j
        if sol[col[(j, 0)]]:
            d += 1.0
        if sol[col[(j, 1)]]:
            d += 1j
        X2[j] += d
    n2 = np.asarray(A @ X2).ravel()
    for v, Arot, co, N in cons:
        N2 = sum(c * n2[i] for i, c in co.items())
        if abs(Arot + 1.0) <= tol:
            fr = (N2.real / 2.0, N2.imag / 2.0)
        else:
            fr = ((N2.real - N2.imag) / 2.0, (N2.real + N2.imag) / 2.0)
        if max(abs(f - round(f)) for f in fr) > 1e-6:
            return n, X, dict(meta, snapped=0, parity_ok=False)
    return n2, X2, dict(meta, snapped=int(sol.sum()), parity_ok=True,
                        pole_dev=float(np.abs(n2 - t).max()))

def _gf2_solve(M, b):
    """Gaussian elimination over GF(2); None if inconsistent.

    Rows are packed into Python bigints so the elimination XORs run at
    word speed; the old int8-array version was O(m^2 * nc) byte traffic
    and dominated the pole snap on large meshes.
    """
    m, nc = M.shape
    packed = np.packbits(np.asarray(M) != 0, axis=1, bitorder="little")
    rows = [int.from_bytes(row.tobytes(), "little") for row in packed]
    bvec = [int(x) for x in np.asarray(b) != 0]
    piv = []
    r = 0
    for c in range(nc):
        bit = 1 << c
        pr = -1
        for rr in range(r, m):
            if rows[rr] & bit:
                pr = rr
                break
        if pr < 0:
            continue
        if pr != r:
            rows[r], rows[pr] = rows[pr], rows[r]
            bvec[r], bvec[pr] = bvec[pr], bvec[r]
        prow = rows[r]
        pb = bvec[r]
        for rr in range(m):
            if rr != r and (rows[rr] & bit):
                rows[rr] ^= prow
                bvec[rr] ^= pb
        piv.append(c)
        r += 1
        if r == m:
            break
    for rr in range(r, m):
        if bvec[rr]:
            return None
    x = np.zeros(nc, dtype=np.int8)
    for i, c in enumerate(piv):
        x[c] = bvec[i]
    return x

def _fixed_offset_solve(mesh, uvres, rr, cc, vv, rhs, w, n_int,
                        seam_weight=1000.0, maxiter=20000):
    pairs = uvres["cut_pairs"]
    nF = mesh.n_faces
    n_uv = uvres["n_uv"]
    Rk = 1j ** uvres["seam_k"]
    ws = seam_weight * float(np.mean(w))
    sr = 2 * nF + np.arange(2 * len(pairs))
    srr = np.concatenate([sr[0::2], sr[0::2], sr[1::2], sr[1::2]])
    scc = np.concatenate([[int(p[3]) for p in pairs],
                          [int(p[1]) for p in pairs],
                          [int(p[4]) for p in pairs],
                          [int(p[2]) for p in pairs]])
    svv = np.concatenate([np.full(len(pairs), ws, dtype=np.complex128),
                          -ws * Rk,
                          np.full(len(pairs), ws, dtype=np.complex128),
                          -ws * Rk])
    srhs = np.zeros(2 * len(pairs), dtype=np.complex128)
    srhs[0::2] = ws * n_int
    srhs[1::2] = ws * n_int
    pin = np.zeros(n_uv, dtype=bool)
    pin[0] = True
    return _normal_solve(np.concatenate([rr, srr]),
                         np.concatenate([cc, scc]),
                         np.concatenate([vv, svv]),
                         np.concatenate([rhs, srhs]),
                         n_uv, pin, maxiter)

def _enforce_seam_offsets(U, C, n_int):
    rhs = C @ U - np.repeat(n_int, 2)
    d = spla.lsmr(C, rhs, atol=1e-14, btol=1e-14, maxiter=10000)[0]
    U = U - d
    return U, float(np.abs(C @ U - np.repeat(n_int, 2)).max())

def untangle_uv(mesh, uvres, n_int, U0, eps=None, max_restarts=4,
                maxiter=1500, log=None, report=None, cancel_check=None):
    """Fold removal in the exact seam nullspace (L-BFGS).

    ``log(message, level)`` receives narrated updates; ``report(nit,
    energy, restart)`` is called every 50 optimizer iterations so the
    UI can show that the solve is alive.  ``cancel_check`` is polled in
    the L-BFGS callback and before each restart; when it returns True a
    :class:`RemeshCancelled` is raised (which aborts scipy's minimize).
    """
    def say(message: str, level: str = "info") -> None:
        if log is not None:
            log(message, level)

    def _check():
        if cancel_check is not None and cancel_check():
            raise RemeshCancelled("integer untangle cancelled")

    corner = np.ascontiguousarray(uvres["corner_uv"], dtype=np.int64)
    roots, rots, pin = seam_nullspace(uvres)
    n_uv = len(roots)
    free = np.unique(roots[~pin])
    nv = len(free)
    col = np.full(n_uv, -1, dtype=int)
    col[~pin] = np.searchsorted(free, roots[~pin])
    valid = col >= 0

    def areas(U):
        Uc = U[corner]
        return np.imag(np.conj(Uc[:, 1] - Uc[:, 0]) * (Uc[:, 2] - Uc[:, 0]))

    a0 = areas(U0)
    if eps is None:
        eps = 0.01 * float(np.median(np.maximum(a0, 0.0)))
        eps = max(eps, 1e-9)
    U = U0.copy()
    nfev = 0
    res = None
    # live optimizer state, read by the iteration callback
    last = {"E": float("inf"), "nit": 0}

    accel = accelerator()
    if accel is not None:
        vidx = np.flatnonzero(valid).astype(np.int64)
        vcol = np.ascontiguousarray(col[vidx], dtype=np.int64)
        rot_re = np.ascontiguousarray(rots.real, dtype=np.float64)
        rot_im = np.ascontiguousarray(rots.imag, dtype=np.float64)
        ub_re = np.empty(n_uv, dtype=np.float64)
        ub_im = np.empty(n_uv, dtype=np.float64)
        zc_re = np.empty(n_uv, dtype=np.float64)
        zc_im = np.empty(n_uv, dtype=np.float64)
        uc_re = np.empty(n_uv, dtype=np.float64)
        uc_im = np.empty(n_uv, dtype=np.float64)
        gc_re = np.empty(n_uv, dtype=np.float64)
        gc_im = np.empty(n_uv, dtype=np.float64)
        gz_re = np.empty(nv, dtype=np.float64)
        gz_im = np.empty(nv, dtype=np.float64)

    for k in range(max_restarts):
        _check()
        Ubase = U

        if accel is not None:
            ub_re[:] = Ubase.real
            ub_im[:] = Ubase.imag

            def fun(x):
                nonlocal nfev
                g = np.empty(2 * nv)
                energy = accel.untangle_eval(
                    x, ub_re, ub_im, rot_re, rot_im, vidx, vcol, corner,
                    eps, zc_re, zc_im, uc_re, uc_im, gc_re, gc_im,
                    gz_re, gz_im, g)
                last["E"] = energy
                nfev += 1
                return energy, g
        else:
            def fun(x):
                nonlocal nfev
                z = x[0::2] + 1j * x[1::2]
                Zc = np.zeros(n_uv, dtype=np.complex128)
                Zc[valid] = z[col[valid]]
                Uc_ = Ubase + rots * Zc
                a = areas(Uc_)
                d = np.maximum(0.0, eps - a)
                E = float(np.sum(d * d))
                f = np.flatnonzero(a < eps)
                gcorn = np.zeros(n_uv, dtype=np.complex128)
                if len(f):
                    Uf = Uc_[corner[f]]
                    x_, y_ = Uf.real, Uf.imag
                    x0, x1, x2 = x_[:, 0], x_[:, 1], x_[:, 2]
                    y0, y1, y2 = y_[:, 0], y_[:, 1], y_[:, 2]
                    gx = np.stack([y1 - y2, y2 - y0, y0 - y1], axis=1)
                    gy = np.stack([x2 - x1, x0 - x2, x1 - x0], axis=1)
                    gF = (gx + 1j * gy) * (-2.0 * d[f])[:, None]
                    np.add.at(gcorn, corner[f], gF)
                gz = np.zeros(nv, dtype=np.complex128)
                np.add.at(gz, col[valid], gcorn[valid] * np.conj(rots[valid]))
                g = np.empty(2 * nv)
                g[0::2] = gz.real
                g[1::2] = gz.imag
                last["E"] = E
                nfev += 1
                return E, g

        if k > 0:
            a_prev = areas(U)
            say(f"integer: untangle restart {k + 1}/{max_restarts} "
                f"({int(np.sum(a_prev <= 0))} folded faces remain)")

        def cb(xk):
            _check()
            last["nit"] += 1
            if report is not None and last["nit"] % 50 == 0:
                report(last["nit"], last["E"], k + 1)

        res = minimize(fun, np.zeros(2 * nv), jac=True, method="L-BFGS-B",
                       callback=cb,
                       options=dict(maxiter=maxiter, maxls=50, ftol=0.0,
                                    gtol=1e-14, maxcor=20, maxfun=10 * maxiter))
        z = res.x[0::2] + 1j * res.x[1::2]
        Zc = np.zeros(n_uv, dtype=np.complex128)
        Zc[valid] = z[col[valid]]
        U = Ubase + rots * Zc
        a = areas(U)
        if int(np.sum(a <= 0)) == 0 and float(a.min()) >= 0.5 * eps:
            break
    a = areas(U)
    return {"U": U, "flips": int(np.sum(a <= 0)), "min_area": float(a.min()),
            "eps": float(eps), "nit": int(res.nit), "nfev": int(nfev),
            "restarts": k + 1, "roots": roots, "rots": rots, "pinned": pin}

def _finish_trial(uvres, prep, unt):
    """Assemble the picklable trial dict from prep + untangle results."""
    U = unt["U"]
    n_int = prep["n"]
    pairs = uvres["cut_pairs"]
    Rk = 1j ** uvres["seam_k"]
    d1 = U[[int(p[4]) for p in pairs]] - \
        Rk * U[[int(p[2]) for p in pairs]] - n_int
    d2 = U[[int(p[3]) for p in pairs]] - \
        Rk * U[[int(p[1]) for p in pairs]] - n_int
    return {"order": prep["label"], "flips": unt["flips"],
            "min_area": unt["min_area"], "dev": prep["dev"],
            "sweeps": prep["sweeps"], "n": n_int, "U": U,
            "poles": prep.get("poles", []),
            "pole_valence": prep.get("pole_valence", []),
            "untangle": {kk: unt[kk] for kk in
                         ("flips", "min_area", "eps", "nit", "nfev",
                          "restarts")},
            "resid": prep["resid"],
            "seam_int_resid": float(max(np.abs(d1).max(),
                                        np.abs(d2).max())),
            "F": int(prep["F"])}


def _trial_prep_job(payload):
    """Lattice offsets + seam-constrained solve for one trial.

    Worker-process entry point for the GPU-batched search: everything
    except the untangle, which runs afterwards for all trials at once
    on the GPU (see :mod:`remesh.gpu_untangle`).  Also computes the
    untangle epsilon when not supplied (same rule as untangle_uv).
    """
    mesh = TriMesh(payload["V"], payload["F"])
    uvres = payload["uvres"]
    lat = integer_lattice_offsets(mesh, uvres, order=payload["order"],
                                  snap_skip=payload["snap_skip"])
    n_int = lat["n"]
    U, resid = _fixed_offset_solve(
        mesh, uvres, payload["rr"], payload["cc"], payload["vv"],
        payload["rhs"], payload["w"], n_int,
        payload["seam_weight"], payload["maxiter"])
    U, _ = _enforce_seam_offsets(U, seam_constraint_matrix(uvres), n_int)
    if payload["eps"] is not None:
        eps = float(payload["eps"])
    else:
        Uc = U[uvres["corner_uv"]]
        a0 = np.imag(np.conj(Uc[:, 1] - Uc[:, 0]) *
                     (Uc[:, 2] - Uc[:, 0]))
        eps = max(0.01 * float(np.median(np.maximum(a0, 0.0))), 1e-9)
    return {"label": payload["label"], "n": n_int, "U": U, "eps": eps,
            "dev": lat["dev"], "sweeps": lat["sweeps"],
            "F": int(lat["F"]),
            "poles": lat.get("poles", []),
            "pole_valence": lat.get("pole_valence", []),
            "resid": resid}


def _trial_job(payload):
    """One integer-offset lattice trial (worker-process entry point).

    Runs the same steps as the in-process path - lattice offsets for the
    given free-variable order, the seam-constrained solve and L-BFGS fold
    removal - returning a picklable trial dict.  Kept at module level so
    ProcessPoolExecutor can submit it.

    Optional Manager-queue channels (picklable proxies):
      * ``prog_queue``  - worker -> parent: (label, nit, energy, restart)
        pushed by the untangle solver every 50 iterations.
      * ``ctrl_queue``  - parent -> workers: a "stop" message makes the
        worker raise RemeshCancelled at its next optimizer callback.
    """
    label = payload["label"]
    ctrl = payload.get("ctrl_queue")
    stop = [False]

    def worker_cancel():
        if ctrl is not None and not stop[0]:
            try:
                while True:
                    if ctrl.get_nowait() == "stop":
                        stop[0] = True
            except Exception:  # empty queue / proxy issues -> keep going
                pass
        return stop[0]

    prog_q = payload.get("prog_queue")

    def worker_report(nit, energy, restart):
        if prog_q is not None:
            try:
                prog_q.put((label, int(nit), float(energy), int(restart)))
            except Exception:
                pass

    mesh = TriMesh(payload["V"], payload["F"])
    uvres = payload["uvres"]
    order = payload["order"]
    lat = integer_lattice_offsets(mesh, uvres, order=order,
                                  snap_skip=payload["snap_skip"])
    n_int = lat["n"]
    U, resid = _fixed_offset_solve(
        mesh, uvres, payload["rr"], payload["cc"], payload["vv"],
        payload["rhs"], payload["w"], n_int,
        payload["seam_weight"], payload["maxiter"])
    U, _ = _enforce_seam_offsets(U, seam_constraint_matrix(uvres), n_int)
    unt = untangle_uv(mesh, uvres, n_int, U, eps=payload["eps"],
                      max_restarts=payload["max_restarts"],
                      maxiter=payload["untangle_maxiter"],
                      report=worker_report, cancel_check=worker_cancel)
    prep = {"label": label, "n": n_int, "dev": lat["dev"],
            "sweeps": lat["sweeps"], "F": int(lat["F"]),
            "poles": lat.get("poles", []),
            "pole_valence": lat.get("pole_valence", []), "resid": resid}
    return _finish_trial(uvres, prep, unt)


def solve_integer_uv(mesh, uvres, seam_weight=1000.0,
                     orders=("reverse", "forward"), eps=None,
                     max_restarts=4, maxiter=20000, untangle_maxiter=1500,
                     random_trials=12, seed=0, snap_skip=(),
                     log=None, report=None, cancel_check=None):
    """Integer seam offsets + constrained solve + untangle.

    ``log(message, level)`` narrates each lattice trial; ``report(frac,
    message)`` tracks progress through the trials (0..1 within this
    stage) so a long solve stays visible in the UI.  ``cancel_check``
    is polled between trials and inside each untangle solve.
    """
    def say(message: str, level: str = "info") -> None:
        if log is not None:
            log(message, level)

    def _check():
        if cancel_check is not None and cancel_check():
            raise RemeshCancelled("integer offsets cancelled")

    pairs = uvres["cut_pairs"]
    if len(pairs) == 0:
        out = dict(uvres)
        out.update(integer_offsets=np.zeros(0, complex), round_resid=0.0,
                   seam_int_resid=0.0, min_uv_area=0.0, lattice_trials=[],
                   round_resid_lattice=0.0, poles=[], pole_valence=[])
        return out
    corner = uvres["corner_uv"]
    n_uv = uvres["n_uv"]
    nF = mesh.n_faces
    h = uvres["h"]
    ks = uvres["seam_k"]
    Rk = 1j ** ks
    rr, cc, vv, rhs, w, det = _arap_rows(
        mesh, corner, uvres["theta_global"], h,
        uvres["frame_u"], uvres["frame_v"])
    A_arap = sp.coo_matrix((vv, (rr, cc)),
                           shape=(2 * nF, n_uv)).tocsr()
    C = seam_constraint_matrix(uvres)
    trials = []
    best = None
    trial_no = 0

    def run_trial(order):
        nonlocal trial_no
        trial_no += 1
        frac_base = min(0.95, (trial_no - 1) / n_planned)
        frac_step = 1.0 / n_planned
        kind = order if isinstance(order, str) else "random"
        say(f"integer: trial {trial_no}/{n_planned} (order {kind}) on "
            f"{len(pairs):,} seams ...")

        def unt_report(nit, energy, restart):
            if report is None:
                return
            frac = frac_base + frac_step * min(
                0.9, nit / max(1, untangle_maxiter))
            report(frac, f"trial {trial_no}: L-BFGS iter {nit} "
                         f"(restart {restart}), energy {energy:.4g}")

        lat = integer_lattice_offsets(mesh, uvres, order=order,
                                      snap_skip=snap_skip)
        n_int = lat["n"]
        U, resid = _fixed_offset_solve(mesh, uvres, rr, cc, vv, rhs, w,
                                       n_int, seam_weight, maxiter)
        U, _ = _enforce_seam_offsets(U, C, n_int)
        unt = untangle_uv(mesh, uvres, n_int, U, eps=eps,
                          max_restarts=max_restarts,
                          maxiter=untangle_maxiter,
                          log=log, report=unt_report,
                          cancel_check=cancel_check)
        U = unt["U"]
        d1 = U[[int(p[4]) for p in pairs]] - \
            Rk * U[[int(p[2]) for p in pairs]] - n_int
        d2 = U[[int(p[3]) for p in pairs]] - \
            Rk * U[[int(p[1]) for p in pairs]] - n_int
        trial = {"order": order, "flips": unt["flips"],
                 "min_area": unt["min_area"], "dev": lat["dev"],
                 "sweeps": lat["sweeps"], "n": n_int, "U": U,
                 "poles": lat.get("poles", []),
                 "pole_valence": lat.get("pole_valence", []),
                 "untangle": unt, "resid": resid,
                 "seam_int_resid": float(max(np.abs(d1).max(),
                                             np.abs(d2).max()))}
        return trial, int(lat["F"])

    def perfect(trial):
        return (trial["flips"] == 0 and
                trial["min_area"] >= 0.5 * trial["untangle"]["eps"])

    def _trial_payload(order, label):
        """Everything one trial needs; pickled to worker processes."""
        return {"V": mesh.V, "F": mesh.F, "uvres": uvres,
                "rr": rr, "cc": cc, "vv": vv, "rhs": rhs, "w": w,
                "order": order, "label": label,
                "seam_weight": seam_weight, "maxiter": maxiter,
                "untangle_maxiter": untangle_maxiter,
                "max_restarts": max_restarts, "eps": eps,
                "snap_skip": tuple(int(s) for s in snap_skip),
                "prog_queue": prog_q, "ctrl_queue": ctrl_q}
        # note: the UI's cancel_check (a threading.Event bound method) is
        # deliberately NOT in the payload - it cannot be pickled.  Worker
        # cancellation goes through the ctrl_queue instead.

    def consider(trial, label=None):
        nonlocal best
        trials.append(trial)
        tag = label if label is not None else str(trial_no)
        say(f"integer: trial {tag} result: flips {trial['flips']}, "
            f"min area {trial['min_area']:.3g} (eps "
            f"{trial['untangle']['eps']:.3g}), "
            f"{trial['untangle']['nit']} L-BFGS iters / "
            f"{trial['untangle']['nfev']} evals, "
            f"{trial['untangle']['restarts']} restarts")
        score = (trial["flips"], -trial["min_area"])
        if best is None or score < best[0]:
            best = (score, trial)

    flip_slack = max(2, min(32, mesh.n_faces // 5000))

    def acceptable(trial):
        """Good enough to skip the randomized restarts.

        A handful of numerically degenerate faces is harmless for quad
        extraction and common on dense meshes; only real folds (min_area
        well below zero) keep the search going.
        """
        eps = trial["untangle"]["eps"]
        return (trial["flips"] <= flip_slack and
                trial["min_area"] >= -0.01 * eps)

    queue = list(orders)
    if not queue:
        queue.append("reverse")
    n_planned = len(queue) + random_trials
    n_free = 0
    stop = False

    # Trials are independent (each starts from a different free-variable
    # ordering), so on big meshes they run in worker processes in
    # parallel; the search early-exits (cancelling the rest) as soon as
    # one trial is fold-free.  REMESH_PARALLEL_TRIALS=0 forces in-process.
    parallel = (os.environ.get("REMESH_PARALLEL_TRIALS", "").strip()
                not in ("0", "false", "False") and len(pairs) >= 2000)
    pool = None
    prog_q = ctrl_q = mgr = None
    if parallel:
        import concurrent.futures as cf
        import multiprocessing as mp
        try:
            workers = max(2, min(8, (os.cpu_count() or 2) - 1, n_planned))
            mgr = mp.Manager()          # queue host for worker progress
            prog_q = mgr.Queue()
            ctrl_q = mgr.Queue()
            pool = cf.ProcessPoolExecutor(max_workers=workers)
        except Exception as ex:
            say(f"integer: parallel pool unavailable ({ex}); running "
                "trials in-process", "warn")
            pool = None

    ran_parallel = False
    if pool is not None:
        # Fast path: workers only do the lattice + constrained solve
        # (the cheap part); the L-BFGS fold removal - the bottleneck -
        # then runs for all trials at once in a single batched GPU
        # solve.  Any failure falls through to the CPU trial loop.
        untangle_gpu = None
        if os.environ.get("REMESH_NO_GPU", "").strip().lower() not in \
                ("1", "true", "yes", "on"):
            try:
                from .gpu_untangle import (gpu_available as _gpu_available,
                                           untangle_batch)
                if _gpu_available():
                    untangle_gpu = untangle_batch
            except Exception:
                untangle_gpu = None
        if untangle_gpu is not None:
            try:
                rng = np.random.default_rng(seed)
                specs = [(o, str(o)) for o in queue]
                for i in range(random_trials):
                    specs.append((int(rng.integers(1, 1 << 30)),
                                  f"random-{i + 1}"))
                say(f"integer: dispatching {len(specs)} trials - lattice "
                    f"offsets on {workers} worker processes, fold removal "
                    "batched on GPU ...")
                live = {}
                last_emit = [0.0]

                def on_iter(label, nit_, energy, restart_):
                    live[label] = (nit_, energy, restart_)
                    now = time.monotonic()
                    if report is not None and now - last_emit[0] > 1.0:
                        last_emit[0] = now
                        txt = "; ".join(
                            f"{l} iter {n} (r{rs}) E {e:.3g}"
                            for l, (n, e, rs)
                            in sorted(live.items())[:3])
                        report(0.05, f"untangle: {txt}")

                # Stream: the GPU untangle runs as soon as a small batch
                # of preps is ready, overlapping it with the workers still
                # solving the rest; once a finished trial is fold-free (or
                # acceptable), the remaining preps are cancelled.
                futs = {}
                for o, lbl in specs:
                    futs[pool.submit(_trial_prep_job,
                                     _trial_payload(o, lbl))] = lbl
                ready = []
                n_prepped = 0
                good_trials = 0
                while futs or ready:
                    if cancel_check is not None and cancel_check():
                        raise RemeshCancelled("integer offsets cancelled")
                    done_set = set()
                    if futs:
                        done_set, _ = cf.wait(list(futs), timeout=0.5,
                                              return_when=cf.FIRST_COMPLETED)
                    for fut in done_set:
                        lbl = futs.pop(fut)
                        n_prepped += 1
                        try:
                            ready.append(fut.result())
                        except Exception as ex:
                            say(f"integer: prep '{lbl}' failed: {ex}",
                                "warn")
                    # first batches stay small so the GPU engages early;
                    # later ones are bigger to amortize the per-call setup
                    batch_cap = 4 if n_prepped <= 8 else 8
                    if not ready or (len(ready) < batch_cap and futs):
                        continue
                    batch, ready = ready, []
                    unts = untangle_gpu(
                        uvres,
                        [{"label": p["label"], "U0": p["U"],
                          "eps": p["eps"]} for p in batch],
                        max_restarts=max_restarts,
                        maxiter=untangle_maxiter,
                        log=say, on_iter=on_iter,
                        cancel_check=cancel_check)
                    for p, unt in zip(batch, unts):
                        trial = _finish_trial(uvres, p, unt)
                        good_trials += 1
                        n_free = int(trial.get("F", 0))
                        consider(trial, p["label"])
                        if perfect(trial) or acceptable(trial):
                            stop = True
                    if report is not None and best is not None:
                        report(min(0.95, n_prepped / n_planned),
                               f"untangle: best flips {best[0][0]}")
                    if stop:
                        for f2 in futs:
                            f2.cancel()
                        break
                ran_parallel = good_trials > 0
            except RemeshCancelled:
                # the CPU fallback below would reuse the pool; on cancel
                # nothing reuses it, so tear it down before propagating
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                if mgr is not None:
                    try:
                        mgr.shutdown()
                    except Exception:
                        pass
                raise
            except Exception as ex:
                import traceback
                say(f"integer: GPU batch untangle failed ({ex}); falling "
                    "back to CPU worker trials", "warn")
                for ln in traceback.format_exc().strip().splitlines():
                    say(f"integer: {ln}", "warn")
                ran_parallel = False
            finally:
                if ran_parallel:
                    # success path: the CPU block below is skipped, so the
                    # workers/manager must be torn down here or they linger
                    try:
                        pool.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                    if mgr is not None:
                        try:
                            mgr.shutdown()
                        except Exception:
                            pass

    if pool is not None and not ran_parallel:
        rng = np.random.default_rng(seed)
        specs = [(o, str(o)) for o in queue]
        for i in range(random_trials):
            specs.append((int(rng.integers(1, 1 << 30)), f"random-{i + 1}"))
        say(f"integer: dispatching {len(specs)} trials to {workers} worker "
            "processes (stops early once one is fold-free) ...")

        # Reader thread: aggregate live per-worker optimizer progress
        # (label -> latest iteration/energy) and mirror it into the UI
        # status line about once a second.
        live = {}
        state = {"frac": 0.0, "extra": ""}
        reader_stop = threading.Event()

        def _drain():
            last_emit = 0.0
            while not reader_stop.is_set():
                try:
                    item = prog_q.get(timeout=0.3)
                except Exception:
                    item = None
                if item is None:
                    continue
                lbl, nit, energy, restart = item
                live[lbl] = (nit, energy, restart)
                now = time.monotonic()
                if report is not None and now - last_emit > 1.0:
                    last_emit = now
                    txt = "; ".join(
                        f"{l} iter {n} (r{rs}) E {e:.3g}"
                        for l, (n, e, rs) in sorted(live.items())[:3])
                    report(state["frac"], f"running: {txt}")

        reader = threading.Thread(target=_drain, daemon=True)
        reader.start()
        try:
            futs = {}
            spec_iter = iter(specs)

            def _submit_next():
                for o, lbl in spec_iter:
                    futs[pool.submit(_trial_job,
                                     _trial_payload(o, lbl))] = lbl
                    break

            for _ in range(workers + 1):
                _submit_next()
            done = good = 0
            while futs:
                if (cancel_check is not None
                        and cancel_check() and not stop):
                    stop = True
                    raise RemeshCancelled("integer offsets cancelled")
                done_set, _ = cf.wait(list(futs), timeout=0.5,
                                      return_when=cf.FIRST_COMPLETED)
                if not done_set:
                    continue
                for fut in done_set:
                    lbl = futs.pop(fut)
                    done += 1
                    try:
                        trial = fut.result()
                    except cf.CancelledError:
                        continue
                    except Exception as ex:
                        say(f"integer: trial '{lbl}' failed: {ex}", "warn")
                        trial = None
                    if trial is not None:
                        good += 1
                        n_free = int(trial.get("F", 0))
                        consider(trial, lbl)
                        if perfect(trial) or acceptable(trial):
                            stop = True
                    state["frac"] = min(0.95, done / n_planned)
                    if report is not None:
                        report(state["frac"],
                               f"trial '{lbl}' done"
                               + (f", best flips {best[0][0]}"
                                  if best is not None else ""))
                if stop:
                    for f2 in futs:
                        f2.cancel()
                    break
                for _ in range(len(done_set)):
                    _submit_next()
        except RemeshCancelled:
            # tell the running workers to abort at their next callback
            if ctrl_q is not None:
                for _ in range(64):
                    try:
                        ctrl_q.put("stop")
                    except Exception:
                        break
            raise
        finally:
            reader_stop.set()
            reader.join(timeout=2.0)
            pool.shutdown(wait=False, cancel_futures=True)
            if mgr is not None:
                try:
                    mgr.shutdown()
                except Exception:
                    pass
        ran_parallel = good > 0
        if not ran_parallel:
            say("integer: all parallel trials failed; falling back to "
                "in-process trials", "warn")

    if not ran_parallel:
        for order in queue:
            _check()
            trial, n_free = run_trial(order)
            consider(trial)
            if perfect(trial):
                stop = True
                break

        if not stop and not acceptable(best[1]):
            # Deterministic orders left folds behind; try randomized free-
            # variable orderings to escape the integer lattice minima.
            say(f"integer: folds remain (flips {best[1]['flips']}, min area "
                f"{best[1]['min_area']:.3g}); running up to {random_trials} "
                "randomized trials ...", "warn")
            rng = np.random.default_rng(seed)
            for _ in range(random_trials):
                _check()
                trial, _ = run_trial(rng.permutation(n_free))
                consider(trial)
                if acceptable(trial):
                    break

    b = best[1]
    U, n_int = b["U"], b["n"]
    rot_err, scale, t2, twist = _seam_stats(U, pairs, ks, h)
    flips, ratio = _uv_metrics(U, corner, det, h)
    a = np.imag(np.conj(U[corner][:, 1] - U[corner][:, 0]) *
                (U[corner][:, 2] - U[corner][:, 0]))
    arap = float(np.linalg.norm(A_arap @ U - rhs) /
                 (np.linalg.norm(rhs) + 1e-300))
    out = dict(uvres)
    out.update(uv=np.stack([U.real, U.imag], axis=1), u_complex=U,
               seam_rot_err=rot_err, seam_scale=scale, seam_t=t2,
               seam_twist=twist, flips=flips, area_ratio=ratio,
               integer_offsets=n_int, poles=b["poles"],
               pole_valence=b["pole_valence"],
               round_resid=float(np.abs(n_int - uvres["seam_t"]).max()),
               round_resid_lattice=float(b["dev"].max()),
               seam_int_resid=b["seam_int_resid"],
               min_uv_area=float(a.min()), arap_resid=arap, resid=b["resid"],
               lattice_trials=[{kk: vv_ for kk, vv_ in tr.items()
                                if kk not in ("n", "U", "untangle")}
                               for tr in trials],
               untangle={kk: b["untangle"][kk] for kk in
                         ("flips", "min_area", "eps", "nit", "nfev",
                          "restarts")})
    return out

def round_seams(mesh, uvres, seam_weight=100.0, maxiter=20000):
    """Integer seam offsets, globally consistent + untangled."""
    return solve_integer_uv(mesh, uvres,
                            seam_weight=max(1000.0, float(seam_weight)),
                            maxiter=maxiter)
