"""Parallel Numba kernels for :mod:`remesh`.

Mirror the NumPy/Python reference passes in the sibling modules; imported
lazily (see :mod:`._backend`), so a missing numba install simply falls back
to the original slow-but-portable paths.  Set REMESH_DISABLE_ACCEL=1
to force those paths at runtime.

The kernels keep the reference arithmetic: curvature uses the same weighted
local quadric fit (lstsq) and 2x2 eigen-solve, and separatrix tracing uses
the same ray/triangle intersections and branch choices as
:func:`remesh.cut._ray_exit`.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Curvature: local quadric fit per vertex
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _vertex_curvature(V, VN, tu, tv, starts, nbr, lo, hi, k1, k2, d1):
    """Weighted quadric fit in each vertex tangent frame (prange)."""
    for vi in prange(lo, hi):
        s = starts[vi]
        e = starts[vi + 1]
        m = e - s
        if m < 5:
            continue

        vx = V[vi, 0]
        vy = V[vi, 1]
        vz = V[vi, 2]
        ux = tu[vi, 0]
        uy = tu[vi, 1]
        uz = tu[vi, 2]
        wx = tv[vi, 0]
        wy = tv[vi, 1]
        wz = tv[vi, 2]
        nx = VN[vi, 0]
        ny = VN[vi, 1]
        nz = VN[vi, 2]

        A = np.empty((m, 5))
        b = np.empty(m)
        for k in range(m):
            j = nbr[s + k]
            px = V[j, 0] - vx
            py = V[j, 1] - vy
            pz = V[j, 2] - vz
            x = px * ux + py * uy + pz * uz
            y = px * wx + py * wy + pz * wz
            z = px * nx + py * ny + pz * nz
            w = 1.0 / (x * x + y * y + 1e-30)
            A[k, 0] = x * x * w
            A[k, 1] = x * y * w
            A[k, 2] = y * y * w
            A[k, 3] = x * w
            A[k, 4] = y * w
            b[k] = z * w

        coef = np.linalg.lstsq(A, b, rcond=-1.0)[0]

        H = np.empty((2, 2))
        H[0, 0] = 2.0 * coef[0]
        H[0, 1] = coef[1]
        H[1, 0] = coef[1]
        H[1, 1] = 2.0 * coef[2]
        evals, evecs = np.linalg.eigh(H)

        k1[vi] = evals[0]
        k2[vi] = evals[1]
        pick = 1 if abs(evals[1]) >= abs(evals[0]) else 0
        dx = evecs[0, pick] * ux + evecs[1, pick] * wx
        dy = evecs[0, pick] * uy + evecs[1, pick] * wy
        dz = evecs[0, pick] * uz + evecs[1, pick] * wz
        # Canonical sign: largest-|component| (first on ties) >= 0; matches
        # remesh.curvature._canonical_d1_sign and the rust port.
        adx = abs(dx)
        ady = abs(dy)
        adz = abs(dz)
        if (dx < 0.0 and adx >= ady and adx >= adz) or \
           (dy < 0.0 and ady >= adx and ady >= adz) or \
           (dz < 0.0 and adz >= adx and adz >= ady):
            dx = -dx
            dy = -dy
            dz = -dz
        d1[vi, 0] = dx
        d1[vi, 1] = dy
        d1[vi, 2] = dz


def vertex_curvature(V, VN, tu, tv, starts, nbr, progress=None,
                     chunks=12):
    """Return (k1, k2, d1); progress advances chunk by chunk."""
    n = len(V)
    k1 = np.zeros(n)
    k2 = np.zeros(n)
    d1 = np.zeros((n, 3))
    if n == 0:
        return k1, k2, d1
    chunks = max(1, min(int(chunks), n))
    for c in range(chunks):
        lo = (n * c) // chunks
        hi = (n * (c + 1)) // chunks
        _vertex_curvature(V, VN, tu, tv, starts, nbr, lo, hi, k1, k2, d1)
        if progress is not None:
            progress((c + 1) / chunks)
    return k1, k2, d1


# ---------------------------------------------------------------------------
# Separatrix tracing
# ---------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _ray_exit(V, F, fid, f, px, py, pz, dx, dy, dz, skip_edge):
    """Numba port of cut._ray_exit.

    Returns (kind, id, x, y, z): kind 1 = crossed edge id at the returned
    point, kind 2 = left through vertex id, kind 0 = no exit.
    """
    a = F[f, 0]
    b = F[f, 1]
    c = F[f, 2]
    ox = V[a, 0]
    oy = V[a, 1]
    oz = V[a, 2]

    e1x = V[b, 0] - ox
    e1y = V[b, 1] - oy
    e1z = V[b, 2] - oz
    e2x = V[c, 0] - ox
    e2y = V[c, 1] - oy
    e2z = V[c, 2] - oz

    fnx = e1y * e2z - e1z * e2y
    fny = e1z * e2x - e1x * e2z
    fnz = e1x * e2y - e1y * e2x

    ll = sqrt(e1x * e1x + e1y * e1y + e1z * e1z) + 1e-300
    rx = e1x / ll
    ry = e1y / ll
    rz = e1z / ll

    yx = fny * rz - fnz * ry
    yy = fnz * rx - fnx * rz
    yz = fnx * ry - fny * rx
    ly = sqrt(yx * yx + yy * yy + yz * yz) + 1e-300
    yx /= ly
    yy /= ly
    yz /= ly

    p2x = (px - ox) * rx + (py - oy) * ry + (pz - oz) * rz
    p2y = (px - ox) * yx + (py - oy) * yy + (pz - oz) * yz
    d2x = dx * rx + dy * ry + dz * rz
    d2y = dx * yx + dy * yy + dz * yz

    best_eid = -1
    best_t = 0.0
    best_sx = 0.0
    best_sy = 0.0
    corner_k = -1
    corner_s = 0.0
    corner_t = 0.0

    for k in range(3):
        eid = fid[f, k]
        if eid == skip_edge:
            continue
        av = F[f, k]
        bv = F[f, (k + 1) % 3]
        a2x = ((V[av, 0] - ox) * rx + (V[av, 1] - oy) * ry +
               (V[av, 2] - oz) * rz)
        a2y = ((V[av, 0] - ox) * yx + (V[av, 1] - oy) * yy +
               (V[av, 2] - oz) * yz)
        b2x = ((V[bv, 0] - ox) * rx + (V[bv, 1] - oy) * ry +
               (V[bv, 2] - oz) * rz)
        b2y = ((V[bv, 0] - ox) * yx + (V[bv, 1] - oy) * yy +
               (V[bv, 2] - oz) * yz)
        ex = b2x - a2x
        ey = b2y - a2y
        det = d2x * (-ey) - d2y * (-ex)
        if abs(det) < 1e-300:
            continue
        rhx = a2x - p2x
        rhy = a2y - p2y
        t = (rhx * (-ey) - rhy * (-ex)) / det
        s = (d2x * rhy - d2y * rhx) / det
        if t > 1e-12:
            if s >= -1e-9 and s <= 1.0 + 1e-9:
                if best_eid < 0 or t < best_t:
                    best_eid = eid
                    best_t = t
                    best_sx = a2x + s * ex
                    best_sy = a2y + s * ey
            elif corner_k < 0 or t < corner_t:
                corner_k = k
                corner_s = s
                corner_t = t

    if best_eid >= 0:
        return (1, best_eid,
                ox + best_sx * rx + best_sy * yx,
                oy + best_sx * ry + best_sy * yy,
                oz + best_sx * rz + best_sy * yz)
    if corner_k >= 0:
        kk = (corner_k + 1) % 3 if corner_s > 0.0 else corner_k
        return (2, F[f, kk], 0.0, 0.0, 0.0)
    return (0, -1, 0.0, 0.0, 0.0)


@njit(cache=True)
def _project_dir(nx, ny, nz, dx, dy, dz, prevx, prevy, prevz):
    """In-plane projection of the previous step, snapped to one of 4 dirs.

    Also returns the best dot product (Dp . Dc); the caller tests it
    against zero to detect a reversed ray.
    """
    dot = prevx * nx + prevy * ny + prevz * nz
    dpx = prevx - nx * dot
    dpy = prevy - ny * dot
    dpz = prevz - nz * dot

    perpx = ny * dz - nz * dy
    perpy = nz * dx - nx * dz
    perpz = nx * dy - ny * dx

    bd = dpx * dx + dpy * dy + dpz * dz
    bx = dx
    by = dy
    bz = dz
    v = -(dpx * dx + dpy * dy + dpz * dz)
    if v > bd:
        bd = v
        bx = -dx
        by = -dy
        bz = -dz
    v = dpx * perpx + dpy * perpy + dpz * perpz
    if v > bd:
        bd = v
        bx = perpx
        by = perpy
        bz = perpz
    v = -(dpx * perpx + dpy * perpy + dpz * perpz)
    if v > bd:
        bd = v
        bx = -perpx
        by = -perpy
        bz = -perpz
    return bx, by, bz, bd


@njit(parallel=True, cache=True)
def _trace_kernel(V, F, fid, E, fL, fR, fn, dirs, vf_start, vf_order,
                  ve_start, ve_order, singv, singular, is_cut, lo, hi,
                  max_steps, ray_steps, ray_why, ray_count):
    """Trace up to four field rays from every singularity in [lo, hi).

    Runs independently per singularity: cut edges are accumulated into the
    shared is_cut byte mask (idempotent stores), so threads never wait on
    one another.  ray_* arrays are indexed si * 4 + launched.
    """
    for si in prange(lo, hi):
        sv = singv[si]
        fs = vf_start[sv]
        fe = vf_start[sv + 1]
        ray_count[si] = 0
        if fe <= fs:
            continue

        # ---- choose the outgoing face/frame closest to "away" ----------
        best_score = -2.0
        bf = -1
        bfx = 0.0
        bfy = 0.0
        bfz = 0.0
        bpx = 0.0
        bpy = 0.0
        bpz = 0.0
        for fi in range(fs, fe):
            f0 = vf_order[fi]
            awx = (V[F[f0, 0], 0] + V[F[f0, 1], 0] + V[F[f0, 2], 0]) / 3.0
            awy = (V[F[f0, 0], 1] + V[F[f0, 1], 1] + V[F[f0, 2], 1]) / 3.0
            awz = (V[F[f0, 0], 2] + V[F[f0, 1], 2] + V[F[f0, 2], 2]) / 3.0
            awx -= V[sv, 0]
            awy -= V[sv, 1]
            awz -= V[sv, 2]
            al = sqrt(awx * awx + awy * awy + awz * awz)
            if al < 1e-14:
                continue
            inv = 1.0 / al
            awx *= inv
            awy *= inv
            awz *= inv

            nfx = fn[f0, 0]
            nfy = fn[f0, 1]
            nfz = fn[f0, 2]
            dx = dirs[f0, 0]
            dy = dirs[f0, 1]
            dz = dirs[f0, 2]
            d = dx * nfx + dy * nfy + dz * nfz
            dx -= nfx * d
            dy -= nfy * d
            dz -= nfz * d
            dl = sqrt(dx * dx + dy * dy + dz * dz)
            if dl < 1e-12:
                continue
            inv = 1.0 / dl
            dx *= inv
            dy *= inv
            dz *= inv
            px = nfy * dz - nfz * dy
            py = nfz * dx - nfx * dz
            pz = nfx * dy - nfy * dx

            sc = dx * awx + dy * awy + dz * awz
            if sc > best_score:
                best_score = sc
                bf = f0
                bfx = dx
                bfy = dy
                bfz = dz
                bpx = px
                bpy = py
                bpz = pz
            sc = px * awx + py * awy + pz * awz
            if sc > best_score:
                best_score = sc
                bf = f0
                bfx = px
                bfy = py
                bfz = pz
                bpx = dx
                bpy = dy
                bpz = dz
            sc = -(dx * awx + dy * awy + dz * awz)
            if sc > best_score:
                best_score = sc
                bf = f0
                bfx = -dx
                bfy = -dy
                bfz = -dz
                bpx = -px
                bpy = -py
                bpz = -pz
            sc = -(px * awx + py * awy + pz * awz)
            if sc > best_score:
                best_score = sc
                bf = f0
                bfx = -px
                bfy = -py
                bfz = -pz
                bpx = -dx
                bpy = -dy
                bpz = -dz
        if bf < 0:
            continue

        # ---- candidate launch directions around the vertex --------------
        m = fe - fs
        cap = 4 * m
        cand_f = np.empty(cap, dtype=np.int64)
        cand_q = np.empty(cap, dtype=np.int64)
        cand_x = np.empty(cap)
        cand_y = np.empty(cap)
        cand_z = np.empty(cap)
        cand_s = np.empty(cap)
        nc = 0
        for fi in range(fs, fe):
            f0 = vf_order[fi]
            awx = (V[F[f0, 0], 0] + V[F[f0, 1], 0] + V[F[f0, 2], 0]) / 3.0
            awy = (V[F[f0, 0], 1] + V[F[f0, 1], 1] + V[F[f0, 2], 1]) / 3.0
            awz = (V[F[f0, 0], 2] + V[F[f0, 1], 2] + V[F[f0, 2], 2]) / 3.0
            awx -= V[sv, 0]
            awy -= V[sv, 1]
            awz -= V[sv, 2]
            al = sqrt(awx * awx + awy * awy + awz * awz)
            if al < 1e-14:
                continue
            inv = 1.0 / al
            awx *= inv
            awy *= inv
            awz *= inv

            nfx = fn[f0, 0]
            nfy = fn[f0, 1]
            nfz = fn[f0, 2]
            dx = dirs[f0, 0]
            dy = dirs[f0, 1]
            dz = dirs[f0, 2]
            d = dx * nfx + dy * nfy + dz * nfz
            dx -= nfx * d
            dy -= nfy * d
            dz -= nfz * d
            dl = sqrt(dx * dx + dy * dy + dz * dz)
            if dl < 1e-12:
                continue
            inv = 1.0 / dl
            dx *= inv
            dy *= inv
            dz *= inv
            px = nfy * dz - nfz * dy
            py = nfz * dx - nfx * dz
            pz = nfx * dy - nfy * dx
            aw_dot = dx * awx + dy * awy + dz * awz
            aw_dot_p = px * awx + py * awy + pz * awz
            cand_f[nc] = f0
            cand_q[nc] = 0
            cand_x[nc] = dx
            cand_y[nc] = dy
            cand_z[nc] = dz
            cand_s[nc] = aw_dot
            nc += 1
            cand_f[nc] = f0
            cand_q[nc] = 1
            cand_x[nc] = px
            cand_y[nc] = py
            cand_z[nc] = pz
            cand_s[nc] = aw_dot_p
            nc += 1
            cand_f[nc] = f0
            cand_q[nc] = 2
            cand_x[nc] = -dx
            cand_y[nc] = -dy
            cand_z[nc] = -dz
            cand_s[nc] = -aw_dot
            nc += 1
            cand_f[nc] = f0
            cand_q[nc] = 3
            cand_x[nc] = -px
            cand_y[nc] = -py
            cand_z[nc] = -pz
            cand_s[nc] = -aw_dot_p
            nc += 1

        seen_f = np.empty(4, dtype=np.int64)
        seen_q = np.empty(4, dtype=np.int64)
        nseen = 0

        # ---- launch from each of the four sectors -----------------------
        for sec in range(4):
            if sec == 0:
                sx = bfx
                sy = bfy
                sz = bfz
            elif sec == 1:
                sx = bpx
                sy = bpy
                sz = bpz
            elif sec == 2:
                sx = -bfx
                sy = -bfy
                sz = -bfz
            else:
                sx = -bpx
                sy = -bpy
                sz = -bpz

            pick = -1
            best_dot = -1e300
            for ci in range(nc):
                if cand_s[ci] <= 0.1:
                    continue
                seen = False
                for t in range(nseen):
                    if seen_f[t] == cand_f[ci] and seen_q[t] == cand_q[ci]:
                        seen = True
                        break
                if seen:
                    continue
                dot = (cand_x[ci] * sx + cand_y[ci] * sy +
                       cand_z[ci] * sz)
                if dot <= best_dot:
                    continue
                kind, _, _, _, _ = _ray_exit(
                    V, F, fid, cand_f[ci], V[sv, 0], V[sv, 1], V[sv, 2],
                    cand_x[ci], cand_y[ci], cand_z[ci], -1)
                if kind == 0:
                    continue
                best_dot = dot
                pick = ci
            if pick < 0:
                continue

            seen_f[nseen] = cand_f[pick]
            seen_q[nseen] = cand_q[pick]
            nseen += 1
            launched = ray_count[si]

            f_cur = cand_f[pick]
            dx = cand_x[pick]
            dy = cand_y[pick]
            dz = cand_z[pick]
            lastx = V[sv, 0]
            lasty = V[sv, 1]
            lastz = V[sv, 2]
            prev_eid = -1
            steps = 0
            why = 0
            last_kind = 0
            hit_kind, hit_a, hx, hy, hz = _ray_exit(
                V, F, fid, f_cur, lastx, lasty, lastz, dx, dy, dz, -1)

            while hit_kind != 0 and steps < max_steps:
                steps += 1
                if hit_kind == 2:
                    w = hit_a
                    curx = V[w, 0]
                    cury = V[w, 1]
                    curz = V[w, 2]
                    if singular[w] and steps > 1:
                        why = 1          # hit another singularity
                        break
                    merged = False
                    if steps > 1:
                        for t in range(ve_start[w], ve_start[w + 1]):
                            ee = ve_order[t]
                            if ee != prev_eid and is_cut[ee] != 0:
                                why = 8      # merge with existing cut
                                merged = True
                                break
                    if merged:
                        break
                    e2 = -1
                    for kk in range(3):
                        ec = fid[f_cur, kk]
                        if ec != prev_eid and (E[ec, 0] == w or
                                               E[ec, 1] == w):
                            e2 = ec
                            break
                    if e2 < 0 or fL[e2] < 0 or fR[e2] < 0:
                        why = 2          # boundary / no continuation
                        break
                    is_cut[e2] = 1
                    f2 = fR[e2] if fL[e2] == f_cur else fL[e2]
                    ddx, ddy, ddz, _bdd = _project_dir(
                        fn[f2, 0], fn[f2, 1], fn[f2, 2],
                        dirs[f2, 0], dirs[f2, 1], dirs[f2, 2],
                        curx - lastx, cury - lasty, curz - lastz)
                    f_cur = f2
                    prev_eid = e2
                    lastx = curx
                    lasty = cury
                    lastz = curz
                    last_kind = 2
                    mx = 0.5 * (V[E[e2, 0], 0] + V[E[e2, 1], 0])
                    my = 0.5 * (V[E[e2, 0], 1] + V[E[e2, 1], 1])
                    mz = 0.5 * (V[E[e2, 0], 2] + V[E[e2, 1], 2])
                    hit_kind, hit_a, hx, hy, hz = _ray_exit(
                        V, F, fid, f_cur, mx, my, mz,
                        ddx, ddy, ddz, e2)
                    continue

                # ---- regular edge crossing ------------------------------
                eid = hit_a
                curx = hx
                cury = hy
                curz = hz
                if is_cut[eid] != 0:
                    why = 8              # merge with existing cut
                    break
                va = E[eid, 0]
                vb = E[eid, 1]
                if steps > 1:
                    elen = sqrt(
                        (V[va, 0] - V[vb, 0]) * (V[va, 0] - V[vb, 0]) +
                        (V[va, 1] - V[vb, 1]) * (V[va, 1] - V[vb, 1]) +
                        (V[va, 2] - V[vb, 2]) * (V[va, 2] - V[vb, 2]))
                    snapped = False
                    for ve in (va, vb):
                        if singular[ve]:
                            dd = sqrt(
                                (curx - V[ve, 0]) * (curx - V[ve, 0]) +
                                (cury - V[ve, 1]) * (cury - V[ve, 1]) +
                                (curz - V[ve, 2]) * (curz - V[ve, 2]))
                            if dd < 0.5 * elen:
                                curx = V[ve, 0]
                                cury = V[ve, 1]
                                curz = V[ve, 2]
                                is_cut[eid] = 1
                                why = 1
                                snapped = True
                                break
                    if snapped:
                        break
                if fL[eid] < 0 or fR[eid] < 0:
                    why = 2
                    break
                is_cut[eid] = 1
                fa = fL[eid]
                f_next = fR[eid] if fa == f_cur else fa
                ddx, ddy, ddz, bdd = _project_dir(
                    fn[f_next, 0], fn[f_next, 1], fn[f_next, 2],
                    dirs[f_next, 0], dirs[f_next, 1], dirs[f_next, 2],
                    curx - lastx, cury - lasty, curz - lastz)
                if bdd <= 0.0:
                    why = 3              # reversed
                    break
                lastx = curx
                lasty = cury
                lastz = curz
                f_cur = f_next
                prev_eid = eid
                last_kind = 1
                mx = 0.5 * (V[va, 0] + V[vb, 0])
                my = 0.5 * (V[va, 1] + V[vb, 1])
                mz = 0.5 * (V[va, 2] + V[vb, 2])
                hit_kind, hit_a, hx, hy, hz = _ray_exit(
                    V, F, fid, f_cur, mx, my, mz, ddx, ddy, ddz, eid)

            if steps >= max_steps:
                why = 4
            elif hit_kind == 0:
                why = 5 + last_kind      # noexit_launch/edge/corner

            idx = si * 4 + launched
            ray_steps[idx] = steps
            ray_why[idx] = why
            ray_count[si] = launched + 1


def trace_separatrices(V, F, fid, E, fL, fR, fn, dirs, vf_start, vf_order,
                       ve_start, ve_order, singv, max_steps,
                       progress=None, chunks=16):
    """Parallel separatrix trace; returns cut mask and per-ray diagnostics."""
    n_sing = len(singv)
    n_edges = len(E)
    is_cut = np.zeros(n_edges, dtype=np.uint8)
    ray_steps = np.zeros(max(1, 4 * n_sing), dtype=np.int64)
    ray_why = np.zeros(max(1, 4 * n_sing), dtype=np.int8)
    ray_count = np.zeros(max(1, n_sing), dtype=np.int64)
    if n_sing == 0:
        return is_cut.astype(bool), ray_steps[:0], ray_why[:0], ray_count[:0]
    singular = np.zeros(len(V), dtype=np.uint8)
    singular[singv] = 1
    chunks = max(1, min(int(chunks), n_sing))
    for c in range(chunks):
        lo = (n_sing * c) // chunks
        hi = (n_sing * (c + 1)) // chunks
        _trace_kernel(V, F, fid, E, fL, fR, fn, dirs, vf_start, vf_order,
                      ve_start, ve_order, singv, singular, is_cut, lo, hi,
                      max_steps, ray_steps, ray_why, ray_count)
        if progress is not None:
            progress((c + 1) / chunks)
    return (is_cut.astype(bool), ray_steps[:4 * n_sing],
            ray_why[:4 * n_sing], ray_count[:n_sing])


# ---------------------------------------------------------------------------
# Integer seam offsets: parallel-free sparse coordinate descent
# ---------------------------------------------------------------------------
@njit(cache=True)
def integer_sweeps(indptr, indices, data, diag, g, X, js):
    changed = False
    for jj in range(len(js)):
        j = js[jj]
        gr = g[j].real
        gi = g[j].imag
        dj = diag[j]
        best_re = 0.0
        best_im = 0.0
        bd = 0.0
        d = 2.0 * gr + dj
        if d < bd - 1e-12:
            bd = d
            best_re = 1.0
        d = -2.0 * gr + dj
        if d < bd - 1e-12:
            bd = d
            best_re = -1.0
            best_im = 0.0
        d = 2.0 * gi + dj
        if d < bd - 1e-12:
            bd = d
            best_re = 0.0
            best_im = 1.0
        d = -2.0 * gi + dj
        if d < bd - 1e-12:
            bd = d
            best_re = 0.0
            best_im = -1.0
        if best_re != 0.0 or best_im != 0.0:
            step = best_re + 1j * best_im
            X[j] = X[j] + step
            lo = indptr[j]
            hi = indptr[j + 1]
            for k in range(lo, hi):
                g[indices[k]] = g[indices[k]] + data[k] * step
            changed = True
    return changed


# ---------------------------------------------------------------------------
# UV untangle objective: area deficit and its exact gradient
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def untangle_eval(x, ub_re, ub_im, rot_re, rot_im, vidx, vcol, corner, eps,
                  zc_re, zc_im, uc_re, uc_im, gc_re, gc_im,
                  gz_re, gz_im, g):
    n_uv = ub_re.shape[0]
    nv = gz_re.shape[0]
    nf = corner.shape[0]

    for i in range(n_uv):
        zc_re[i] = 0.0
        zc_im[i] = 0.0
        gc_re[i] = 0.0
        gc_im[i] = 0.0
    for k in range(nv):
        gz_re[k] = 0.0
        gz_im[k] = 0.0

    for t in range(vidx.shape[0]):
        i = vidx[t]
        j = vcol[t]
        zc_re[i] = x[2 * j]
        zc_im[i] = x[2 * j + 1]

    for i in range(n_uv):
        ur = ub_re[i]
        ui = ub_im[i]
        rr = rot_re[i]
        ri = rot_im[i]
        zr = zc_re[i]
        zi = zc_im[i]
        uc_re[i] = ur + rr * zr - ri * zi
        uc_im[i] = ui + rr * zi + ri * zr

    energy = 0.0
    for f in range(nf):
        a = corner[f, 0]
        b = corner[f, 1]
        c = corner[f, 2]
        ax = uc_re[b] - uc_re[a]
        ay = uc_im[b] - uc_im[a]
        bx = uc_re[c] - uc_re[a]
        by = uc_im[c] - uc_im[a]
        area = ax * by - ay * bx
        if area < eps:
            d = eps - area
            energy += d * d
            x0 = uc_re[a]
            y0 = uc_im[a]
            x1 = uc_re[b]
            y1 = uc_im[b]
            x2 = uc_re[c]
            y2 = uc_im[c]
            scale = -2.0 * d
            gc_re[a] += (y1 - y2) * scale
            gc_im[a] += (x2 - x1) * scale
            gc_re[b] += (y2 - y0) * scale
            gc_im[b] += (x0 - x2) * scale
            gc_re[c] += (y0 - y1) * scale
            gc_im[c] += (x1 - x0) * scale

    for t in range(vidx.shape[0]):
        i = vidx[t]
        j = vcol[t]
        rr = rot_re[i]
        ri = rot_im[i]
        gr = gc_re[i]
        gi = gc_im[i]
        gz_re[j] += gr * rr + gi * ri
        gz_im[j] += gi * rr - gr * ri

    for j in range(nv):
        g[2 * j] = gz_re[j]
        g[2 * j + 1] = gz_im[j]
    return energy
