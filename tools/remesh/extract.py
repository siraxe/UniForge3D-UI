"""Integer-grid quad extraction."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations

import numpy as np

from .mesh import TriMesh


def _bary(tri, p):
    a = tri[0]
    d = tri[1] - a
    e = tri[2] - a
    v = p - a
    den = d.real * e.imag - d.imag * e.real
    if abs(den) < 1e-300:
        return np.array([-1.0, -1.0, -1.0])
    l1 = (v.real * e.imag - v.imag * e.real) / den
    l2 = (d.real * v.imag - d.imag * v.real) / den
    return np.array([1.0 - l1 - l2, l1, l2])


def _pt(lam, V, F, f):
    return (lam[0] * V[F[f, 0]] + lam[1] * V[F[f, 1]] +
            lam[2] * V[F[f, 2]])


def _locate(Uc, fid, fL, fR, seam, f0, target, max_steps=256, tol=1e-9,
            anchor=None, V=None, F=None, max_dist=None):
    """Walk the straight segment across faces/seams to `target`."""
    f = int(f0)
    for _ in range(max_steps):
        lam = _bary(Uc[f], target)
        i = int(np.argmin(lam))
        if lam[i] >= -tol:
            lam = np.maximum(lam, 0.0)
            s = lam.sum()
            if s <= 0:
                return None
            lam = lam / s
            if anchor is not None:
                if np.linalg.norm(_pt(lam, V, F, f) - anchor) > max_dist:
                    return None
            return f, lam
        e = int(fid[f, (i + 1) % 3])
        if e in seam:
            R, n = seam[e]
            if int(fL[e]) == f:
                target = R * target + n
                g = int(fR[e])
            else:
                target = np.conj(R) * (target - n)
                g = int(fL[e])
        else:
            g = int(fR[e]) if int(fL[e]) == f else int(fL[e])
        if g < 0:
            return None
        f = g
    return None


def _seam_transitions(uvres):
    pairs = uvres["cut_pairs"]
    ks = uvres["seam_k"]
    ns = uvres["integer_offsets"]
    return {int(p[0]): (1j ** int(ks[i]), complex(ns[i]))
            for i, p in enumerate(pairs)}


def _hybrid_locator(Uc, patch, n_patches, fL, fR, fid, seam, tol=1e-9,
                    V=None, F=None, max_dist=None):
    """In-patch exhaustive query, else straight walk across seams.

    With an anchor, pick the candidate preimage nearest that 3D point
    (the UV layout overlaps globally, so a plain containment scan can
    alias a corner onto a far-away patch)."""
    faces_of = [np.flatnonzero(patch == p) for p in range(n_patches)]

    def qpatch(p, pt, anchor, near=False):
        best = None
        near_best = None
        for f in faces_of[p]:
            f = int(f)
            lam = _bary(Uc[f], pt)
            m = float(lam.min())
            if m < -tol:
                if near:
                    lam_n = np.maximum(lam, 0.0)
                    s = lam_n.sum()
                    if s <= 0:
                        continue
                    lam_n = lam_n / s
                    if anchor is not None:
                        dd = float(np.linalg.norm(
                            _pt(lam_n, V, F, f) - anchor))
                        sc_n = (-dd, m)
                    else:
                        sc_n = (0.0, m)
                    if near_best is None or sc_n > near_best[0]:
                        near_best = (sc_n, f, lam_n)
                continue
            lam = np.maximum(lam, 0.0)
            s = lam.sum()
            if s <= 0:
                continue
            lam = lam / s
            if anchor is not None:
                dd = float(np.linalg.norm(_pt(lam, V, F, f) - anchor))
                sc = (-dd, m)
            else:
                sc = (0.0, m)
            if best is None or sc > best[0]:
                best = (sc, f, lam)
        if best is None and near:
            best = near_best
        return best

    def locate(f0, target, anchor=None):
        r = _locate(Uc, fid, fL, fR, seam, f0, target,
                    anchor=anchor, V=V, F=F, max_dist=max_dist)
        if r is not None:
            return r
        hit = qpatch(int(patch[f0]), target, anchor)
        if hit is not None:
            _, g, lam = hit
            return g, lam
        return None

    return locate


def _patch_locator(Uc, patch, n_patches, pairs, ks, ns, fL, fR,
                   max_depth=6, tol=1e-9):
    """Per-patch exhaustive point query + BFS across seams."""

    faces_of = [np.flatnonzero(patch == p) for p in range(n_patches)]

    def qpatch(p, pt):
        best = None
        for f in faces_of[p]:
            f = int(f)
            lam = _bary(Uc[f], pt)
            m = float(lam.min())
            if m >= -tol and (best is None or m > best[0]):
                best = (m, f, lam)
        return best

    pcs = defaultdict(list)
    for i, p in enumerate(pairs):
        e = int(p[0])
        R = 1j ** int(ks[i])
        n = complex(ns[i])
        pl = int(patch[fL[e]])
        pr = int(patch[fR[e]])
        pcs[pl].append((pr, int(fR[e]), R, n))
        pcs[pr].append((pl, int(fL[e]), np.conj(R), -np.conj(R) * n))

    def locate(f0, target):
        q = deque([(int(f0), complex(target), 0)])
        seen = set()
        while q:
            f, pt, d = q.popleft()
            p = int(patch[f])
            key = (p, round(pt.real, 6), round(pt.imag, 6))
            if key in seen:
                continue
            seen.add(key)
            hit = qpatch(p, pt)
            if hit is not None:
                _, g, lam = hit
                lam = np.maximum(lam, 0.0)
                s = lam.sum()
                if s <= 0:
                    return None
                return g, lam / s
            if d >= max_depth:
                continue
            for pp, gf, R, n in pcs[p]:
                q.append((gf, R * pt + n, d + 1))
        return None

    return locate


def _fill_boundary(verts, quads, tris, log=None):
    """Cap closed boundary loops; reverse directed cycle."""
    directed = set()
    for f in list(quads) + list(tris):
        for k in range(len(f)):
            directed.add((int(f[k]), int(f[(k + 1) % len(f)])))
    bnd = {}
    for a, b in directed:
        if (b, a) not in directed:
            bnd.setdefault(a, []).append(b)
    for a in bnd:
        bnd[a].sort()
    covered = set()
    n_fill = 0
    for a0 in sorted(bnd):
        for b0 in bnd[a0]:
            if (a0, b0) in covered:
                continue
            loop = [a0, b0]
            cur = b0
            ok = False
            while len(loop) <= len(bnd) + 2:
                nxts = [x for x in bnd.get(cur, ())
                        if (cur, x) not in covered and x not in loop[1:]]
                if not nxts:
                    break
                nxt = nxts[0]
                if nxt == a0:
                    ok = True
                    break
                loop.append(nxt)
                cur = nxt
            if not ok:
                continue
            for k in range(len(loop)):
                covered.add((loop[k], loop[(k + 1) % len(loop)]))
            cyc = loop[::-1]
            if log is not None:
                log.append({"kind": "fill", "loop": list(cyc)})
            if len(cyc) == 3:
                tris.append(cyc)
            elif len(cyc) == 4:
                quads.append(cyc)
            else:
                c = len(verts)
                verts.append(np.mean([verts[i] for i in cyc], axis=0))
                for k in range(len(cyc)):
                    tris.append([cyc[k], cyc[(k + 1) % len(cyc)], c])
            n_fill += 1
    return n_fill


def _merge_pinches(verts, faces, max_iter=64):
    """Fuse face pairs sharing 2 edges (non-manifold pinch)."""
    n_merge = 0
    for _ in range(max_iter):
        emap = {}
        for fi, f in enumerate(faces):
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                emap.setdefault((min(a, b), max(a, b)), []).append(fi)
        pair = {}
        for e, fs in emap.items():
            if len(fs) == 2:
                pair.setdefault(tuple(sorted(fs)), set()).add(e)
        hit = None
        for (f1, f2), sh in pair.items():
            if len(sh) >= 2:
                hit = (f1, f2, sh)
                break
        if hit is None:
            break
        f1, f2, sh = hit
        segs = []
        for fi in (f1, f2):
            f = faces[fi]
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                if (min(a, b), max(a, b)) not in sh:
                    segs.append((a, b))
        new = [f for i, f in enumerate(faces) if i not in (f1, f2)]
        out = {}
        clash = False
        for a, b in segs:
            if a in out and out[a] != b:
                clash = True
                break
            out[a] = b
        if clash:
            break
        ends = set(out.values())
        starts = [a for a in out if a not in ends] or (list(out)[:1])
        used = set()
        for s in starts:
            if s in used:
                continue
            cyc = [s]
            used.add(s)
            cur = s
            while True:
                nxt = out.get(cur)
                if nxt is None:
                    cyc = []
                    break
                if nxt == s:
                    break
                if nxt in used:
                    cyc = []
                    break
                used.add(nxt)
                cyc.append(nxt)
                cur = nxt
            if len(cyc) < 3:
                continue
            if len(cyc) <= 4:
                new.append(cyc)
            else:
                c = len(verts)
                verts.append(np.mean([verts[i] for i in cyc], axis=0))
                for k in range(len(cyc)):
                    new.append([cyc[k], cyc[(k + 1) % len(cyc)], c])
        faces[:] = new
        n_merge += 1
    return n_merge


def _face_area(verts, f):
    p = [verts[int(i)] for i in f]
    if len(p) == 3:
        return 0.5 * float(np.linalg.norm(
            np.cross(p[1] - p[0], p[2] - p[0])))
    n = np.zeros(3)
    for k in range(len(p)):
        n += np.cross(p[k], p[(k + 1) % len(p)])
    return 0.5 * float(np.linalg.norm(n))


def _repair_duplicate_edges(verts, faces, max_iter=32, log=None):
    """Drop largest same-direction duplicate faces, then re-cap."""
    n_rep = 0
    for _ in range(max_iter):
        d = {}
        for fi, f in enumerate(faces):
            for k in range(len(f)):
                a, b = int(f[k]), int(f[(k + 1) % len(f)])
                d.setdefault((a, b), []).append(fi)
        bad = [fs for fs in d.values() if len(fs) > 1]
        if not bad:
            break
        cand = sorted(set(fi for fs in bad for fi in fs))
        fi = max(cand, key=lambda j: _face_area(verts, faces[j]))
        if log is not None:
            log.append({"kind": "repair", "dup": list(bad[0]),
                        "deleted": list(faces[fi]),
                        "pos": list(np.mean([verts[x] for x in faces[fi]],
                                            axis=0))})
        faces.pop(fi)
        quads = [f for f in faces if len(f) == 4]
        tris = [f for f in faces if len(f) == 3]
        _fill_boundary(verts, quads, tris, log=log)
        faces[:] = quads + tris
        n_rep += 1
    return n_rep


def _merge_triangle_pairs(verts, faces):
    """Merge adjacent triangle pairs into quads."""
    tidx = [i for i, f in enumerate(faces) if len(f) == 3]
    if not tidx:
        return 0
    V = np.asarray(verts, dtype=np.float64)
    em = defaultdict(list)
    for i in tidx:
        t = [int(x) for x in faces[i]]
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            em[(min(a, b), max(a, b))].append(i)
    qset = set(tuple(sorted(int(x) for x in f))
               for f in faces if len(f) == 4)
    paired = set()
    merged = []
    for i in tidx:
        if i in paired:
            continue
        t = [int(x) for x in faces[i]]
        found = None
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            for j in em[(min(a, b), max(a, b))]:
                if j == i or j in paired:
                    continue
                w = [int(x) for x in faces[j]]
                if not (((a, b) == (w[1], w[0])) or
                        ((a, b) == (w[2], w[1])) or
                        ((a, b) == (w[0], w[2]))):
                    continue
                c = [x for x in t if x not in (a, b)][0]
                d = [x for x in w if x not in (a, b)][0]
                quad = [b, c, a, d]
                if len(set(quad)) < 4:
                    continue
                if tuple(sorted(quad)) in qset:
                    continue
                p = V[np.array(quad)]
                nq = np.zeros(3)
                for m in range(4):
                    nq += np.cross(p[m], p[(m + 1) % 4])
                if float(np.linalg.norm(nq)) < 1e-14:
                    continue
                nt = np.cross(V[t[1]] - V[t[0]], V[t[2]] - V[t[0]])
                nw = np.cross(V[w[1]] - V[w[0]], V[w[2]] - V[w[0]])
                if float(nq @ (nt + nw)) <= 0:
                    continue
                found = (j, quad)
                break
            if found:
                break
        if found is None:
            continue
        j, quad = found
        paired.add(i)
        paired.add(j)
        merged.append(quad)
        qset.add(tuple(sorted(quad)))
    if not merged:
        return 0
    keep = [list(f) for i, f in enumerate(faces)
            if len(f) == 4 or i not in paired]
    faces[:] = keep + merged
    return len(merged)


def _tri_project_batch(p, a, b, c):
    n = np.cross(b - a, c - a)
    nn = np.einsum("ij,ij->i", n, n)
    ok = nn >= 1e-24
    nns = np.where(ok, nn, 1.0)
    ap = p - a
    q = p - n * (np.einsum("ij,ij->i", ap, n) / nns)[:, None]
    v0 = b - a
    v1 = c - a
    v2 = q - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    den = d00 * d11 - d01 * d01
    ok &= np.abs(den) >= 1e-30
    dens = np.where(ok, den, 1.0)
    v = (d11 * d20 - d01 * d21) / dens
    w = (d00 * d21 - d01 * d20) / dens
    bary = np.minimum(1.0 - v - w, np.minimum(v, w))
    dist = np.linalg.norm(q - p, axis=1)
    return q, bary, dist, ok


def _quad_score(pts, nref):
    score = 1.0
    turn = 0.0
    for k in range(4):
        a, b, c = pts[k - 1], pts[k], pts[(k + 1) % 4]
        e0, e1 = a - b, c - b
        l0 = float(np.linalg.norm(e0))
        l1 = float(np.linalg.norm(e1))
        if l0 < 1e-30 or l1 < 1e-30:
            return -1.0
        sn = -float(np.cross(e0, e1) @ nref) / (l0 * l1)
        cs = -float(e0 @ e1) / (l0 * l1)
        score = min(score, sn)
        turn += np.arctan2(sn, cs)
    if turn <= np.pi:
        return -1.0
    return score


def _kernel_point(pts2, p2):
    R = 4.0 * max(max(abs(c) for c in p) for p in pts2) + 1.0
    poly = [(-R, -R), (R, -R), (R, R), (-R, R)]
    n = len(pts2)
    for i in range(n):
        ax, ay = pts2[i]
        bx, by = pts2[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        out = []
        m = len(poly)
        for j in range(m):
            c = poly[j]
            d = poly[(j + 1) % m]
            sc = ex * (c[1] - ay) - ey * (c[0] - ax)
            sd = ex * (d[1] - ay) - ey * (d[0] - ax)
            if sc >= 0:
                out.append(c)
            if (sc > 0) != (sd > 0):
                t = sc / (sc - sd)
                out.append((c[0] + t * (d[0] - c[0]),
                            c[1] + t * (d[1] - c[1])))
        poly = out
        if not poly:
            return None
    px, py = p2
    m = len(poly)
    inside = all(
        (poly[(i + 1) % m][0] - poly[i][0]) * (py - poly[i][1]) -
        (poly[(i + 1) % m][1] - poly[i][1]) * (px - poly[i][0]) >= 0
        for i in range(m))
    if inside:
        return p2
    best = None
    for j in range(m):
        ax, ay = poly[j]
        bx, by = poly[(j + 1) % m]
        ex, ey = bx - ax, by - ay
        ll = ex * ex + ey * ey
        t = 0.0 if ll <= 0 else max(0.0, min(1.0, ((px - ax) * ex +
                                                   (py - ay) * ey) / ll))
        q = (ax + t * ex, ay + t * ey)
        d = (q[0] - px) ** 2 + (q[1] - py) ** 2
        if best is None or d < best[0]:
            best = (d, q)
    return best[1]


def _fan_case(verts, faces, rem, val, ppos):
    cnt = {}
    for fi in rem:
        f = faces[fi]
        for k in range(len(f)):
            a, b = int(f[k]), int(f[(k + 1) % len(f)])
            cnt[(a, b)] = cnt.get((a, b), 0) + 1
    if len(cnt) != sum(len(faces[fi]) for fi in rem):
        return None
    if any(c != 1 for c in cnt.values()):
        return None
    bnd = [(a, b) for (a, b), c in cnt.items() if cnt.get((b, a), 0) == 0]
    if len(bnd) != 2 * val:
        return None
    nxt = {}
    for a, b in bnd:
        if a in nxt:
            return None
        nxt[a] = b
    start = bnd[0][0]
    cyc = [start]
    while True:
        b = nxt.get(cyc[-1])
        if b is None:
            return None
        if b == start:
            break
        cyc.append(b)
        if len(cyc) > len(bnd):
            return None
    if len(cyc) != len(bnd):
        return None
    nref = np.zeros(3)
    for fi in rem:
        pts = [verts[i] for i in faces[fi]]
        for k in range(len(pts)):
            nref += np.cross(pts[k], pts[(k + 1) % len(pts)])
    nn = float(np.linalg.norm(nref))
    if nn < 1e-24:
        return None
    nref /= nn
    Q = np.array([verts[i] for i in cyc])
    org = Q.mean(axis=0)
    u = Q[0] - org
    lu = float(np.linalg.norm(u))
    if lu < 1e-30:
        return None
    u /= lu
    w = np.cross(nref, u)
    pts2 = [((q - org) @ u, (q - org) @ w) for q in Q]
    area = sum(pts2[i][0] * pts2[(i + 1) % len(pts2)][1] -
               pts2[(i + 1) % len(pts2)][0] * pts2[i][1]
               for i in range(len(pts2)))
    if area <= 0:
        return None
    q2 = _kernel_point(pts2, (float((ppos - org) @ u),
                              float((ppos - org) @ w)))
    if q2 is None:
        return None
    hp = float((ppos - org) @ nref)
    pnew = org + q2[0] * u + q2[1] * w + hp * nref
    score = 1.0
    for k in range(val):
        pts = [pnew, verts[cyc[(2 * k) % len(cyc)]],
               verts[cyc[2 * k + 1]], verts[cyc[(2 * k + 2) % len(cyc)]]]
        score = min(score, _quad_score(pts, nref))
    return cyc, score, pnew


def _insert_grid_poles(verts, faces, points, valences, h=0.0):
    """Local GridPole surgery: replace quads by a fan."""
    if h <= 0:
        P = np.asarray(verts, dtype=float)
        h = float(np.max(P.max(0) - P.min(0))) * 1e-2
    used = {int(x) for f in faces for x in f}
    n_ins = 0
    for ppos, val in zip(points, valences):
        val = int(val)
        if val not in (3, 5):
            continue
        ppos = np.asarray(ppos, dtype=float)
        # Snapshot vertex positions (new pole vertices are appended below).
        Varr = np.asarray(verts, dtype=float)
        used_arr = np.fromiter(used, dtype=np.int64)
        dmin = (float(np.linalg.norm(Varr[used_arr] - ppos, axis=1).min())
                if len(used_arr) else np.inf)
        if dmin <= 0.05 * h:
            continue
        qidx = [fi for fi, f in enumerate(faces) if len(f) == 4]
        hit = None
        if qidx:
            Q = np.asarray([faces[fi] for fi in qidx], dtype=np.int64)
            best_d = np.inf
            best_qi = -1
            best_q = None
            for ia, ib, ic in ((0, 1, 2), (0, 2, 3)):
                q, bary, dist, ok = _tri_project_batch(
                    ppos, Varr[Q[:, ia]], Varr[Q[:, ib]], Varr[Q[:, ic]])
                ok &= bary >= -1e-6
                if not ok.any():
                    continue
                dist = np.where(ok, dist, np.inf)
                idx = int(np.argmin(dist))
                if dist[idx] < best_d:
                    best_d = float(dist[idx])
                    best_qi = idx
                    best_q = q[idx]
            if best_qi >= 0:
                hit = (best_d, qidx[best_qi], best_q)
        if hit is None:
            continue
        em = {}
        vadj = {}
        for fi, f in enumerate(faces):
            for k in range(len(f)):
                a, b = int(f[k]), int(f[(k + 1) % len(f)])
                e = (min(a, b), max(a, b))
                em.setdefault(e, []).append(fi)
                vadj.setdefault(a, set()).add(e)
                vadj.setdefault(b, set()).add(e)
        qidx = hit[1]
        fq = faces[qidx]
        nbrs = []
        ok = True
        for k in range(4):
            a, b = int(fq[k]), int(fq[(k + 1) % 4])
            fs = em.get((min(a, b), max(a, b)), [])
            nb = [g for g in fs if g != qidx]
            if len(nb) != 1:
                ok = False
                break
            nbrs.append(nb[0])
        if not ok:
            continue
        cand = set()
        for g in [qidx] + nbrs:
            f = faces[g]
            if len(f) != 4:
                continue
            for k in range(4):
                a, b = int(f[k]), int(f[(k + 1) % 4])
                cand.update(em.get((min(a, b), max(a, b)), []))
        cand.discard(qidx)
        best = None
        for sub in combinations(sorted(cand), val - 2):
            rem = [qidx] + list(sub)
            if len(set(rem)) != len(rem):
                continue
            r = _fan_case(verts, faces, rem, val, hit[2])
            if r is None or r[1] <= -0.999:
                continue
            remset = set(rem)
            dele = set()
            for fi in rem:
                f = faces[fi]
                for k in range(len(f)):
                    a, b = int(f[k]), int(f[(k + 1) % len(f)])
                    e = (min(a, b), max(a, b))
                    if all(g in remset for g in em[e]):
                        dele.add(e)
            gain = {}
            for k in range(val):
                gain[r[0][2 * k]] = gain.get(r[0][2 * k], 0) + 1
            aff = set(gain)
            for fi in rem:
                aff.update(int(x) for x in faces[fi])
            dirr = 1
            nv2 = 0
            nv6 = 0
            for x in aff:
                old = len(vadj.get(x, ()))
                new = len(vadj.get(x, set()) - dele) + gain.get(x, 0)
                if old != 4:
                    dirr -= 1
                if new != 4:
                    dirr += 1
                if new < 3:
                    nv2 += 1
                if new > 5:
                    nv6 += 1
            d = float(np.linalg.norm(r[2] - ppos))
            key = (-(dirr + 10 * (nv2 + nv6)), r[1] - 0.05 * d / h)
            if best is None or key > best[0]:
                best = (key, rem, r)
        if best is None:
            continue
        _, rem, r = best
        cyc, score, pnew = r
        Pidx = len(verts)
        verts.append(np.asarray(pnew, dtype=float))
        used.add(Pidx)
        for fi in sorted(rem, reverse=True):
            faces.pop(fi)
        for k in range(val):
            faces.append([Pidx, cyc[2 * k], cyc[2 * k + 1],
                          cyc[(2 * k + 2) % len(cyc)]])
        n_ins += 1
    return n_ins


def extract_quads(mesh: TriMesh, uvres, dedup_tol=None, locate=None,
                  fill_holes=True, seed_poles=False, insert_poles=True,
                  debug_log=None, progress=None):
    """Extract a quad mesh from an integer-seam UV layout.

    ``progress`` is called with a fraction in [0, 1] as the grid-cell scan
    advances."""
    corner = uvres["corner_uv"]
    U = uvres["u_complex"]
    Uc = U[corner]
    fL, fR = mesh.edge_faces()
    V = mesh.V
    F = mesh.F
    nF = mesh.n_faces
    h = float(uvres.get("h", 0.0) or 0.0)
    if locate is None:
        patch = np.asarray(uvres["patch_of_face"], dtype=np.int64)
        max_dist = 4.0 * h if h > 0 else None
        locate = _hybrid_locator(Uc, patch, int(uvres["n_patches"]),
                                 fL, fR, mesh.face_edge_ids,
                                 _seam_transitions(uvres),
                                 V=V, F=F, max_dist=max_dist)

    if dedup_tol is None:
        ext = float(np.max(V.max(axis=0) - V.min(axis=0)))
        dedup_tol = max(ext, 1e-12) * 1e-7

    verts = []
    vmap = {}
    quads = []
    tris = []
    seen_q = set()
    seen_t = set()
    n_fail = 0
    n_cells = 0
    n_dup = 0
    n_rep = 0
    n_reject = 0

    def key_of(p):
        return (int(round(p[0] / dedup_tol)),
                int(round(p[1] / dedup_tol)),
                int(round(p[2] / dedup_tol)))

    def add_vertex(f, lam):
        p = (lam[0] * V[F[f, 0]] + lam[1] * V[F[f, 1]] +
             lam[2] * V[F[f, 2]])
        k = key_of(p)
        j = vmap.get(k)
        if j is None:
            j = len(verts)
            vmap[k] = j
            verts.append(p)
        return j

    def emit(f, i, j, anchor, pole=None):
        nonlocal n_fail, n_dup, n_rep, n_reject
        cs = []
        locs = []
        for c in (complex(i, j), complex(i + 1, j),
                  complex(i + 1, j + 1), complex(i, j + 1)):
            try:
                loc = locate(f, c, anchor=anchor)
            except TypeError:
                loc = locate(f, c)
            if loc is None:
                n_fail += 1
                if debug_log is not None:
                    debug_log.append({"kind": "fail", "f": f,
                                      "ij": (i, j)})
                return
            cs.append(add_vertex(*loc))
            locs.append(loc)
        if pole is not None:
            ps = [verts[j] for j in cs]
            emax = max(float(np.linalg.norm(ps[(k + 1) % 4] - ps[k]))
                       for k in range(4))
            if emax > pole[1]:
                n_reject += 1
                return
        nuniq = len(set(cs))
        if nuniq == 3:
            t = [cs[k] for k in range(4) if cs[k] != cs[(k + 1) % 4]]
            if len(t) != 3:
                n_dup += 1
                return
            tk = tuple(sorted(t))
            if tk in seen_t or tk in seen_q:
                n_rep += 1
                return
            seen_t.add(tk)
            tris.append(t)
            return
        if nuniq < 4:
            n_dup += 1
            return
        qk = tuple(sorted(cs))
        if qk in seen_q or qk in seen_t:
            n_rep += 1
            return
        seen_q.add(qk)
        quads.append(cs)
        if debug_log is not None:
            debug_log.append({"f": f, "ij": (i, j), "cs": tuple(cs),
                              "loc": [(int(l[0]), tuple(float(x) for x in l[1])) for l in locs]})

    for f in range(nF):
        if progress is not None and f % 256 == 0:
            progress(f / max(1, nF))
        tri = Uc[f]
        lo_i = int(np.floor(tri.real.min()))
        hi_i = int(np.ceil(tri.real.max()))
        lo_j = int(np.floor(tri.imag.min()))
        hi_j = int(np.ceil(tri.imag.max()))
        for i in range(lo_i, hi_i):
            for j in range(lo_j, hi_j):
                lam = _bary(tri, complex(i + 0.5, j + 0.5))
                if lam.min() < -1e-9:
                    continue
                n_cells += 1
                lamc = np.maximum(lam, 0.0)
                lamc /= lamc.sum()
                emit(f, i, j, _pt(lamc, V, F, f))

    patches = uvres.get("patch_of_face")
    poles = uvres.get("poles")
    if seed_poles and poles is not None and patches is not None and h > 0:
        done = set()
        for v in poles:
            v = int(v)
            pv = V[v]
            if any(float(np.linalg.norm(p - pv)) < 0.05 * h for p in verts):
                continue
            for f in mesh.vertex_faces(v):
                f = int(f)
                for k in range(3):
                    if int(F[f, k]) != v:
                        continue
                    z = Uc[f, k]
                    zi = int(round(z.real))
                    zj = int(round(z.imag))
                    if abs(z.real - zi) > 1e-6 or abs(z.imag - zj) > 1e-6:
                        continue
                    key = (int(patches[f]), zi, zj)
                    if key in done:
                        continue
                    done.add(key)
                    for di, dj in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
                        n_cells += 1
                        emit(f, zi + di, zj + dj, pv, pole=(pv, 1.5 * h))

    n_fill = (_fill_boundary(verts, quads, tris, log=debug_log)
              if fill_holes else 0)
    n_merge = 0
    n_repair = 0
    n_pair = 0
    if fill_holes:
        faces = [list(f) for f in quads] + [list(f) for f in tris]
        n_merge = _merge_pinches(verts, faces)
        n_repair = _repair_duplicate_edges(verts, faces, log=debug_log)
        n_pair = _merge_triangle_pairs(verts, faces)
        quads = [f for f in faces if len(f) == 4]
        tris = [f for f in faces if len(f) == 3]
    n_pole = 0
    poles = uvres.get("poles")
    pval = uvres.get("pole_valence")
    if (insert_poles and poles is not None and pval is not None and
            len(poles) == len(pval) and len(poles)):
        faces = [list(f) for f in quads] + [list(f) for f in tris]
        n_pole = _insert_grid_poles(verts, faces,
                                    [V[int(v)] for v in poles],
                                    [int(x) for x in pval], h)
        quads = [f for f in faces if len(f) == 4]
        tris = [f for f in faces if len(f) == 3]
    faces = [list(f) for f in quads] + [list(f) for f in tris]
    used = np.zeros(len(verts), dtype=bool)
    for f in faces:
        used[f] = True
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    verts = [verts[i] for i in np.flatnonzero(used)]
    for f in faces:
        f[:] = [int(remap[i]) for i in f]
    quads = [f for f in faces if len(f) == 4]
    tris = [f for f in faces if len(f) == 3]
    verts = np.asarray(verts, dtype=np.float64)
    quads = np.asarray(quads, dtype=np.int64)
    tris = np.asarray(tris, dtype=np.int64)
    if len(quads):
        qa = verts[quads]
        n0 = np.cross(qa[:, 1] - qa[:, 0], qa[:, 2] - qa[:, 1])
        n1 = np.cross(qa[:, 3] - qa[:, 2], qa[:, 0] - qa[:, 3])
        area = 0.5 * (np.linalg.norm(n0, axis=1) +
                      np.linalg.norm(n1, axis=1))
    else:
        area = np.zeros(0)
    if len(tris):
        ta = verts[tris]
        ta_area = 0.5 * np.linalg.norm(
            np.cross(ta[:, 1] - ta[:, 0], ta[:, 2] - ta[:, 0]), axis=1)
    else:
        ta_area = np.zeros(0)

    ec = Counter()
    for q in quads:
        for k in range(4):
            a, b = int(q[k]), int(q[(k + 1) % 4])
            ec[(min(a, b), max(a, b))] += 1
    for t in tris:
        for k in range(3):
            a, b = int(t[k]), int(t[(k + 1) % 3])
            ec[(min(a, b), max(a, b))] += 1
    inc = Counter(ec.values())
    n_elems = len(quads) + len(tris)
    euler = len(verts) - len(ec) + n_elems
    return {
        "vertices": verts, "quads": quads, "tris": tris, "area": area,
        "tri_area": ta_area,
        "n_cells": n_cells, "n_fail": n_fail, "n_dup": n_dup,
        "n_rep": n_rep, "n_tris": len(tris), "n_fill": n_fill,
        "n_merge": n_merge, "n_repair": n_repair, "n_pair": n_pair,
        "n_reject": n_reject, "n_pole": n_pole,
        "edge_incidence": dict(inc), "euler": euler,
        "n_vertices": len(verts), "n_quads": len(quads),
        "dedup_tol": dedup_tol,
    }


def write_quad_obj(path, vertices, quads, tris=None):
    with open(path, "w", encoding="utf-8") as fh:
        for p in vertices:
            fh.write("v %.9g %.9g %.9g\n" % (p[0], p[1], p[2]))
        for q in quads:
            fh.write("f %d %d %d %d\n" % tuple(int(i) + 1 for i in q))
        if tris is not None:
            for t in tris:
                fh.write("f %d %d %d\n" % tuple(int(i) + 1 for i in t))
