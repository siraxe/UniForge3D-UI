"""Cut-graph construction: separatrix tracing and seam completion."""
from __future__ import annotations

import heapq
from collections import defaultdict, deque

import numpy as np
import scipy.sparse as sp

from ._backend import accelerator
from .field import _edge_frame_angles
from .mesh import TriMesh


def face_field_dirs(mesh: TriMesh, theta, frame_u, frame_v):
    d = np.cos(theta)[:, None] * frame_u + np.sin(theta)[:, None] * frame_v
    n = np.linalg.norm(d, axis=1)
    ok = n > 1e-12
    d[ok] /= n[ok, None]
    return d

def _ray_exit(mesh, f, p, D, skip_edge=None):
    F = mesh.F
    tri = mesh.V[F[f]]
    e_ids = [int(e) for e in mesh.face_edge_ids[f]]
    best = None
    corner = None
    o = tri[0]
    E1 = tri[1] - tri[0]
    E2 = tri[2] - tri[0]
    nrm = np.cross(E1, E2)
    ref = E1 / (np.linalg.norm(E1) + 1e-300)
    y = np.cross(nrm, ref)
    y /= (np.linalg.norm(y) + 1e-300)
    def to2(Q):
        return np.array([np.dot(Q - o, ref), np.dot(Q - o, y)])
    p2 = to2(p)
    d2 = np.array([np.dot(D, ref), np.dot(D, y)])
    for k in range(3):
        eid = e_ids[k]
        if skip_edge is not None and eid == skip_edge:
            continue
        a2 = to2(tri[k])
        b2 = to2(tri[(k + 1) % 3])
        e2 = b2 - a2
        det = d2[0] * (-e2[1]) - d2[1] * (-e2[0])
        if abs(det) < 1e-300:
            continue
        rhs = a2 - p2
        t = (rhs[0] * (-e2[1]) - rhs[1] * (-e2[0])) / det
        s = (d2[0] * rhs[1] - d2[1] * rhs[0]) / det
        if t > 1e-12:
            if -1e-9 <= s <= 1 + 1e-9:
                if best is None or t < best[2]:
                    best = (eid, a2 + s * e2, t)
            elif corner is None or t < corner[2]:
                corner = (k, s, t)
    if best is not None:
        eid, ex2, _ = best
        ex = o + ex2[0] * ref + ex2[1] * y
        return ("edge", eid, ex)
    if corner is not None:
        k, s, _ = corner
        w = int(F[f][(k + 1) % 3]) if s > 0 else int(F[f][k])
        return ("vertex", w, None)
    return None

def trace_separatrices(mesh: TriMesh, theta, transitions, frame_u, frame_v,
                       singular_vertices, max_steps_factor=50, progress=None):
    singv = np.asarray(singular_vertices, dtype=np.int64).ravel()
    accel = accelerator()
    if accel is None:
        return _trace_sequential(mesh, theta, transitions, frame_u, frame_v,
                                 singv, max_steps_factor, progress)

    V, F = mesh.V, mesh.F
    dirs = face_field_dirs(mesh, theta, frame_u, frame_v)
    fL, fR = mesh.edge_faces()
    max_steps = max(64, int(max_steps_factor * len(F)))
    cut_mask, ray_steps, ray_why, ray_count = accel.trace_separatrices(
        V, F, mesh.face_edge_ids, mesh.edges, fL, fR, mesh.face_normals,
        dirs, mesh._vface_start, mesh._vface_order,
        mesh._vedge_start, mesh._vedge_order, singv, max_steps,
        progress=progress)
    cut_edges = set(int(e) for e in np.flatnonzero(cut_mask))

    n_sing = len(singv)
    if n_sing:
        valid = np.arange(4 * n_sing).reshape(n_sing, 4) < \
            ray_count[:, None]
        line_edges = ray_steps.reshape(n_sing, 4)[valid].tolist()
        why = ray_why.reshape(n_sing, 4)[valid]
        names = {1: "sing", 2: "boundary", 3: "reversed", 4: "maxsteps",
                 5: "noexit_launch", 6: "noexit_edge", 7: "noexit_corner",
                 8: "merge"}
        end_reasons = {}
        for code, count in zip(*np.unique(why, return_counts=True)):
            key = names.get(int(code), "noexit_%d" % int(code))
            end_reasons[key] = int(count)
    else:
        line_edges = []
        end_reasons = {}
    return _finish_trace(mesh, cut_edges, [], end_reasons, line_edges)


def _trace_sequential(mesh: TriMesh, theta, transitions, frame_u, frame_v,
                      singular_vertices, max_steps_factor=50,
                      progress=None):
    """Portable scalar separatrix tracer (the original implementation)."""
    V, F = mesh.V, mesh.F
    dirs = face_field_dirs(mesh, theta, frame_u, frame_v)
    fL, fR = mesh.edge_faces()
    inner = fR >= 0
    fl_i, fr_i = fL[inner], fR[inner]
    e_lo, e_hi = mesh.edges[inner, 0], mesh.edges[inner, 1]
    d = mesh.V[e_hi] - mesh.V[e_lo]

    def ang(fidx):
        return np.arctan2(np.sum(d * frame_v[fidx], axis=1),
                          np.sum(d * frame_u[fidx], axis=1))

    phi = ang(fl_i) - ang(fr_i)
    edge_cross = {}
    for i in range(len(fl_i)):
        edge_cross[int(np.flatnonzero(inner)[i])] = (fl_i[i], fr_i[i])

    midpoint = 0.5 * (mesh.V[mesh.edges[:, 0]] + mesh.V[mesh.edges[:, 1]])
    emean = np.mean(mesh.edge_lengths() if hasattr(mesh, "edge_lengths")
                    else np.linalg.norm(mesh.V[mesh.edges[:, 1]] -
                                        mesh.V[mesh.edges[:, 0]], axis=1))
    max_steps = max(64, int(max_steps_factor * len(F)))

    cut_edges = set()
    polylines = []
    end_reasons = {}
    line_edges = []
    singular_set = set(int(v) for v in singular_vertices)

    singular_vertices = list(singular_vertices)
    n_sing = max(1, len(singular_vertices))
    for s_i, sv in enumerate(singular_vertices):
        if progress is not None:
            progress(s_i / n_sing)
        inc_faces = mesh.vertex_faces(sv)
        v_pos = mesh.V[sv]
        best = None
        for f0 in inc_faces:
            nf0 = mesh.face_normals[f0]
            away = mesh.V[F[f0]].mean(axis=0) - v_pos
            al = np.linalg.norm(away)
            if al < 1e-14:
                continue
            away /= al
            df = dirs[f0] - nf0 * np.dot(dirs[f0], nf0)
            dl = np.linalg.norm(df)
            if dl < 1e-12:
                continue
            df /= dl
            perp = np.cross(nf0, df)
            for dr in (df, perp, -df, -perp):
                score = float(np.dot(dr, away))
                if best is None or score > best[0]:
                    best = (score, int(f0), df, perp)
        if best is None:
            continue
        _, f_best, df_best, perp_best = best
        sectors = (df_best, perp_best, -df_best, -perp_best)
        cands = []
        for f0 in inc_faces:
            nf0 = mesh.face_normals[f0]
            away = mesh.V[F[f0]].mean(axis=0) - v_pos
            al = np.linalg.norm(away)
            if al < 1e-14:
                continue
            away /= al
            dfc = dirs[f0] - nf0 * np.dot(dirs[f0], nf0)
            dlc = np.linalg.norm(dfc)
            if dlc < 1e-12:
                continue
            dfc /= dlc
            perpc = np.cross(nf0, dfc)
            for q, drc in enumerate((dfc, perpc, -dfc, -perpc)):
                cands.append((int(f0), q, drc, float(np.dot(drc, away))))
        seen = set()
        n_launched = 0
        for r_i in sectors:
            avail = sorted([c for c in cands if (c[0], c[1]) not in seen
                            and c[3] > 0.1],
                           key=lambda c: -float(np.dot(c[2], r_i)))
            f0 = q = dr = None
            for c in avail:
                if _ray_exit(mesh, int(c[0]), mesh.V[sv], c[2]) is not None:
                    f0, q, dr = int(c[0]), c[1], c[2]
                    break
            if f0 is None:
                continue
            seen.add((f0, q))
            n_launched += 1
            best_face, best_dir = f0, dr
            pts = [mesh.V[sv].copy()]
            f_cur = int(best_face)
            p = mesh.V[sv].copy()
            hit = _ray_exit(mesh, f_cur, p, best_dir)
            prev_eid = None
            steps = 0
            ended = False
            why = "noexit"
            last_kind = "launch"
            while hit is not None and steps < max_steps:
                steps += 1
                kind, a1, a2 = hit
                if kind == "vertex":
                    w = int(a1)
                    pts.append(mesh.V[w].copy())
                    if w in singular_set and steps > 1:
                        ended = True
                        why = "sing"
                        break
                    if steps > 1 and any(
                            int(e) in cut_edges
                            for e in mesh.vertex_edges(w)
                            if int(e) != prev_eid):
                        ended = True
                        why = "merge"
                        break
                    e2 = -1
                    for e2c in mesh.face_edge_ids[f_cur]:
                        e2c = int(e2c)
                        if e2c != prev_eid and (mesh.edges[e2c, 0] == w or
                                                mesh.edges[e2c, 1] == w):
                            e2 = e2c
                            break
                    if e2 < 0 or fL[e2] == -1 or fR[e2] == -1:
                        ended = True
                        why = "boundary"
                        break
                    cut_edges.add(e2)
                    fa2, fb2 = fL[e2], fR[e2]
                    f2 = int(fb2 if fa2 == f_cur else fa2)
                    Dprev = pts[-1] - pts[-2]
                    n2 = mesh.face_normals[f2]
                    Dp = Dprev - n2 * np.dot(Dprev, n2)
                    df2 = dirs[f2]
                    perp2 = np.cross(n2, df2)
                    D = max((df2, perp2, -df2, -perp2),
                            key=lambda c: np.dot(Dp, c))
                    f_cur = f2
                    prev_eid = e2
                    last_kind = "corner"
                    hit = _ray_exit(mesh, f_cur, midpoint[e2], D, skip_edge=e2)
                    continue
                eid, ex = int(a1), a2
                pts.append(ex.copy())
                if eid in cut_edges:
                    ended = True
                    why = "merge"
                    break
                va, vb = int(mesh.edges[eid, 0]), int(mesh.edges[eid, 1])
                if singular_set and steps > 1:
                    elen = np.linalg.norm(mesh.V[va] - mesh.V[vb])
                    for ve_ in (va, vb):
                        if ve_ in singular_set and \
                                np.linalg.norm(ex - mesh.V[ve_]) < 0.5 * elen:
                            pts[-1] = mesh.V[ve_].copy()
                            cut_edges.add(eid)
                            ended = True
                            why = "sing"
                            break
                    if ended:
                        break
                if fL[eid] == -1 or fR[eid] == -1:
                    ended = True
                    why = "boundary"
                    break
                cut_edges.add(eid)
                fa, fb = fL[eid], fR[eid]
                f_next = int(fb if fa == f_cur else fa)
                Dprev = pts[-1] - pts[-2]
                n_next = mesh.face_normals[f_next]
                Dp = Dprev - n_next * np.dot(Dprev, n_next)
                df = dirs[f_next]
                perp = np.cross(n_next, df)
                Dc = max((df, perp, -df, -perp), key=lambda c: np.dot(Dp, c))
                if np.dot(Dp, Dc) <= 0.0:
                    ended = True
                    why = "reversed"
                    break
                D = Dc
                p_start = midpoint[eid]
                f_cur = f_next
                prev_eid = eid
                last_kind = "edge"
                hit = _ray_exit(mesh, f_cur, p_start, D, skip_edge=eid)
            polylines.append((np.array(pts), not ended and steps >= max_steps))
            if steps >= max_steps:
                why = "maxsteps"
            elif hit is None:
                why = "noexit_" + last_kind
            end_reasons[why] = end_reasons.get(why, 0) + 1
            line_edges.append(steps)

    return _finish_trace(mesh, cut_edges, polylines, end_reasons, line_edges)


def _finish_trace(mesh, cut_edges, polylines, end_reasons, line_edges):
    """Connect remaining cut components through shortest face paths."""
    F = mesh.F
    fL, fR = mesh.edge_faces()
    elen = np.linalg.norm(mesh.V[mesh.edges[:, 0]] -
                          mesh.V[mesh.edges[:, 1]], axis=1)
    face_nbrs = []
    for f in range(len(F)):
        lst = []
        for k in range(3):
            e = int(mesh.face_edge_ids[f, k])
            a, b = fL[e], fR[e]
            g = b if a == f else a
            if g >= 0:
                lst.append((int(g), e))
        face_nbrs.append(lst)

    bridges_added = 0
    for _ in range(64):
        cadj = {}
        for e in cut_edges:
            a, b = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
            cadj.setdefault(a, []).append(b)
            cadj.setdefault(b, []).append(a)
        comp = {}
        cid = 0
        for s in cadj:
            if s in comp:
                continue
            stack = [s]
            comp[s] = cid
            while stack:
                x = stack.pop()
                for y in cadj[x]:
                    if y not in comp:
                        comp[y] = cid
                        stack.append(y)
            cid += 1
        if cid <= 1:
            break
        INF = float("inf")
        dist = [INF] * len(F)
        label = [-1] * len(F)
        par = [None] * len(F)
        pq = []
        for e in cut_edges:
            a, b = fL[e], fR[e]
            for f in (a, b):
                if f >= 0 and dist[f] > 0:
                    dist[f] = 0.0
                    label[f] = comp[int(mesh.edges[e, 0])]
                    pq.append((0.0, f))
        heapq.heapify(pq)
        while pq:
            d, f = heapq.heappop(pq)
            if d > dist[f] + 1e-300:
                continue
            for g, e in face_nbrs[f]:
                if e in cut_edges:
                    continue
                nd = d + elen[e]
                if nd < dist[g] - 1e-300:
                    dist[g] = nd
                    label[g] = label[f]
                    par[g] = (f, e)
                    heapq.heappush(pq, (nd, g))
        meet = None
        best_cost = INF
        for e in range(len(mesh.edges)):
            a, b = fL[e], fR[e]
            if e in cut_edges or b < 0:
                continue
            la, lb = label[a], label[b]
            if la >= 0 and lb >= 0 and la != lb:
                cost = dist[a] + dist[b] + elen[e]
                if cost < best_cost:
                    best_cost = cost
                    meet = (int(a), int(b), e)
        if meet is None:
            break
        f, g, e = meet
        path = [e]
        x = f
        while par[x] is not None:
            path.append(par[x][1])
            x = par[x][0]
        x = g
        while par[x] is not None:
            path.append(par[x][1])
            x = par[x][0]
        for e in path:
            cut_edges.add(e)
        bridges_added += 1

    return {"cut_edges": cut_edges, "polylines": polylines,
            "end_reasons": end_reasons, "line_edges": line_edges,
            "bridges_added": bridges_added}



def _uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def _uf_union(parent, rank, a, b):
    a, b = _uf_find(parent, a), _uf_find(parent, b)
    if a == b:
        return
    if rank[a] < rank[b]:
        a, b = b, a
    parent[b] = a
    if rank[a] == rank[b]:
        rank[a] += 1

def unwrap_field_branches(mesh, theta, frame_u, frame_v, cut_edges):
    """Lift theta (mod pi/2) to a continuous 3D field."""
    nF = mesh.n_faces
    cut = set(int(e) for e in cut_edges)
    fl, fr, m4, inner = _edge_frame_angles(mesh, frame_u, frame_v)
    eids = np.flatnonzero(inner)
    phi = 0.25 * m4
    nbrs = [[] for _ in range(nF)]
    for i, e in enumerate(eids):
        if int(e) in cut:
            continue
        l, r = int(fl[i]), int(fr[i])
        nbrs[r].append((l, float(phi[i])))
        nbrs[l].append((r, -float(phi[i])))
    thg = np.empty(nF)
    seen = np.zeros(nF, dtype=bool)
    for root in range(nF):
        if seen[root]:
            continue
        seen[root] = True
        thg[root] = theta[root]
        dq = deque([root])
        while dq:
            f = dq.popleft()
            for g, off in nbrs[f]:
                if seen[g]:
                    continue
                seen[g] = True
                want = thg[f] + off
                p = int(round((want - theta[g]) / (np.pi / 2.0)))
                thg[g] = theta[g] + p * (np.pi / 2.0)
                dq.append(g)
    return thg

def build_cut_mesh(mesh, cut_edges):
    """Cut along edges; union corners into UV vertices."""
    nF = mesh.n_faces
    cut = set(int(e) for e in cut_edges)
    fL, fR = mesh.edge_faces()
    Fpy = mesh.F.tolist()

    parent = np.arange(nF)
    rank = np.zeros(nF, dtype=np.int32)
    for e in range(mesh.n_edges):
        if fR[e] < 0 or e in cut:
            continue
        _uf_union(parent, rank, int(fL[e]), int(fR[e]))
    froots = np.array([_uf_find(parent, f) for f in range(nF)])
    proots, patch_of_face = np.unique(froots, return_inverse=True)

    parent = np.arange(3 * nF)
    rank = np.zeros(3 * nF, dtype=np.int32)

    def corner_of(f, v):
        t = Fpy[f]
        if t[0] == v:
            return 3 * f
        if t[1] == v:
            return 3 * f + 1
        return 3 * f + 2

    for e in range(mesh.n_edges):
        if fR[e] < 0 or e in cut:
            continue
        l, r = int(fL[e]), int(fR[e])
        a, b = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
        _uf_union(parent, rank, corner_of(l, a), corner_of(r, a))
        _uf_union(parent, rank, corner_of(l, b), corner_of(r, b))
    croots = np.array([_uf_find(parent, i) for i in range(3 * nF)])
    _, inv = np.unique(croots, return_inverse=True)
    corner_uv = inv.reshape(nF, 3)
    n_uv = int(corner_uv.max()) + 1
    uv_patch = np.full(n_uv, -1, dtype=int)
    uv_patch[corner_uv] = patch_of_face[:, None]

    cuts = []
    for e in sorted(cut):
        l, r = int(fL[e]), int(fR[e])
        if r < 0:
            continue
        a, b = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
        cuts.append((e, int(corner_uv[l, 0]) if Fpy[l][0] == a else
                     int(corner_uv[l, 1]) if Fpy[l][1] == a else
                     int(corner_uv[l, 2]),
                     int(corner_uv[l, 0]) if Fpy[l][0] == b else
                     int(corner_uv[l, 1]) if Fpy[l][1] == b else
                     int(corner_uv[l, 2]),
                     int(corner_uv[r, 0]) if Fpy[r][0] == a else
                     int(corner_uv[r, 1]) if Fpy[r][1] == a else
                     int(corner_uv[r, 2]),
                     int(corner_uv[r, 0]) if Fpy[r][0] == b else
                     int(corner_uv[r, 1]) if Fpy[r][1] == b else
                     int(corner_uv[r, 2])))
    return {"corner_uv": corner_uv, "n_uv": n_uv,
            "patch_of_face": patch_of_face, "n_patches": len(proots),
            "uv_patch": uv_patch, "cut_pairs": cuts}

def _fan_cut_signs(mesh, cut, v):
    """Cut-spoke orientation sign along the face fan at v."""
    fL, fR = mesh.edge_faces()
    F = mesh.F
    fid = mesh.face_edge_ids
    f0 = int(mesh.vertex_faces(v)[0])
    f = f0
    out = {}
    for _ in range(1000):
        k = int(np.flatnonzero(F[f] == v)[0])
        e = int(fid[f, k])
        if e in cut:
            out[e] = 1 if int(fL[e]) == f else -1
        g = int(fR[e]) if int(fL[e]) == f else int(fL[e])
        if g < 0 or g == f0:
            break
        f = g
    return out

def assign_seam_rotations(mesh, cut_edges, sing_idx):
    cut = sorted(set(int(e) for e in cut_edges))
    E = mesh.edges
    verts = sorted(set(int(x) for e in cut for x in E[e]))
    vix = {v: i for i, v in enumerate(verts)}
    signs = {v: _fan_cut_signs(mesh, set(cut), v) for v in verts}
    tgt = np.array([(-int(sing_idx[v])) % 4 for v in verts], dtype=int)

    adj = [[] for _ in verts]
    for e in cut:
        a, b = int(E[e, 0]), int(E[e, 1])
        ia, ib = vix[a], vix[b]
        sa = signs[a].get(e)
        sb = signs[b].get(e)
        if sa is None or sb is None:
            raise RuntimeError("cut edge endpoint not on fan")
        adj[ia].append((ib, e, sa))
        adj[ib].append((ia, e, sb))

    k_of = {e: 0 for e in cut}
    seen = np.zeros(len(verts), dtype=bool)
    bad = []
    for root in range(len(verts)):
        if seen[root]:
            continue
        seen[root] = True
        par = {root: (None, 0)}
        order = [root]
        dq = deque([root])
        while dq:
            u = dq.popleft()
            for w, e, s in adj[u]:
                if seen[w]:
                    continue
                seen[w] = True
                par[w] = (u, e)
                order.append(w)
                dq.append(w)
        for u in reversed(order):
            r = tgt[u]
            for w, e, s in adj[u]:
                if par.get(w, (None, None))[0] == u:
                    r -= signs[verts[u]][e] * k_of[e]
            r %= 4
            pu = par[u][0]
            if pu is None:
                if r:
                    bad.append((verts[u], r))
                continue
            eup = par[u][1]
            k_of[eup] = signs[verts[u]][eup] * r % 4
    if bad:
        raise RuntimeError("inconsistent Z4 fan targets: %s" % bad[:8])
    return k_of

def augment_cut_for_singularities(mesh, cut_edges, singular_vertices):
    E = mesh.edges
    cut = set(int(e) for e in cut_edges)
    terms = set(int(s) for s in singular_vertices)
    cv = set(int(x) for e in cut for x in E[e])
    missing = terms - cv
    if missing:
        nV = mesh.n_vertices
        L = np.linalg.norm(mesh.V[E[:, 0]] - mesh.V[E[:, 1]], axis=1)
        src = np.array(sorted(cv), dtype=int) if cv else \
            np.array([min(missing)])
        # One Dijkstra from a virtual source joined to every cut vertex with
        # zero-weight edges.  Equivalent to the old per-source table, but
        # O(V) memory instead of O(len(src) * V).
        aux = nV
        W = sp.coo_matrix(
            (np.concatenate([L, L, np.zeros(len(src))]),
             (np.concatenate([E[:, 0], E[:, 1], np.full(len(src), aux)]),
              np.concatenate([E[:, 1], E[:, 0], src]))),
            shape=(nV + 1, nV + 1)).tocsr()
        pred = sp.csgraph.dijkstra(W, indices=aux,
                                   return_predecessors=True)[1]
        lookup = {(int(E[e, 0]), int(E[e, 1])): e for e in range(len(E))}
        for s in sorted(missing):
            v = s
            path = [v]
            while v not in cv:
                v = int(pred[v])
                if v < 0 or v == aux:
                    raise RuntimeError("no path to cut graph")
                path.append(v)
            for a, b in zip(path[:-1], path[1:]):
                e = lookup[(min(a, b), max(a, b))]
                cut.add(int(e))
                cv.update([a, b])
    while True:
        deg = defaultdict(int)
        for e in cut:
            for x in E[e]:
                deg[int(x)] += 1
        rm = [e for e in cut
              if any(deg[int(x)] == 1 and int(x) not in terms for x in E[e])]
        if not rm:
            break
        cut -= set(rm)
    return sorted(cut)

def complete_cut_for_field_consistency(mesh, theta, frame_u, frame_v,
                                       cut_edges, max_iter=8,
                                       max_ends=40):
    """Seam edges carrying branch jumps; attach slit ends."""
    E = mesh.edges
    nV = mesh.n_vertices
    cut = set(int(e) for e in cut_edges)
    fl, fr, m4, inner = _edge_frame_angles(mesh, frame_u, frame_v)
    eids = np.flatnonzero(inner)
    phi = 0.25 * m4
    L = np.linalg.norm(mesh.V[E[:, 0]] - mesh.V[E[:, 1]], axis=1)
    attached = set(int(x) for e in cut for x in E[e])
    extra = set()
    for _ in range(max_iter):
        thg = unwrap_field_branches(mesh, theta, frame_u, frame_v,
                                    cut | extra)
        q = np.round((thg[fl] - thg[fr] - phi) / (np.pi / 2.0)).astype(int)
        noncut = np.array([int(e) not in cut and int(e) not in extra
                           for e in eids])
        dep = set(int(e) for e in eids[noncut & (np.mod(q, 4) != 0)])
        if not dep:
            break
        extra |= dep
        if not attached:
            attached.update(int(x) for e in dep for x in E[e])
            continue
        deg = defaultdict(int)
        for e in extra:
            for x in E[e]:
                deg[int(x)] += 1
        ends = [x for x, c in deg.items() if c == 1 and x not in attached]
        if not ends or len(ends) > max_ends:
            continue
        src = np.array(sorted(attached), dtype=int)
        W = sp.coo_matrix(
            (np.concatenate([L, L, np.zeros(len(src))]),
             (np.concatenate([E[:, 0], E[:, 1], np.full(len(src), nV)]),
              np.concatenate([E[:, 1], E[:, 0], src]))),
            shape=(nV + 1, nV + 1)).tocsr()
        dist, pred = sp.csgraph.dijkstra(W, indices=nV,
                                         return_predecessors=True)
        lookup = {(int(E[e, 0]), int(E[e, 1])): e for e in range(len(E))}
        for v in ends:
            if not np.isfinite(dist[v]):
                continue
            path = [v]
            while v not in attached:
                v = int(pred[v])
                if v < 0 or v == nV:
                    break
                path.append(v)
            for a, b in zip(path[:-1], path[1:]):
                e = lookup[(min(a, b), max(a, b))]
                extra.add(int(e))
            attached.update(path)
    return sorted(cut | extra)
