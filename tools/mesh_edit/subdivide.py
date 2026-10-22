"""Catmull-Clark subdivision on triangle or quad arrays, in memory.

Each level splits every n-gon face into n quads (a triangle mesh yields
3 quads per triangle, a quad mesh 4 quads per quad); new vertices are
placed with the standard Catmull-Clark rules:

  * face points   - centroid of the face
  * edge points   - interior edges: average of the 2 endpoints and the
                    2 adjacent face points; boundary edges: midpoint
  * vertex points - interior: (Q + 2R + (n-3)P) / n with Q the average
                    of adjacent face points, R the average of adjacent
                    edge midpoints, n the edge valence;
                    boundary: (6P + M1 + M2) / 8 with M the two
                    boundary-edge midpoints

The result is the familiar smooth "spherify" look, converging to the
limit surface as levels stack.  Pure NumPy, no dependencies; output
keeps quad topology (all faces have 4 corners).  Note the viewer shows
quads fine, but GLB/glTF files are triangles-only, so Save triangulates
them again on disk.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

#: Hard cap; each level is a 3x (tri) / 4x (quad) face blow-up.
MAX_LEVELS = 3

LogFn = Callable[[str, str], None]


def _unique_edges(f: np.ndarray):
    """Return (edges, face_edge_ids) where edges is (nE, 2) sorted
    vertex index pairs and face_edge_ids (nF, k) maps each face corner
    pair (corner i -> corner i+1) to its edge id."""
    nV = int(f.max()) + 1
    k = f.shape[1]
    a = np.concatenate([f[:, i] for i in range(k)])
    b = np.concatenate([np.roll(f, -1, axis=1)[:, i] for i in range(k)])
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    keys = lo * np.int64(nV) + hi
    uniq, inverse = np.unique(keys, return_inverse=True)
    edges = np.stack([uniq // nV, uniq - (uniq // nV) * nV], axis=1)
    # inverse is column-blocked ((c0,c1) for all faces, then (c1,c2),
    # ...) - regroup to one row per face
    return edges, inverse.reshape(k, -1).T


def subdivide_arrays(vertices, faces, *, levels: int = 1,
                     log: LogFn | None = None) -> dict:
    """Subdivide vertex/face arrays ``levels`` Catmull-Clark rounds.

    Faces may be triangles (n, 3) or quads (n, 4); the output is always
    quads (an n-gon face becomes n quads).  Returns a result dict with
    ``vertices``/``faces``, input/output counts and timing.  Nothing is
    written to disk.
    """
    def say(message: str, level: str = "info") -> None:
        if log is not None:
            log(message, level)

    started = time.perf_counter()
    levels = int(max(1, min(MAX_LEVELS, int(levels))))
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] not in (3, 4):
        raise ValueError("faces must be triangles (n, 3) or quads (n, 4)")
    if not len(v) or not len(f):
        raise ValueError("the input mesh has no faces")

    in_v, in_f = len(v), len(f)
    for lv in range(levels):
        v, f = _subdivide_once(v, f)
        say(f"Subdivide: level {lv + 1}/{levels} -> "
            f"{len(v):,} vertices, {len(f):,} quads")

    seconds = time.perf_counter() - started
    say(f"Subdivide done: {in_f:,} -> {len(f):,} quads in "
        f"{seconds:.2f}s; result kept in memory (not saved).", "ok")

    return {
        "mode_label": "Subdivide",
        "vertices": v,
        "faces": f,
        "vertex_count": int(len(v)),
        "face_count": int(len(f)),
        "face_corners": int(f.shape[1]),
        "input_vertex_count": int(in_v),
        "input_face_count": int(in_f),
        "levels": levels,
        "seconds": float(seconds),
    }


def _subdivide_once(v: np.ndarray, f: np.ndarray):
    """One Catmull-Clark round over an n-gon mesh (see module doc)."""
    nV, nF, k = len(v), len(f), f.shape[1]
    edges, fe = _unique_edges(f)          # fe: (nF, k) edge ids per face
    nE = len(edges)
    ea, eb = edges[:, 0], edges[:, 1]

    face_pts = v[f].mean(axis=1)          # (nF, 3) centroid per face

    # faces adjacent to each edge (1 on boundary, 2 interior)
    face_of_entry = np.repeat(np.arange(nF), k)
    order = np.argsort(fe.ravel(), kind="stable")
    sorted_edges = fe.ravel()[order]
    sorted_faces = face_of_entry[order]
    starts = np.searchsorted(sorted_edges, np.arange(nE))
    counts = np.bincount(sorted_edges, minlength=nE)
    has_two = counts == 2
    f0 = np.zeros(nE, dtype=np.int64)
    f1 = np.zeros(nE, dtype=np.int64)
    f0[has_two] = sorted_faces[starts[has_two]]
    f1[has_two] = sorted_faces[starts[has_two] + 1]

    mids = (v[ea] + v[eb]) * 0.5
    edge_pts = mids.copy()
    edge_pts[has_two] = (v[ea[has_two]] + v[eb[has_two]]
                         + face_pts[f0[has_two]] + face_pts[f1[has_two]]) * 0.25

    # vertex points ------------------------------------------------------
    # Q: average of adjacent face points; R: average of adjacent edge mids
    q_sum = np.zeros_like(v)
    q_cnt = np.zeros(nV, dtype=np.int64)
    np.add.at(q_sum, f.ravel(), np.repeat(face_pts, k, axis=0))
    np.add.at(q_cnt, f.ravel(), 1)

    r_sum = np.zeros_like(v)
    r_cnt = np.zeros(nV, dtype=np.int64)
    np.add.at(r_sum, ea, mids)
    np.add.at(r_sum, eb, mids)
    np.add.at(r_cnt, ea, 1)
    np.add.at(r_cnt, eb, 1)

    b_sum = np.zeros_like(v)              # boundary midpoints per vertex
    b_cnt = np.zeros(nV, dtype=np.int64)
    boundary = ~has_two
    np.add.at(b_sum, ea[boundary], mids[boundary])
    np.add.at(b_sum, eb[boundary], mids[boundary])
    np.add.at(b_cnt, ea[boundary], 1)
    np.add.at(b_cnt, eb[boundary], 1)

    new_v = v.copy()
    is_boundary_v = b_cnt > 0
    n = r_cnt.astype(np.float64)
    interior = ~is_boundary_v
    n_int = n[interior]
    # (Q + 2R + (n-3)P)/n with Q, R as averages -> with raw sums the
    # numerator gains an extra factor n: (sumQ + 2 sumR + n(n-3)P)/n^2
    new_v[interior] = ((q_sum[interior] + 2.0 * r_sum[interior]
                        + (n_int * (n_int - 3.0))[:, None] * v[interior])
                       / (n_int * n_int)[:, None])
    new_v[is_boundary_v] = ((6.0 * v[is_boundary_v]
                             + b_sum[is_boundary_v]) / 8.0)

    # assemble k quads per face, one around each original corner --------
    # quad i = (corner i, edge point on edge i->i+1, face point,
    #           edge point on edge i-1->i), preserving orientation.
    fp_off = nV
    ep_off = nV + nF
    fp = fp_off + np.arange(nF)                       # face point ids
    ep = ep_off + fe                                  # (nF, k) edge pt ids

    out = np.empty((nF * k, 4), dtype=np.int64)
    out[:, 0] = f.ravel()                             # corner i per quad
    out[:, 1] = ep.ravel()                            # edge (i, i+1) point
    out[:, 2] = np.repeat(fp, k)                      # face point
    out[:, 3] = np.roll(ep, 1, axis=1).ravel()        # edge (i-1, i) point

    return np.concatenate((new_v, face_pts, edge_pts), axis=0), out
