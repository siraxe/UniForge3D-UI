"""Cut the mesh with a plane, keep one side, mirror it back and weld.

Mirror-and-merge turns a half-decent model into an exactly symmetric
one: every plane given (world-space axis planes; the caller converts the
checked X/Y/Z checkboxes and the pivot into ``(normal, offset)`` pairs in
the arrays' coordinate space) cuts the mesh, the picked side is kept,
reflected across the plane with flipped winding, and welded to the
original along the cut so the result is one continuous surface.

How each plane is applied:
* faces entirely on the keep side survive unchanged (quads stay quads),
* faces entirely on the delete side are dropped,
* faces crossing the plane are clipped polygon-style (Sutherland-Hodgman
  against the plane) and fan-triangulated, so the cut is watertight -
  no T-junctions, no boundary loop to fill afterwards,
* vertices within ``_PLANE_EPS`` of the plane snap exactly onto it, so
  their mirror images land on the very same coordinates and the weld is
  an exact index mapping rather than a tolerance search.

Nothing is written to disk; the result dict matches the other
``*_arrays`` tools and composes with the in-memory edit stack.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np

LogFn = Callable[[str, str], None]

#: vertices closer than this (coordinate units) to the cut plane snap
#: exactly onto it, which is what makes the mirror weld exact
_PLANE_EPS = 1e-6


def _apply_plane(vertices, faces, normal, offset):
    """One cut + keep + mirror + weld pass; returns (vertices, faces)."""
    n = np.asarray(normal, dtype=np.float64)
    length = float(np.linalg.norm(n))
    if length < 1e-12:
        raise ValueError("mirror plane normal must be non-zero")
    n = n / length
    offset = float(offset)
    dist = vertices @ n - offset

    # snap near-plane vertices exactly onto the plane: every on-plane
    # original vertex then has dist == 0.0 bit-exact, which the
    # classification and the mirror weld below rely on (cut vertices are
    # tracked as on-plane by construction - their float coordinates are
    # never re-tested against the plane)
    snap = np.abs(dist) <= _PLANE_EPS
    if snap.any():
        vertices = vertices.copy()
        vertices[snap] -= dist[snap][:, None] * n
        dist = vertices @ n - offset
    on_plane = dist == 0.0

    # classify faces: 0 = delete, 1 = keep unchanged, 2 = crossing
    d_all = dist[faces]
    kind = np.full(len(faces), 2, dtype=np.int8)
    kind[(d_all <= 0.0).all(axis=1)] = 1
    kind[(d_all >= 0.0).all(axis=1)] = 0
    kind[(d_all == 0.0).all(axis=1)] = 1   # a face lying flat on the
    # plane is boundary; keeping it welds it to its own mirror image

    # -- cut vertices, one per crossing edge ----------------------------------
    # keyed by the UNDIRECTED edge (lo, hi), so the two faces sharing an
    # edge share the interpolated vertex - the cut cannot crack
    cut_of = {}
    cut_pos = []
    for face in faces[kind == 2]:
        k = len(face)
        for i in range(k):
            a, b = int(face[i]), int(face[(i + 1) % k])
            da, db = dist[a], dist[b]
            if not ((da < 0.0 < db) or (db < 0.0 < da)):
                continue
            key = (a, b) if a < b else (b, a)
            if key in cut_of:
                continue
            t = da / (da - db)
            cut_of[key] = len(vertices) + len(cut_pos)
            cut_pos.append(vertices[a] + (vertices[b] - vertices[a]) * t)

    # extend the tables with the cut vertices so compaction can index them
    if cut_pos:
        all_v = np.vstack((vertices, np.asarray(cut_pos)))
        on_all = np.concatenate((on_plane,
                                 np.ones(len(cut_pos), dtype=bool)))
    else:
        all_v = vertices
        on_all = on_plane

    # -- rebuild the keep-side surface ----------------------------------------
    kept_faces = []
    for fi, face in enumerate(faces):
        if kind[fi] == 0:
            continue
        if kind[fi] == 1:
            kept_faces.append(face)
            continue
        poly = [int(v) for v in face]
        clipped = []
        k = len(poly)
        for i in range(k):
            a, b = poly[i], poly[(i + 1) % k]
            da, db = dist[a], dist[b]
            if da <= 0.0:
                clipped.append(a)
            if (da < 0.0 < db) or (db < 0.0 < da):
                clipped.append(cut_of[(a, b) if a < b else (b, a)])
        for i in range(1, len(clipped) - 1):     # fan-triangulate
            kept_faces.append((clipped[0], clipped[i], clipped[i + 1]))
    if not kept_faces:
        raise ValueError("the plane deletes every face of the mesh")
    widths = {len(face) for face in kept_faces}
    if len(widths) > 1:
        # mixed quads (kept unchanged) and triangles (clipped): fan the
        # quads so the face table stays uniform
        tri_faces = []
        for face in kept_faces:
            if len(face) == 3:
                tri_faces.append(face)
            else:
                for i in range(1, len(face) - 1):
                    tri_faces.append((face[0], face[i], face[i + 1]))
        kept_faces = tri_faces
    kept_faces = np.asarray(kept_faces, dtype=np.int64)

    # -- compact the vertex table ---------------------------------------------
    used, inverse = np.unique(kept_faces, return_inverse=True)
    base_v = all_v[used]
    base_f = inverse.reshape(kept_faces.shape)

    # -- mirror append with flipped winding -----------------------------------
    # on-plane kept vertices (snapped boundary + cut vertices) map onto
    # themselves; everything else gets a fresh reflected copy
    on_kept = on_all[used]
    mirror_map = np.empty(len(base_v), dtype=np.int64)
    mirror_map[on_kept] = np.nonzero(on_kept)[0]
    off = ~on_kept
    mirror_map[off] = len(base_v) + np.arange(int(off.sum()), dtype=np.int64)
    mirrored_v = base_v[off] - 2.0 * ((base_v[off] @ n - offset)[:, None] * n)
    out_v = np.vstack((base_v, mirrored_v))
    out_f = np.vstack((base_f, mirror_map[base_f[:, ::-1]]))
    return out_v, out_f


def mirror_merge_arrays(vertices, faces, planes, *,
                        log: LogFn | None = None) -> dict:
    """Cut/keep/mirror/weld across every ``(normal, offset)`` plane.

    ``planes`` is an iterable of pairs in the arrays' coordinate space;
    they are applied in order (two planes chain into a quarter mirror).
    Returns the merged mesh plus statistics in the standard result dict.
    """
    def say(message: str, level: str = "info") -> None:
        if log is not None:
            log(message, level)

    started = time.perf_counter()
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] not in (3, 4):
        raise ValueError("faces must be triangles or quads ((n, 3)/(n, 4))")
    if not len(f):
        raise ValueError("the input mesh has no faces")
    planes = list(planes)
    if not planes:
        raise ValueError("no mirror plane was given")

    in_v, in_f = len(v), len(f)
    for normal, offset in planes:
        v, f = _apply_plane(v, f, normal, offset)

    seconds = time.perf_counter() - started
    say(f"Mirror and merge: {in_v:,} -> {len(v):,} vertices, "
        f"{in_f:,} -> {len(f):,} faces across {len(planes)} plane(s).", "ok")
    return {
        "mode_label": "Mirror and merge",
        "vertices": np.ascontiguousarray(v, dtype=np.float64),
        "faces": np.ascontiguousarray(f, dtype=np.int64),
        "vertex_count": int(len(v)),
        "face_count": int(len(f)),
        "input_vertex_count": int(in_v),
        "input_face_count": int(in_f),
        "seconds": float(seconds),
    }
