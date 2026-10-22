"""Detect and close boundary holes in a triangle mesh, in memory.

Detection: every undirected edge used by exactly one triangle is a
boundary edge; the boundary edge count (and the loop listing via
pymeshfix) tells us how holey the mesh is.

Filling uses pymeshfix (MeshFix) when available -
``PyTMesh.fill_small_boundaries`` triangulates each boundary loop and,
with ``refine=True``, adds a few interior points so larger holes get a
nicely subdivided cap instead of one giant flat fan.  Only new
vertices/faces are appended; the existing ones keep their exact
geometry and ordering.  Without pymeshfix the tool falls back to
trimesh's ``repair.fill_holes`` (no new vertices, weaker but
dependency-free).
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

LogFn = Callable[[str, str], None]


def _boundary_stats(f: np.ndarray):
    """(open undirected edge count, boundary loop count) of a mesh."""
    nV = int(f.max()) + 1 if len(f) else 0
    if nV == 0 or not len(f):
        return 0, 0
    a = f.ravel()
    b = np.roll(f, -1, axis=1).ravel()
    keys = np.minimum(a, b) * np.int64(nV) + np.maximum(a, b)
    _, counts = np.unique(keys, return_counts=True)
    return int((counts == 1).sum()), 0


def _boundary_loops(f: np.ndarray):
    """List boundary loops via pymeshfix (best effort; may be empty)."""
    try:
        import pymeshfix
    except Exception:  # noqa: BLE001 - optional dependency
        return []
    try:
        fixer = pymeshfix.PyTMesh()
        fixer.set_quiet(True)
        fixer.load_array(np.ascontiguousarray(f.astype(np.int32)))
        return fixer.boundaries()
    except Exception:  # noqa: BLE001 - diagnostics only
        return []


def close_holes_arrays(vertices, faces, *, max_loop_edges: int = 0,
                       refine: bool = True,
                       log: LogFn | None = None) -> dict:
    """Close the boundary holes of a triangle mesh.

    ``vertices`` (n, 3) float and ``faces`` (m, 3) int are returned
    alongside the originals in a result dict with hole statistics and
    timing.  Nothing is written to disk; existing vertices/faces are
    never modified, only appended to.  ``max_loop_edges`` limits the
    size of the holes that get closed (0 = close every hole).
    """
    def say(message: str, level: str = "info") -> None:
        if log is not None:
            log(message, level)

    started = time.perf_counter()
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must be triangles (n, 3); triangulate "
                         "quads first")
    if not len(v) or not len(f):
        raise ValueError("the input mesh has no faces")

    in_v, in_f = len(v), len(f)
    open_before, _ = _boundary_stats(f)
    loops = _boundary_loops(f)
    if open_before == 0:
        seconds = time.perf_counter() - started
        say("Close holes: the mesh has no boundary edges - already "
            "watertight.", "ok")
        return _result(v, f, in_v, in_f, 0, 0, 0, "none", seconds)

    say(f"Close holes: {open_before} boundary edge(s)"
        + (f" in {len(loops)} loop(s)" if loops else "")
        + " to close.", "info")

    engine = "pymeshfix"
    try:
        import pymeshfix
        fixer = pymeshfix.PyTMesh()
        fixer.set_quiet(True)
        fixer.load_array(np.ascontiguousarray(v),
                         np.ascontiguousarray(f.astype(np.int32)))
        fixer.fill_small_boundaries(nbe=int(max_loop_edges),
                                    refine=bool(refine))
        out_v, out_f = fixer.return_arrays()
    except ImportError:
        engine = "trimesh"
        import trimesh
        from trimesh.repair import fill_holes
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        try:
            fill_holes(mesh)
        except Exception as ex:  # noqa: BLE001 - keep partial results
            say(f"Close holes: the trimesh filler stopped early ({ex}).",
                "warn")
        out_v, out_f = mesh.vertices, mesh.faces
    except Exception as ex:  # noqa: BLE001 - report, keep the input
        say(f"Close holes: the filler failed ({ex}); mesh unchanged.",
            "error")
        out_v, out_f = v, f

    out_v = np.asarray(out_v, dtype=np.float64)
    out_f = np.asarray(out_f, dtype=np.int64)
    if engine != "none" and len(out_f) > in_f:
        out_v, out_f, exact = _restore_input_order(v, f, out_v, out_f)
        if not exact:
            say("Close holes: the filler reindexed the mesh; the result "
                "keeps every original vertex position and face, but the "
                "array order differs from the input.", "warn")
    open_after, _ = _boundary_stats(out_f)
    new_v = max(0, len(out_v) - in_v)
    new_f = max(0, len(out_f) - in_f)
    closed = max(0, open_before - open_after)
    seconds = time.perf_counter() - started

    if new_f and not np.array_equal(out_f[:in_f], f):
        say("Close holes: the filler reordered faces; geometry is "
            "unchanged but face order differs from the input.", "warn")

    if closed == 0:
        say(f"Close holes: nothing could be closed (non-manifold "
            f"boundary or filler failure); mesh unchanged.", "warn")
    elif open_after == 0:
        say(f"Close holes: watertight! closed {closed} boundary edge(s) "
            f"with {new_f:,} new triangles ({engine}"
            + (f", {new_v:,} new vertices" if new_v else "")
            + ").", "ok")
    else:
        say(f"Close holes: closed {closed} of {open_before} boundary "
            f"edge(s) with {new_f:,} triangles ({engine}); "
            f"{open_after} edge(s) still open.", "warn")

    return _result(out_v, out_f, in_v, in_f, closed, new_f, open_after,
                   engine, seconds)


def _restore_input_order(iv, iff, ov, of):
    """Re-index the filled mesh so the input arrays survive verbatim.

    MeshFix preserves every original vertex position (bit-exact) and
    every original face, but reindexes them - and its refinement can
    even duplicate vertices at identical positions, so no global vertex
    map exists.  Instead: match the input faces to output slots by
    exact corner positions; the remaining output faces are the patches,
    which get re-expressed in input indices by position (genuinely new
    positions become appended vertices).  The result is therefore
    ``iv + [new vertices]`` and ``iff + [patch faces]`` verbatim.

    Returns ``(vertices, faces, exact)``; ``exact`` is False when the
    remap could not be completed (then the filler output is returned
    unchanged).
    """
    def poskey(tri_vtx, tri_idx):
        return tuple(sorted(tuple(tri_vtx[i]) for i in tri_idx))

    # face position-signature -> output slots
    by_sig: dict = {}
    for pos in range(len(of)):
        by_sig.setdefault(poskey(ov, of[pos]), []).append(pos)

    slots = np.empty(len(iff), dtype=np.int64)
    for k, tri in enumerate(iff):
        bucket = by_sig.get(poskey(iv, tri))
        if not bucket:                       # input face not preserved
            return ov, of, False
        slots[k] = bucket.pop(0)

    # position -> input vertex id (first occurrence)
    pos_to_in: dict = {}
    for i, p in enumerate(iv):
        pos_to_in.setdefault(tuple(p), i)

    used_f = np.zeros(len(of), dtype=bool)
    used_f[slots] = True
    patch_of = np.flatnonzero(~used_f)
    remap: dict = {}                          # filler id -> final id
    new_pos: list = []
    patch_rows = []
    for pos in patch_of:
        row = []
        for j in of[pos]:
            if j not in remap:
                key = tuple(ov[j])
                i = pos_to_in.get(key)
                remap[j] = len(iv) + len(new_pos) if i is None else i
                if i is None:
                    new_pos.append(key)
            row.append(remap[j])
        patch_rows.append(row)
    new_v = np.vstack([iv, np.asarray(new_pos)]) if new_pos else iv
    new_f = (np.vstack([iff, np.asarray(patch_rows, dtype=np.int64)])
             if patch_rows else iff)
    return new_v, new_f, True


def _result(v, f, in_v, in_f, closed, tris, open_edges, engine,
            seconds) -> dict:
    return {
        "mode_label": "Close holes",
        "vertices": np.asarray(v, dtype=np.float64),
        "faces": np.asarray(f, dtype=np.int64),
        "vertex_count": int(len(v)),
        "face_count": int(len(f)),
        "input_vertex_count": int(in_v),
        "input_face_count": int(in_f),
        "edges_closed": int(closed),
        "triangles_added": int(tris),
        "open_edges_remaining": int(open_edges),
        "watertight": bool(open_edges == 0),
        "engine": str(engine),
        "seconds": float(seconds),
    }
