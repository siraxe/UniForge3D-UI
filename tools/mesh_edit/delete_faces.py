"""Delete faces by boolean mask and compact the mesh, in memory.

The mask is computed by the caller (the UI reads the polygroup visibility
filter); this module only performs the actual mesh operation: drop every
face whose mask entry is False, then remove the vertices that lost their
last face.  Surviving vertices keep their exact positions and relative
order, so this composes cleanly with the in-memory edit stack.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

LogFn = Callable[[str, str], None]


def delete_faces_arrays(vertices, faces, keep_mask, *,
                        log: LogFn | None = None) -> dict:
    """Delete the faces whose ``keep_mask`` entry is False.

    ``vertices`` (n, 3) float and ``faces`` (m, 3)/(m, 4) int are returned
    without the masked-out faces alongside statistics in a result dict;
    vertices that no face references anymore are compacted away.  Nothing
    is written to disk and surviving geometry is never modified.
    """
    def say(message: str, level: str = "info") -> None:
        if log is not None:
            log(message, level)

    started = time.perf_counter()
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] not in (3, 4):
        raise ValueError("faces must be triangles or quads ((n, 3)/(n, 4))")
    mask = np.asarray(keep_mask, dtype=bool)
    if mask.shape != (len(f),):
        raise ValueError("keep_mask must have one entry per face")

    in_v, in_f = len(v), len(f)
    delete_count = int((~mask).sum())
    if in_f == 0:
        raise ValueError("the input mesh has no faces")
    if delete_count == 0:
        seconds = time.perf_counter() - started
        return _result(v, f, in_v, in_f, 0, 0, seconds)

    kept = f[mask]
    if not len(kept):
        raise ValueError("the mask would delete every face of the mesh")

    used, inverse = np.unique(kept, return_inverse=True)
    out_v = v[used]
    out_f = inverse.reshape(kept.shape)
    removed_v = in_v - len(out_v)
    seconds = time.perf_counter() - started
    say(f"Delete hidden: removed {delete_count:,} face(s) and "
        f"{removed_v:,} now-unused vertex/vertices "
        f"({in_f:,} -> {len(out_f):,} faces, "
        f"{in_v:,} -> {len(out_v):,} vertices).", "ok")

    return _result(out_v, out_f, in_v, in_f, delete_count, removed_v,
                   seconds)


def _result(v, f, in_v, in_f, deleted_f, removed_v, seconds) -> dict:
    return {
        "mode_label": "Delete hidden",
        "vertices": np.ascontiguousarray(v, dtype=np.float64),
        "faces": np.ascontiguousarray(f, dtype=np.int64),
        "vertex_count": int(len(v)),
        "face_count": int(len(f)),
        "input_vertex_count": int(in_v),
        "input_face_count": int(in_f),
        "faces_deleted": int(deleted_f),
        "vertices_removed": int(removed_v),
        "seconds": float(seconds),
    }
