"""Decimate: quadric-error-metric triangle reduction, on arrays.

Collapses edges whose removal changes the surface the least, so the
overall shape survives even aggressive reductions (the quadric error
metric encodes the original surface planes; vertices merge along the
best-fit planes).

Backends, first available wins:
  * open3d        SimplifyQuadricDecimation
  * fast_simplification (via trimesh.simplify_quadric_decimation)

Array-level entry point for the UI, mirroring tools/remesh/api.py:
nothing touches disk, the caller receives plain (vertices, faces).
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

#: Never decimate below this many output triangles.
MIN_TRIS = 100

LogFn = Callable[[str, str], None]


def status() -> dict:
    """Report whether a decimation backend is importable."""
    backends = []
    try:
        import open3d
        backends.append("open3d")
    except Exception:
        pass
    try:
        import fast_simplification
        backends.append("fast_simplification")
    except Exception:
        pass
    return {
        "ready": bool(backends),
        "backend": backends[0] if backends else "",
        "error": ("" if backends else
                  "no decimation backend available "
                  "(open3d or fast_simplification required)"),
    }


def _decimate_open3d(v, f, target_tris, report):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(v, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(f, dtype=np.int32))
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    report(20, "decimate")
    out = mesh.simplify_quadric_decimation(int(target_tris))

    report(90, "cleanup")
    out.remove_duplicated_vertices()
    out.remove_duplicated_triangles()
    out.remove_degenerate_triangles()
    out.remove_unreferenced_vertices()
    ov = np.asarray(out.vertices, dtype=np.float64)
    of = np.asarray(out.triangles)
    if len(ov) == 0 or len(of) == 0:
        raise RuntimeError("open3d decimation produced an empty mesh")
    return ov, of.astype(np.int64)


def _decimate_fast_simplify(v, f, target_tris, report):
    import trimesh
    mesh = trimesh.Trimesh(vertices=np.asarray(v, dtype=np.float64),
                           faces=np.asarray(f, dtype=np.int64),
                           process=False)
    report(20, "decimate")
    out = mesh.simplify_quadric_decimation(face_count=int(target_tris))
    report(90, "cleanup")
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    if len(out.vertices) == 0 or len(out.faces) == 0:
        raise RuntimeError("fast_simplification produced an empty mesh")
    return (np.asarray(out.vertices, dtype=np.float64),
            np.asarray(out.faces, dtype=np.int64))


def decimate_arrays(vertices, faces, *, target_tris: int,
                    log: LogFn | None = None) -> dict:
    """Reduce triangle arrays to about *target_tris*, in memory.

    Returns a result dict with ``vertices``/``faces``, input/output
    counts and the backend used.  Nothing is written to disk.
    """
    def say(message: str, level: str = "info") -> None:
        if log is not None:
            log(message, level)

    def report(percent: int, stage: str = "") -> None:
        pass  # decimation is a single fast pass; no staged progress

    started = time.perf_counter()
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if not len(v) or not len(f):
        raise ValueError("the input mesh has no triangles")

    target = max(MIN_TRIS, int(target_tris))
    info = status()
    if not info["ready"]:
        raise RuntimeError(info["error"])
    backend = info["backend"]

    say(f"Decimate: {len(v):,} vertices, {len(f):,} triangles -> "
        f"target {target:,} ({backend}) ...", "info")
    if len(f) <= target:
        say(f"Decimate: already at or below {target:,} triangles; "
            "nothing to do.", "warn")
        return {
            "mode_label": "Decimate",
            "vertices": v,
            "faces": f,
            "vertex_count": int(len(v)),
            "face_count": int(len(f)),
            "input_vertex_count": int(len(v)),
            "input_face_count": int(len(f)),
            "target_tris": target,
            "backend": backend,
            "seconds": time.perf_counter() - started,
        }

    if backend == "open3d":
        out_v, out_f = _decimate_open3d(v, f, target, report)
    else:
        out_v, out_f = _decimate_fast_simplify(v, f, target, report)

    seconds = time.perf_counter() - started
    say(f"Decimate done: {len(f):,} -> {len(out_f):,} triangles "
        f"({len(out_v):,} vertices) in {seconds:.1f}s; result kept in "
        "memory (not saved).", "ok")

    return {
        "mode_label": "Decimate",
        "vertices": out_v,
        "faces": out_f,
        "vertex_count": int(len(out_v)),
        "face_count": int(len(out_f)),
        "input_vertex_count": int(len(v)),
        "input_face_count": int(len(f)),
        "target_tris": target,
        "backend": backend,
        "seconds": float(seconds),
    }
