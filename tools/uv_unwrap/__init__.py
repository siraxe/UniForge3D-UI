"""UV unwrapping tool: an xatlas chart atlas for the displayed mesh.

Used by the UltraTex panel's "Unwrap UVs" button. The unwrap cuts UV seams
and packs the charts into one atlas of ``resolution`` x ``resolution``
texels with ``padding`` texels of gutter, which is the layout the later
UltraTex texture pass bakes into.

Seams duplicate vertices (each chart carries its own corner copies), so the
vertex count grows even though the surface is unchanged - the output
topology differs from the input, like Decimate. Run the unwrap as the last
geometry step before texturing: a later topology-changing edit (Decimate,
Remesh, MirrorAndMerge, ...) invalidates the UVs.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

LogFn = Callable[[str, str], None]
CancelFn = Callable[[], bool]


def unwrap_arrays(vertices, faces, *, resolution: int = 2048,
                  padding: int = 4, texels_per_unit: float = 0.0,
                  log: LogFn | None = None,
                  cancel: CancelFn | None = None) -> dict:
    """Unwrap one triangle mesh (n,3)/(m,3) into a packed UV atlas.

    Returns a result dict with the duplicated vertices/faces (xatlas chart
    copies) and per-vertex ``uvs`` in [0, 1] (y already in glTF convention,
    origin top-left). The surface is unchanged; only seams split vertices.
    """
    started = time.perf_counter()

    def say(message: str, level: str = "info") -> None:
        if log is not None:
            log(message, level)

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (n, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be triangles (m, 3), got {f.shape}")
    if len(f) == 0:
        raise ValueError("the mesh has no faces")
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError("faces reference vertices outside the mesh")
    if cancel is not None and cancel():
        raise CancelledError()

    say(f"UV unwrap: {len(v):,} vertices, {len(f):,} triangles; atlas "
        f"{resolution}px, padding {padding} texels ...")

    import xatlas

    positions = np.ascontiguousarray(v, dtype=np.float32)
    indices = np.ascontiguousarray(f, dtype=np.uint32)

    atlas = xatlas.Atlas()
    atlas.add_mesh(positions, indices)
    pack_options = xatlas.PackOptions()
    pack_options.resolution = int(resolution)
    pack_options.padding = int(padding)
    if texels_per_unit > 0:
        pack_options.texels_per_unit = float(texels_per_unit)
    atlas.generate(pack_options=pack_options)
    if cancel is not None and cancel():
        raise CancelledError()

    vertex_map, new_faces, uvs = atlas.get_mesh(0)
    vertex_map = np.asarray(vertex_map, dtype=np.int64)
    new_faces = np.asarray(new_faces, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float32)

    new_vertices = v[vertex_map]
    chart_count = int(atlas.get_mesh_chart_count(0))
    utilization = float(atlas.get_utilization(0)) * 100.0

    if new_faces.min() < 0 or new_faces.max() >= len(new_vertices):
        raise RuntimeError("xatlas returned faces outside the new mesh")
    uvs = np.clip(uvs, 0.0, 1.0)

    seconds = time.perf_counter() - started
    stats = {
        "mode_label": "Unwrap UVs",
        "input_vertex_count": int(len(v)),
        "input_face_count": int(len(f)),
        "vertex_count": int(len(new_vertices)),
        "face_count": int(len(new_faces)),
        "seam_vertices_added": int(len(new_vertices) - len(v)),
        "chart_count": chart_count,
        "atlas_resolution": int(resolution),
        "padding": int(padding),
        "atlas_utilization_percent": round(utilization, 1),
        "seconds": round(seconds, 2),
    }
    say(f"UV unwrap finished: {chart_count} charts, atlas utilization "
        f"{utilization:.1f}% ({len(v):,} -> {len(new_vertices):,} vertices "
        f"after seam splits) in {seconds:.1f}s.", "ok")
    return {
        "vertices": new_vertices,
        "faces": new_faces,
        "uvs": uvs,
        **stats,
    }


class CancelledError(RuntimeError):
    """Raised when the controller asked the job to stop."""
