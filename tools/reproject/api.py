"""Detail reprojection from an unrepaired mesh onto a repaired target.

Four modes, all driven by the same closest-point query:

* ``project``  - snap vertices within Dist onto the donor surface.
* ``normal``   - move only along the target normal, no tangential slide.
* ``highpass`` - copy donor high-pass detail (donor - smoothed donor)
  sampled at the closest point; the target keeps its own silhouette.
* ``higher_subdiv`` - midpoint-subdivide the target, then snap every
  vertex using a 1.1 * max gap tolerance.

``reproject_meshes`` is compute-only and can consume unsaved in-memory
target arrays, so it stacks on Polish/other edits.  ``save_reprojected`` is
the explicit manual Save used by PS6_UI to write ``<model>_edit.glb``.

The closest-point query uses VTK's static cell locator by default.  When
Open3D is installed, the sibling ``accel`` module answers the same query
with an Embree BVH in parallel.  Set ``REPROJECT_DISABLE_ACCEL=1`` to
force the VTK path.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import trimesh

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
except ImportError:
    vtk = None
    numpy_to_vtk = None
    numpy_to_vtkIdTypeArray = None

try:
    from . import accel
except Exception:
    accel = None


DEFAULT_DIST = 0.1
MIN_TOLERANCE = 0.001
DEFAULT_DETAIL_SMOOTH = 5
DEFAULT_SUBDIV_LEVELS = 1

MODE_PROJECT = "project"
MODE_NORMAL = "normal"
MODE_HIGHPASS = "highpass"
MODE_HIGHER_SUBDIV = "higher_subdiv"

MODE_LABELS = {
    MODE_PROJECT: "ProjectAll snap",
    MODE_NORMAL: "Normal only",
    MODE_HIGHPASS: "High-pass detail",
    MODE_HIGHER_SUBDIV: "Higher subdiv",
}
MODES = tuple(MODE_LABELS)

LogFn = Callable[[str, str], None]

_CLOSEST_BACKEND_USED: str | None = None


def max_bbox_extent(vertex_arrays: Sequence[np.ndarray]) -> float:
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for vertices in vertex_arrays:
        v = np.asarray(vertices, dtype=np.float64)
        if not len(v):
            continue
        lo = np.minimum(lo, v.min(axis=0))
        hi = np.maximum(hi, v.max(axis=0))
    if not np.isfinite(lo).all():
        return 2.0
    return float(np.max(hi - lo))


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = np.zeros_like(v)
    if not len(f):
        return n
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    rows = np.concatenate((f[:, 0], f[:, 1], f[:, 2]))
    for k in range(3):
        n[:, k] = np.bincount(rows, weights=np.tile(fn[:, k], 3),
                              minlength=len(v))
    length = np.linalg.norm(n, axis=1)
    length[length == 0.0] = 1.0
    return n / length[:, None]


def laplacian_smooth(vertices: np.ndarray, faces: np.ndarray,
                     iterations: int) -> np.ndarray:
    """Umbrella smoothing (PA Blur pre-pass)."""
    v = np.asarray(vertices, dtype=np.float64).copy()
    f = np.asarray(faces, dtype=np.int64)
    if iterations <= 0 or not len(f):
        return v
    edges = np.vstack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    edges = np.vstack((edges, edges[:, ::-1]))
    rows = np.ascontiguousarray(edges[:, 0])
    cols = np.ascontiguousarray(edges[:, 1])
    degree = np.bincount(rows, minlength=len(v)).astype(np.float64)
    degree[degree == 0.0] = 1.0
    count = len(v)
    for _ in range(int(iterations)):
        acc = np.empty_like(v)
        for k in range(3):
            acc[:, k] = np.bincount(rows, weights=v[cols, k],
                                    minlength=count)
        v = 0.5 * v + 0.5 * (acc / degree[:, None])
    return v


def subdivide_midpoint(vertices: np.ndarray, faces: np.ndarray,
                       levels: int = 1):
    v = np.asarray(vertices, dtype=np.float64).copy()
    f = np.asarray(faces, dtype=np.int64).copy()
    for _ in range(max(0, int(levels))):
        if not len(f):
            break
        count = len(v)
        edges = np.vstack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
        lo = np.minimum(edges[:, 0], edges[:, 1]).astype(np.int64)
        hi = np.maximum(edges[:, 0], edges[:, 1]).astype(np.int64)
        keys = lo * np.int64(count) + hi
        uniq, inverse = np.unique(keys, return_inverse=True)
        inverse = inverse.reshape(-1)
        ulo = uniq // count
        uhi = uniq - ulo * count
        mid = 0.5 * (v[ulo] + v[uhi])
        m = count + inverse
        ab, bc, ca = m[0::3], m[1::3], m[2::3]
        a, b, c = f[:, 0], f[:, 1], f[:, 2]
        f = np.vstack((
            np.stack((a, ab, ca), axis=1),
            np.stack((ab, b, bc), axis=1),
            np.stack((ca, bc, c), axis=1),
            np.stack((ab, bc, ca), axis=1),
        ))
        v = np.vstack((v, mid))
    return v, f


def _sample_at_closest(closest: np.ndarray, triangle: np.ndarray,
                       source_vertices: np.ndarray, source_faces: np.ndarray,
                       data: np.ndarray) -> np.ndarray:
    tri = np.asarray(source_faces, dtype=np.int64)[
        np.asarray(triangle, dtype=np.int64)]
    a = source_vertices[tri[:, 0]]
    b = source_vertices[tri[:, 1]]
    c = source_vertices[tri[:, 2]]
    p = np.asarray(closest, dtype=np.float64)
    v0, v1, v2 = b - a, c - a, p - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01
    good = np.abs(denom) > 1e-20
    safe = np.where(good, denom, 1.0)
    wb = np.where(good, (d11 * d20 - d01 * d21) / safe, 0.0)
    wc = np.where(good, (d00 * d21 - d01 * d20) / safe, 0.0)
    wa = 1.0 - wb - wc
    weights = np.clip(np.stack((wa, wb, wc), axis=1), 0.0, None)
    total = weights.sum(axis=1)
    total[total == 0.0] = 1.0
    weights /= total[:, None]
    if data.ndim == 1:
        return (weights[:, 0] * data[tri[:, 0]] +
                weights[:, 1] * data[tri[:, 1]] +
                weights[:, 2] * data[tri[:, 2]])
    return (weights[:, 0, None] * data[tri[:, 0]] +
            weights[:, 1, None] * data[tri[:, 1]] +
            weights[:, 2, None] * data[tri[:, 2]])


def _combined_targets(source_meshes: Sequence, combine_sources: bool):
    meshes = list(source_meshes)
    if not combine_sources or len(meshes) <= 1:
        return meshes
    offsets = np.cumsum([0] + [len(m.vertices) for m in meshes[:-1]])
    verts = np.concatenate(
        [np.asarray(m.vertices, dtype=np.float64) for m in meshes], axis=0)
    faces = np.concatenate(
        [np.asarray(m.faces, dtype=np.int64) + int(o)
         for m, o in zip(meshes, offsets)], axis=0)
    return [trimesh.Trimesh(vertices=verts, faces=faces,
                            process=False, validate=False)]


def _shell_vector(projection_shell: float, shell_axes,
                  scale: float) -> np.ndarray:
    axis = np.asarray(shell_axes, dtype=np.float64)
    return float(projection_shell) * 0.5 * float(scale) * axis


def _query_closest(vertices: np.ndarray, faces: np.ndarray, source,
                   shell: np.ndarray, *, recompute_normals: bool,
                   original_normals: np.ndarray | None):
    """Closest points on *source*, offsetting the query by the shell."""
    if np.any(shell != 0.0):
        normals = (vertex_normals(vertices, faces) if recompute_normals
                   else original_normals)
        query = vertices + normals * shell
    else:
        query = vertices
    return _closest_points(source, query)


def _weight_mask(weights) -> np.ndarray | None:
    if weights is None:
        return None
    return np.clip(np.asarray(weights, dtype=np.float64) / 65535.0, 0.0, 1.0)


def _accelerator():
    if os.environ.get("REPROJECT_DISABLE_ACCEL", "").strip():
        return None
    return accel if (accel is not None and accel.available()) else None


def closest_backend() -> str:
    return "Open3D (embree)" if _accelerator() is not None else "VTK"


def _closest_points_vtk(src_vertices: np.ndarray, src_faces: np.ndarray,
                        points: np.ndarray):
    if vtk is None:
        raise ImportError("vtk is not installed")
    vertices = np.ascontiguousarray(src_vertices, dtype=np.float64)
    faces = np.ascontiguousarray(src_faces, dtype=np.int64)
    queries = np.ascontiguousarray(points, dtype=np.float64)
    if not len(vertices) or not len(faces):
        raise ValueError("source mesh has no triangles")

    polydata = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(vertices, deep=True))
    polydata.SetPoints(vtk_points)

    cells = vtk.vtkCellArray()
    offsets = np.arange(0, faces.size + 1, 3, dtype=np.int64)
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(faces.ravel(), deep=True))
    polydata.SetPolys(cells)

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()

    closest = np.empty_like(queries)
    distance = np.empty(len(queries), dtype=np.float64)
    triangle = np.empty(len(queries), dtype=np.int64)
    out_point = [0.0, 0.0, 0.0]
    ref = getattr(vtk, "reference", None) or getattr(vtk, "mutable")
    out_cell = ref(-1)
    out_sub = ref(-1)
    out_d2 = ref(-1.0)
    find = locator.FindClosestPoint
    for i in range(len(queries)):
        find(queries[i], out_point, out_cell, out_sub, out_d2)
        closest[i, 0] = out_point[0]
        closest[i, 1] = out_point[1]
        closest[i, 2] = out_point[2]
        distance[i] = out_d2
        triangle[i] = out_cell
    np.sqrt(distance, out=distance)
    return closest, distance, triangle


def _closest_points(source, points: np.ndarray):
    global _CLOSEST_BACKEND_USED
    accelerator = _accelerator()
    if accelerator is not None:
        try:
            result = accelerator.closest_points(
                source.vertices, source.faces, points)
            _CLOSEST_BACKEND_USED = "Open3D (embree)"
            return result
        except Exception:
            pass
    try:
        result = _closest_points_vtk(source.vertices, source.faces, points)
        _CLOSEST_BACKEND_USED = "VTK"
        return result
    except ImportError:
        result = trimesh.proximity.closest_point(source, points)
        _CLOSEST_BACKEND_USED = "trimesh"
        return result


def project_all(active_vertices: np.ndarray, active_faces: np.ndarray,
                source_meshes: Sequence, *, dist: float,
                projection_shell: float = 0.0,
                shell_axes=(True, True, True), pa_blur: int = 0,
                strength: float = 1.0, weights=None,
                scale: float | None = None,
                combine_sources: bool = False,
                recompute_normals: bool = True,
                tolerance: float | None = None) -> np.ndarray:
    v = np.asarray(active_vertices, dtype=np.float64).copy()
    f = np.asarray(active_faces, dtype=np.int64)
    if scale is None:
        scale = max_bbox_extent(
            [v] + [np.asarray(m.vertices) for m in source_meshes])

    if tolerance is None:
        tolerance = max(MIN_TOLERANCE, float(dist) * 0.5 * float(scale))
    else:
        tolerance = max(0.0, float(tolerance))
    shell = _shell_vector(projection_shell, shell_axes, scale)

    if pa_blur > 0:
        v = laplacian_smooth(v, f, int(pa_blur))

    w_mask = _weight_mask(weights)
    targets = _combined_targets(source_meshes, combine_sources)

    shell_active = bool(np.any(shell != 0.0))
    original_normals = None
    if shell_active and not recompute_normals:
        original_normals = vertex_normals(np.asarray(active_vertices), f)

    for source in targets:
        if not len(source.vertices) or not len(source.faces):
            continue
        closest, distance, _ = _query_closest(
            v, f, source, shell, recompute_normals=recompute_normals,
            original_normals=original_normals)
        hit = distance <= tolerance
        if not np.any(hit):
            continue
        delta = closest - v
        weight = np.full(len(v), float(strength))
        if w_mask is not None:
            weight *= w_mask
            hit &= w_mask > 0.0
        v[hit] += delta[hit] * weight[hit, None]

    return v


def _project_normals(vertices: np.ndarray, faces: np.ndarray,
                     source_meshes: Sequence, *, tolerance: float,
                     shell: np.ndarray, strength: float, w_mask,
                     recompute_normals: bool = True) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64).copy()
    f = np.asarray(faces, dtype=np.int64)
    targets = list(source_meshes)
    normals = vertex_normals(v, f)

    for i, source in enumerate(targets):
        if not len(source.vertices) or not len(source.faces):
            continue
        if i and recompute_normals:
            normals = vertex_normals(v, f)
        closest, distance, _ = _query_closest(
            v, f, source, shell, recompute_normals=recompute_normals,
            original_normals=normals)
        hit = distance <= tolerance
        if w_mask is not None:
            hit &= w_mask > 0.0
        if not np.any(hit):
            continue
        delta = closest - v
        along = np.einsum("ij,ij->i", delta, normals)
        weight = np.full(len(v), float(strength))
        if w_mask is not None:
            weight *= w_mask
        v[hit] += (along * weight)[hit, None] * normals[hit]
    return v


def _transfer_highpass(vertices: np.ndarray, faces: np.ndarray,
                       source_meshes: Sequence, *, tolerance: float,
                       shell: np.ndarray, strength: float, w_mask,
                       detail_iterations: int = DEFAULT_DETAIL_SMOOTH,
                       recompute_normals: bool = True) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64).copy()
    f = np.asarray(faces, dtype=np.int64)
    original_normals = (vertex_normals(v, f)
                        if np.any(shell != 0.0) and not recompute_normals
                        else None)

    for source in source_meshes:
        src_v = np.asarray(source.vertices, dtype=np.float64)
        src_f = np.asarray(source.faces, dtype=np.int64)
        if not len(src_v) or not len(src_f):
            continue
        smoothed = laplacian_smooth(src_v, src_f, int(detail_iterations))
        detail = src_v - smoothed
        closest, distance, triangle = _query_closest(
            v, f, source, shell, recompute_normals=recompute_normals,
            original_normals=original_normals)
        hit = distance <= tolerance
        if w_mask is not None:
            hit &= w_mask > 0.0
        if not np.any(hit):
            continue
        sampled = _sample_at_closest(closest, triangle, src_v, src_f, detail)
        weight = np.full(len(v), float(strength))
        if w_mask is not None:
            weight *= w_mask
        v[hit] += sampled[hit] * weight[hit, None]
    return v


def _snap_all_to_surface(vertices: np.ndarray, faces: np.ndarray,
                         source_meshes: Sequence, *,
                         distance_scale: float = 1.0,
                         shell: np.ndarray, strength: float, w_mask,
                         recompute_normals: bool = True):
    v = np.asarray(vertices, dtype=np.float64).copy()
    f = np.asarray(faces, dtype=np.int64)
    last_tolerance = MIN_TOLERANCE
    original_normals = (vertex_normals(v, f)
                        if np.any(shell != 0.0) and not recompute_normals
                        else None)

    for source in source_meshes:
        if not len(source.vertices) or not len(source.faces):
            continue
        closest, distance, _ = _query_closest(
            v, f, source, shell, recompute_normals=recompute_normals,
            original_normals=original_normals)
        if not len(distance):
            continue
        gap = float(np.max(distance))
        last_tolerance = max(MIN_TOLERANCE,
                             1.1 * gap * float(distance_scale))
        hit = distance <= last_tolerance
        if w_mask is not None:
            hit &= w_mask > 0.0
        if not np.any(hit):
            continue
        weight = np.full(len(v), float(strength))
        if w_mask is not None:
            weight *= w_mask
        v[hit] += (closest - v)[hit] * weight[hit, None]
    return v, last_tolerance


def reproject_geometry(active_vertices: np.ndarray, active_faces: np.ndarray,
                       source_meshes: Sequence, *,
                       mode: str = MODE_PROJECT, dist: float = DEFAULT_DIST,
                       projection_shell: float = 0.0,
                       shell_axes=(True, True, True), pa_blur: int = 0,
                       strength: float = 1.0, weights=None,
                       scale: float | None = None,
                       combine_sources: bool = False,
                       recompute_normals: bool = True,
                       detail_iterations: int = DEFAULT_DETAIL_SMOOTH,
                       subdiv_levels: int = DEFAULT_SUBDIV_LEVELS,
                       distance_scale: float = 1.0):
    """Run one reprojection mode on arrays.

    Returns ``(vertices, faces, reference_vertices, info)``.  *faces* differs
    from the input only in higher-subdiv mode; *reference_vertices* are the
    pre-projection positions of the returned topology, for displacement
    statistics.  Nothing is written to disk.
    """
    mode = str(mode)
    if mode not in MODE_LABELS:
        raise ValueError(f"unknown reprojection mode: {mode!r}")

    v = np.asarray(active_vertices, dtype=np.float64).copy()
    f = np.asarray(active_faces, dtype=np.int64)
    source_meshes = list(source_meshes)
    if scale is None:
        scale = max_bbox_extent(
            [v] + [np.asarray(m.vertices) for m in source_meshes])

    tolerance = max(MIN_TOLERANCE, float(dist) * 0.5 * float(scale))
    shell = _shell_vector(projection_shell, shell_axes, scale)
    w_mask = _weight_mask(weights)
    targets = _combined_targets(source_meshes, combine_sources)

    info = {
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "tolerance": tolerance,
        "scale": float(scale),
        "dist": float(dist),
        "strength": float(strength),
        "pa_blur": int(pa_blur),
        "projection_shell": float(projection_shell),
        "detail_iterations": 0,
        "subdiv_levels": 0,
    }

    if mode == MODE_PROJECT:
        out = project_all(
            v, f, targets, dist=dist, projection_shell=projection_shell,
            shell_axes=shell_axes, pa_blur=pa_blur, strength=strength,
            weights=weights, scale=scale, recompute_normals=recompute_normals)
        return out, f, v, info

    base = (laplacian_smooth(v, f, int(pa_blur)) if pa_blur > 0
            else v.copy())

    if mode == MODE_HIGHER_SUBDIV:
        levels = max(0, int(subdiv_levels))
        sub_v, sub_f = subdivide_midpoint(base, f, levels)
        out, snap_tolerance = _snap_all_to_surface(
            sub_v, sub_f, targets, distance_scale=distance_scale,
            shell=shell, strength=strength, w_mask=w_mask,
            recompute_normals=recompute_normals)
        info["tolerance"] = snap_tolerance
        info["subdiv_levels"] = levels
        return out, sub_f, sub_v, info

    if mode == MODE_NORMAL:
        out = _project_normals(
            base, f, targets, tolerance=tolerance, shell=shell,
            strength=strength, w_mask=w_mask,
            recompute_normals=recompute_normals)
    elif mode == MODE_HIGHPASS:
        iterations = max(1, int(detail_iterations))
        out = _transfer_highpass(
            base, f, targets, tolerance=tolerance, shell=shell,
            strength=strength, w_mask=w_mask,
            detail_iterations=iterations,
            recompute_normals=recompute_normals)
        info["detail_iterations"] = iterations
    else:
        raise ValueError(f"unknown reprojection mode: {mode!r}")
    return out, f, v, info


def _load_mesh(path: Path):
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path.name} did not load as a single triangle mesh")
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"{path.name} contains no triangles")
    return mesh


def reproject_meshes(target_path: Path, source_path: Path, *,
                     mode: str = MODE_PROJECT, dist: float = DEFAULT_DIST,
                     projection_shell: float = 0.0, pa_blur: int = 0,
                     strength: float = 1.0,
                     detail_iterations: int = DEFAULT_DETAIL_SMOOTH,
                     subdiv_levels: int = DEFAULT_SUBDIV_LEVELS,
                     target_vertices: np.ndarray | None = None,
                     target_faces: np.ndarray | None = None,
                     log: LogFn | None = None) -> dict:
    """Reproject *source_path* detail onto *target_path*, in memory.

    The target keeps its own connectivity, UVs and material (except in
    higher-subdiv mode, which rebuilds the target topology).  No file is
    written: the result arrays plus statistics are returned and the UI keeps
    them until the user presses Save.

    ``target_vertices``/``target_faces`` optionally supply the current
    in-memory geometry (e.g. an unsaved Polish result on top of a CelloCut
    mesh).  The on-disk target is then only used as the Save/material
    reference.

    The returned dict uses ``"vertices"``/``"faces"`` for the new arrays and
    ``vertex_count``/``face_count`` for their sizes.
    """
    target_path = Path(target_path)
    source_path = Path(source_path)

    def say(message: str, level: str = "info") -> None:
        if log is not None:
            log(message, level)

    started = time.perf_counter()
    if target_vertices is None or target_faces is None:
        say(f"Reproject: loading {target_path.name} ...", "info")
        target = _load_mesh(target_path)
        original = np.asarray(target.vertices, dtype=np.float64).copy()
        faces = np.asarray(target.faces, dtype=np.int64).copy()
    else:
        say(f"Reproject: using the current in-memory {target_path.name} "
            "geometry ...", "info")
        original = np.asarray(target_vertices, dtype=np.float64).copy()
        faces = np.asarray(target_faces, dtype=np.int64).copy()
        if not len(original) or not len(faces):
            raise ValueError("the in-memory target has no triangles")
    say(f"Reproject: loading source {source_path.name} ...", "info")
    source = _load_mesh(source_path)
    scale = max_bbox_extent([original, source.vertices])

    mode_label = MODE_LABELS.get(str(mode), str(mode))
    backend = closest_backend()
    say(f"Reproject ({mode_label}): {len(original):,} vertices, source "
        f"{len(source.faces):,} triangles, strength {float(strength):g}, "
        f"blur {int(pa_blur)}, shell {float(projection_shell):g}"
        + (f", Dist {float(dist):g}" if mode != MODE_HIGHER_SUBDIV
           else f", subdiv x{int(subdiv_levels)}")
        + f", closest {backend} ...", "info")

    global _CLOSEST_BACKEND_USED
    _CLOSEST_BACKEND_USED = None
    moved_vertices, moved_faces, reference, info = reproject_geometry(
        original, faces, [source], mode=mode, dist=dist,
        projection_shell=projection_shell, pa_blur=pa_blur,
        strength=strength, scale=scale,
        detail_iterations=detail_iterations, subdiv_levels=subdiv_levels)

    displacement = np.linalg.norm(moved_vertices - reference, axis=1)
    moved = displacement > 1e-9
    count = int(np.count_nonzero(moved))
    max_move = float(displacement.max()) if len(displacement) else 0.0
    mean_move = float(displacement[moved].mean()) if count else 0.0
    seconds = time.perf_counter() - started

    tolerance = float(info.get("tolerance", 0.0))
    if count:
        say(f"Reproject ({mode_label}): moved {count:,}/"
            f"{len(displacement):,} vertices (max {max_move:.6g}, mean "
            f"{mean_move:.6g}, tolerance {tolerance:.6g}) in "
            f"{seconds:.1f}s (closest {_CLOSEST_BACKEND_USED or backend}); "
            "result kept in memory (not saved).", "ok")
    elif mode != MODE_HIGHER_SUBDIV:
        say(f"Reproject ({mode_label}): no vertex was within tolerance "
            f"{tolerance:.6g}; nothing changed. Check that the two meshes "
            "share the same scale and position.", "warn")
    else:
        say(f"Reproject ({mode_label}): the snap did not move any vertex.",
            "warn")

    result = dict(info)
    result.update({
        "target_path": str(target_path),
        "source_path": str(source_path),
        "vertices": moved_vertices,
        "faces": moved_faces,
        "vertex_count": int(len(moved_vertices)),
        "face_count": int(len(moved_faces)),
        "input_vertex_count": int(len(original)),
        "input_face_count": int(len(faces)),
        "moved": count,
        "max_displacement": max_move,
        "mean_displacement": mean_move,
        "source_vertices": int(len(source.vertices)),
        "source_faces": int(len(source.faces)),
        "closest_backend": _CLOSEST_BACKEND_USED or backend,
        "seconds": round(seconds, 2),
    })
    return result


def save_reprojected(source_path: Path, output_path: Path,
                     vertices: np.ndarray, faces: np.ndarray | None = None,
                     *, log: LogFn | None = None) -> dict:
    source_path = Path(source_path)
    output_path = Path(output_path)

    def say(message: str, level: str = "info") -> None:
        if log is not None:
            log(message, level)

    started = time.perf_counter()
    target = _load_mesh(source_path)
    v = np.asarray(vertices, dtype=np.float64)
    f = (np.asarray(target.faces, dtype=np.int64) if faces is None
         else np.asarray(faces, dtype=np.int64))

    same_topology = (
        len(v) == len(target.vertices)
        and f.shape == target.faces.shape
        and np.array_equal(f, np.asarray(target.faces, dtype=np.int64)))
    if same_topology:
        mesh = target
        mesh.vertices = v
    else:
        quad_input = f.ndim == 2 and f.shape[1] == 4
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False,
                               validate=False)
        if quad_input:
            # GLB/glTF is triangles-only; quads are split here on disk.
            say("Save: quad topology triangulated (GLB cannot store "
                "quads).", "warn")

    say(f"Save: writing {output_path.name} ...", "info")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))
    seconds = time.perf_counter() - started
    stats = {
        "output": str(output_path),
        "vertex_count": int(len(v)),
        "face_count": int(len(f)),
        "bytes": int(output_path.stat().st_size),
        "seconds": round(seconds, 2),
        "source": str(source_path),
        "preserved_material": bool(same_topology),
    }
    say(f"Save: wrote {output_path.name} "
        f"({stats['vertex_count']:,} vertices, {stats['face_count']:,} "
        f"faces) in {seconds:.1f}s.", "ok")
    return stats
