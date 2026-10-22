"""Polish: umbrella smoothing plus tangential relaxation, on arrays.

Default path (iterations derived from the vertex count and slider):

    n      = int(vertex_count * slider * 0.0002 + 2.0)
    V      = umbrella_smooth(V, n)
    repeat clamp(n, 5, 25) times:
        P0 = V
        N  = vertex_normals(P0)
        N  = 5x umbrella smoothing of N (last step normalized)
        V  = tangential_relax(P0, N)

The UI slider is 0..100, mapped to the internal 0..1 range.  The operation
only changes vertex positions; connectivity, UVs and materials are kept.

The reference implementation below is pure NumPy.  For multi-million-vertex
meshes an optional Numba backend (the sibling ``accel`` module) runs the
same passes in parallel.  Set ``POLISH_DISABLE_ACCEL=1`` to force the
NumPy path.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

#: Internal slider range is 0..1.
DEFAULT_AMOUNT = 10
#: Normal smoothing passes per Polish pass.
NORMAL_ITERATIONS = 5

LogFn = Callable[[str, str], None]


# ---------------------------------------------------------------------------
# Adjacency / kernels
# ---------------------------------------------------------------------------
def build_adjacency(faces, vertex_count: int):
    """Return (indptr, ring_neighbors, degree, repeated_rows) in CSR order."""
    n = int(vertex_count)
    f = np.asarray(faces, dtype=np.int64)
    if len(f):
        a = np.concatenate((f[:, 0], f[:, 1], f[:, 2]))
        b = np.concatenate((f[:, 1], f[:, 2], f[:, 0]))
        ok = ((a >= 0) & (a < n) & (b >= 0) & (b < n))
        a = a[ok]
        b = b[ok]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        keys = lo * np.int64(n) + hi
        undirected = np.unique(keys)
        u = undirected // n
        v = undirected - u * n
        rows = np.concatenate((u, v))
        cols = np.concatenate((v, u))
        deg = np.bincount(rows, minlength=n).astype(np.int64)
        order = np.lexsort((cols, rows))       # CSR rows, sorted neighbors
        nbr = cols[order].astype(np.int64)
    else:
        nbr = np.zeros(0, dtype=np.int64)
        deg = np.zeros(n, dtype=np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])
    rows = np.repeat(np.arange(n, dtype=np.int64), deg)
    return indptr, nbr, deg, rows


def _ring_sum(values: np.ndarray, adj) -> np.ndarray:
    _, nbr, _, rows = adj
    n = len(adj[0]) - 1
    cols = values.shape[1] if values.ndim > 1 else 1
    src = values if values.ndim > 1 else values[:, None]
    out = np.empty((n, cols), dtype=np.float64)
    for k in range(cols):
        out[:, k] = np.bincount(rows, weights=src[nbr, k], minlength=n)
    return out if values.ndim > 1 else out[:, 0]


def _umbrella_step(values: np.ndarray, adj) -> np.ndarray:
    """P_new = 0.5*P + 0.5*mean(one-ring)."""
    _, _, deg, _ = adj
    moving = deg > 0
    acc = _ring_sum(values, adj)
    out = np.array(values, dtype=np.float64, copy=True)
    out[moving] = (0.5 * values[moving]
                   + acc[moving] * (0.5 / deg[moving])[:, None])
    return out


def umbrella_smooth(values: np.ndarray, adj, iterations: int) -> np.ndarray:
    v = np.array(values, dtype=np.float64, copy=True)
    for _ in range(max(0, int(iterations))):
        v = _umbrella_step(v, adj)
    return v


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted smooth vertex normals."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    nrm = np.zeros_like(v)
    if not len(f):
        return nrm
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(nrm, f[:, k], fn)
    length = np.linalg.norm(nrm, axis=1)
    length[length == 0.0] = 1.0
    return nrm / length[:, None]


def smooth_normals(normals: np.ndarray, adj,
                   iterations: int = NORMAL_ITERATIONS) -> np.ndarray:
    """Smooth normals; the last umbrella iteration is renormalized."""
    v = np.array(normals, dtype=np.float64, copy=True)
    iterations = max(1, int(iterations))
    for it in range(iterations):
        v = _umbrella_step(v, adj)
        if it == iterations - 1:
            length = np.linalg.norm(v, axis=1)
            length[length == 0.0] = 1.0
            v = v / length[:, None]
    return v


def tangential_relax(vertices: np.ndarray, normals: np.ndarray, adj,
                     mask=None) -> np.ndarray:
    """Tangential relaxation with optional mask blend."""
    _, nbr, deg, rows = adj
    v = np.asarray(vertices, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    moving = deg > 0
    acc = _ring_sum(v, adj)
    avg = np.array(v, dtype=np.float64, copy=True)
    avg[moving] = ((v[moving] + acc[moving])
                   / (deg[moving] + 1)[:, None])

    d = avg[rows] - v[nbr]
    t = np.einsum("ij,ij->i", d, nrm[nbr])
    contrib = -t[:, None] * nrm[nbr]
    tang = np.empty_like(v)
    for k in range(3):
        tang[:, k] = np.bincount(rows, weights=contrib[:, k],
                                 minlength=len(v))

    out = np.array(v, dtype=np.float64, copy=True)
    out[moving] = avg[moving] + tang[moving] * (0.5 / deg[moving])[:, None]

    if mask is not None:
        w = np.clip(np.asarray(mask, dtype=np.float64) / 65535.0, 0.0, 1.0)
        out = v + w[:, None] * (out - v)
    return out


def polish_count(vertex_count: int, slider: float) -> int:
    """n = int(count * slider * 0.0002 + 2.0)."""
    return int(float(vertex_count) * float(slider) * 0.0002 + 2.0)


def _accelerator():
    """Optional Numba backend, unless missing or disabled."""
    if os.environ.get("POLISH_DISABLE_ACCEL", "").strip():
        return None
    try:
        from . import accel
    except Exception:
        return None
    return accel


def backend_name(mask=None) -> str:
    """Human-readable name of the path ``polish()`` will use."""
    if mask is None and _accelerator() is not None:
        return "parallel CPU (numba)"
    return "NumPy"


def polish(vertices: np.ndarray, faces: np.ndarray, slider: float,
           mask=None, preserve_peaks: bool = False,
           normal_iterations: int = NORMAL_ITERATIONS) -> np.ndarray:
    """Polish; *slider* is 0..1 (0 = no-op)."""
    v = np.asarray(vertices, dtype=np.float64).copy()
    f = np.asarray(faces, dtype=np.int64)
    if float(slider) == 0.0 or not len(v) or not len(f):
        return v

    n = polish_count(len(v), slider)
    if n <= 0:
        return v
    adj = build_adjacency(f, len(v))
    passes = min(25, max(5, n))

    # Numba path for mask-less meshes.
    if mask is None:
        accel = _accelerator()
        if accel is not None:
            return accel.polish(v, f, n, passes, preserve_peaks, adj,
                                normal_iterations=normal_iterations)

    if preserve_peaks:
        # Alternate preserve-peaks path (experimental).
        base = v.copy()
        n0 = vertex_normals(base, f)
        smoothed = umbrella_smooth(base, adj, 2 * n)
        disp = smoothed - base
        dist = np.linalg.norm(disp, axis=1)
        weight = np.zeros(len(v), dtype=np.float64)
        nz = dist > 0.0
        weight[nz] = 0.5 + 0.5 * np.einsum(
            "ij,ij->i", disp[nz] / dist[nz, None], n0[nz])
        weight = umbrella_smooth(weight[:, None], adj, n)[:, 0]
        if mask is not None:
            weight *= np.clip(
                np.asarray(mask, dtype=np.float64) / 65535.0, 0.0, 1.0)
        v = base + weight[:, None] * (smoothed - base)
    elif mask is None:
        v = umbrella_smooth(v, adj, n)
    else:
        base = v.copy()
        smoothed = umbrella_smooth(base, adj, n)
        w = np.clip(np.asarray(mask, dtype=np.float64) / 65535.0, 0.0, 1.0)
        v = base + w[:, None] * (smoothed - base)

    for _ in range(passes):
        p0 = v.copy()
        nrm = vertex_normals(p0, f)
        nrm = smooth_normals(nrm, adj, normal_iterations)
        v = tangential_relax(p0, nrm, adj, mask=mask)
    return v


# ---------------------------------------------------------------------------
# Array wrapper for the UI
# ---------------------------------------------------------------------------
def polish_arrays(vertices, faces, amount: int | float, *,
                  preserve_peaks: bool = False,
                  mask=None, log: LogFn | None = None) -> dict:
    """Polish in-memory geometry and return arrays plus statistics.

    *amount* is the UI value 0..100.  The operation keeps *faces* unchanged.
    The result is compute-only; the UI displays it and waits for Save.
    """
    def say(message: str, level: str = "info") -> None:
        if log is not None:
            log(message, level)

    started = time.perf_counter()
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if not len(v) or not len(f):
        raise ValueError("Polish needs a non-empty triangle mesh")

    slider = max(0.0, min(1.0, float(amount) / 100.0))
    n = polish_count(len(v), slider)
    passes = min(25, max(5, n)) if slider > 0.0 else 0
    backend = backend_name(mask)
    label = "Polish preserve peaks" if preserve_peaks else "Polish"
    say(f"Polish: {len(v):,} vertices, slider {amount:g} "
        f"(n={n}, passes={passes}, {backend})"
        + (", alternate preserve-peaks path" if preserve_peaks else "")
        + " ...", "info")

    out = polish(v, f, slider, mask=mask, preserve_peaks=preserve_peaks)
    displacement = np.linalg.norm(out - v, axis=1)
    moved = int(np.count_nonzero(displacement > 1e-12))
    max_move = float(displacement.max()) if len(displacement) else 0.0
    mean_move = (float(displacement[displacement > 1e-12].mean())
                 if moved else 0.0)
    seconds = time.perf_counter() - started
    say(f"Polish finished in {seconds:.1f}s: moved {moved:,}/"
        f"{len(v):,} vertices (max {max_move:.6g}, mean {mean_move:.6g}); "
        f"result kept in memory (not saved).", "ok")
    return {
        "vertices": out,
        "faces": f,
        "vertex_count": int(len(out)),
        "face_count": int(len(f)),
        "input_vertex_count": int(len(v)),
        "input_face_count": int(len(f)),
        "moved": moved,
        "max_displacement": max_move,
        "mean_displacement": mean_move,
        "iterations": int(n),
        "passes": int(passes),
        "amount": float(amount),
        "preserve_peaks": bool(preserve_peaks),
        "backend": backend,
        "operation": label,
        "mode_label": label,
        "seconds": round(seconds, 2),
    }


def polish_mesh(input_path, output_path, amount: float = DEFAULT_AMOUNT,
                *, preserve_peaks: bool = False) -> dict:
    """File-level helper: load, polish and write a mesh (CLI/testing)."""
    import trimesh

    mesh = trimesh.load(str(input_path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{input_path} did not load as a single triangle mesh")
    result = polish_arrays(mesh.vertices, mesh.faces, amount,
                           preserve_peaks=preserve_peaks)
    mesh.vertices = result["vertices"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))
    result["output"] = str(output_path)
    return result


__all__ = [
    "DEFAULT_AMOUNT", "NORMAL_ITERATIONS", "build_adjacency",
    "umbrella_smooth", "vertex_normals", "smooth_normals",
    "tangential_relax", "polish_count", "backend_name", "polish",
    "polish_arrays", "polish_mesh",
]
