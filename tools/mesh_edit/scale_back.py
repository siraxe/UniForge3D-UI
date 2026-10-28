"""Undo UniForge's automatic source normalization after a 2nd-stage upscale.

``run_stage2`` first converts the source mesh from display space to model
space, then calls ``normalize_to_cube`` before voxelizing it: the source's
AABB center is subtracted and its longest side is scaled to the canonical
``target_max_side``.  A source that already came from UniForge is already in
that cube, so the transform is effectively identity.  An imported mesh (or a
mesh from another tool) can be any size and anywhere in space, and the
Shape-1024 result comes back in the canonical cube instead of its original
size/location.

``scale_back_info`` turns the metadata returned by ``normalize_to_cube`` into
the inverse display-space transform, and ``scale_back_arrays`` applies it in
memory.  Nothing is written to disk; the UI keeps the result as a pending
edit and the normal Save button publishes it.

Display space relates to model space as ``(x, y, z) -> (x, z, -y)`` (the same
convention as ``uniforge3d.utils.mesh``).  Because ``normalize_to_cube`` works
in model space, the display-space inverse of ``(v - center) * scale`` is
``v / scale + (cx, cz, -cy)``; no axis swap is needed when both the source
and the upscaled result are display-space meshes.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

LogFn = Callable[[str, str], None]

#: Relative tolerance for deciding that UniForge actually changed the source
#: mesh's size or location.  Native UniForge meshes can differ from the exact
#: canonical bounds by a fraction of a percent after remeshing, so 1% keeps
#: those from offering a pointless Scale back action.
SCALE_TOLERANCE = 0.01
TRANSLATION_TOLERANCE = 0.01


def scale_back_info(
    normalization: dict,
    *,
    scale_tolerance: float = SCALE_TOLERANCE,
    translation_tolerance: float = TRANSLATION_TOLERANCE,
) -> dict:
    """Build the display-space inverse transform from normalization metadata.

    ``normalization`` is the dict returned by ``normalize_to_cube`` (the
    pipeline records it while seeding the 1024 upscale) with at least
    ``center_model``, ``uniform_scale`` and ``target_max_side``.  The result
    is JSON-serializable so it can travel through the worker protocol::

        {"scaled": true, "uniform_scale": 0.1666, "scale_back_factor": 6.0,
         "translation_display": [11.0, 33.0, -23.0], ...}

    ``scaled`` is False when the source was already close enough to
    UniForge's canonical size/location (scale within ``scale_tolerance`` and
    center within ``translation_tolerance`` of the source's longest side).
    """
    try:
        center = np.asarray(normalization["center_model"], dtype=np.float64)
        scale = float(normalization["uniform_scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "normalization metadata needs center_model and uniform_scale"
        ) from error
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("normalization center_model must be 3 finite values")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("normalization uniform_scale must be positive")

    bounds = normalization.get("source_bounds_model")
    if bounds is not None:
        bounds_array = np.asarray(bounds, dtype=np.float64)
        if bounds_array.shape != (2, 3):
            raise ValueError("source_bounds_model must be a 2x3 array")
        source_longest = float(np.ptp(bounds_array, axis=0).max())
    else:
        source_longest = 0.0

    target = float(normalization.get("target_max_side") or 0.0)
    if not np.isfinite(target) or target <= 0:
        # Reconstruct it from the uniform scale (canonical side = s * longest).
        if source_longest <= 0:
            raise ValueError("normalization has no usable target_max_side")
        target = scale * source_longest
    if source_longest <= 0:
        source_longest = target / scale
    if not np.isfinite(source_longest) or source_longest <= 0:
        raise ValueError("normalization has degenerate source bounds")

    # display(model) = (x, z, -y); the inverse translation is D(center).
    translation = np.array(
        [center[0], center[2], -center[1]], dtype=np.float64)
    scale_change = abs(scale - 1.0)
    translation_relative = float(
        np.linalg.norm(translation) / max(source_longest, 1e-12))
    return {
        "scaled": bool(scale_change > scale_tolerance
                       or translation_relative > translation_tolerance),
        "uniform_scale": float(scale),
        "scale_back_factor": float(1.0 / scale),
        "center_model": [float(value) for value in center],
        "translation_display": [float(value) for value in translation],
        "target_max_side": float(target),
        "source_longest_side": float(source_longest),
        "scale_change": float(scale_change),
        "translation_relative": float(translation_relative),
        "scale_tolerance": float(scale_tolerance),
        "translation_tolerance": float(translation_tolerance),
    }


def radial_scale_back_info(
    source_vertices,
    *,
    scale_tolerance: float = SCALE_TOLERANCE,
    translation_tolerance: float = TRANSLATION_TOLERANCE,
) -> dict:
    """Build the inverse of Hi3D's output normalization for a source mesh.

    Hi3D's postprocess subtracts the decoded mesh's vertex mean and divides by
    its maximum radial norm.  Mapping the result back to *source_vertices*'
    mean and maximum radius restores the displayed model's size/location::

        v_out = v_hi3d * source_max_radius + source_vertex_mean

    The returned dict matches :func:`scale_back_info`, so
    :func:`scale_back_arrays` can apply it unchanged.
    """
    v = np.asarray(source_vertices, dtype=np.float64)
    if (v.ndim != 2 or v.shape[1] != 3 or len(v) == 0
            or not np.isfinite(v).all()):
        raise ValueError("source_vertices must be a nonempty finite (N, 3) array")

    center = v.mean(axis=0)
    radius = float(np.linalg.norm(v - center, axis=1).max())
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("source mesh has degenerate radial extent")

    bounds = np.stack([v.min(axis=0), v.max(axis=0)])
    source_longest = float(np.ptp(bounds, axis=0).max())
    uniform_scale = 1.0 / radius
    # center_model is stored as the inverse display transform D^-1(center);
    # scale_back_arrays turns it back into translation_display.
    center_model = [float(center[0]), float(-center[2]), float(center[1])]
    scale_change = abs(uniform_scale - 1.0)
    translation_relative = float(
        np.linalg.norm(center) / max(source_longest, 1e-12))
    return {
        "scaled": bool(scale_change > scale_tolerance
                       or translation_relative > translation_tolerance),
        "uniform_scale": float(uniform_scale),
        "scale_back_factor": float(radius),
        "center_model": center_model,
        "translation_display": [float(value) for value in center],
        "target_max_side": float(source_longest),
        "source_longest_side": float(source_longest),
        "scale_change": float(scale_change),
        "translation_relative": float(translation_relative),
        "scale_tolerance": float(scale_tolerance),
        "translation_tolerance": float(translation_tolerance),
    }


def scale_back_arrays(
    vertices,
    faces,
    *,
    info: dict | None = None,
    center_model=None,
    uniform_scale: float | None = None,
    log: LogFn | None = None,
) -> dict:
    """Apply UniForge's inverse normalization to an in-memory display mesh.

    Pass ``info`` from :func:`scale_back_info`, or the raw ``center_model``
    and ``uniform_scale`` values.  ``vertices`` and ``faces`` are never
    modified; the scaled-back geometry is returned in a result dict shaped
    like the other ``tools.mesh_edit`` operations.  Nothing is written to
    disk.  This is a pure affine transform, so it is cheap even for
    million-triangle 1024 results.
    """
    def say(message: str, level: str = "info") -> None:
        if log is not None:
            log(message, level)

    started = time.perf_counter()
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
        raise ValueError("vertices must be a nonempty (N, 3) array")
    if not np.isfinite(v).all():
        raise ValueError("vertices contain non-finite coordinates")
    if f.ndim != 2 or f.shape[1] not in (3, 4) or len(f) == 0:
        raise ValueError("faces must be triangles (n, 3) or quads (n, 4)")

    if info is not None:
        try:
            center = np.asarray(info["center_model"], dtype=np.float64)
            scale = float(info["uniform_scale"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("scale-back info is incomplete") from error
    else:
        if center_model is None or uniform_scale is None:
            raise ValueError("pass info or center_model + uniform_scale")
        center = np.asarray(center_model, dtype=np.float64)
        scale = float(uniform_scale)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center_model must be 3 finite values")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("uniform_scale must be positive")

    translation = np.array(
        [center[0], center[2], -center[1]], dtype=np.float64)
    scaled = v / scale + translation

    say(f"Scale back: x{1.0 / scale:.6g}, offset "
        f"[{translation[0]:g}, {translation[1]:g}, {translation[2]:g}] "
        f"({len(v):,} vertices).")

    elapsed = time.perf_counter() - started
    return {
        "mode_label": "Scale back",
        "vertices": scaled,
        "faces": f,
        "vertex_count": int(len(scaled)),
        "face_count": int(len(f)),
        "face_corners": int(f.shape[1]),
        "input_vertex_count": int(len(v)),
        "input_face_count": int(len(f)),
        "uniform_scale": float(scale),
        "scale_back_factor": float(1.0 / scale),
        "translation_display": [float(value) for value in translation],
        "seconds": float(elapsed),
    }
