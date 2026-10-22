"""Optional Open3D/Embree accelerator for :mod:`reproject`.

Parallel batch closest-point queries; the Embree-chosen triangle is
re-evaluated in float64, so results keep double precision.  Imported
lazily by ``reproject._closest_points``; without Open3D the VTK path is
used unchanged.
"""

from __future__ import annotations

import numpy as np

#: Points per Embree query.  Bounds the temporary float32 result tensors;
#: the query itself is so fast that chunking costs almost nothing.
QUERY_CHUNK = 1_000_000

#: Candidates per float64 refinement batch.  Each refinement allocates a
#: handful of temporaries, so keep the chunk moderate.
REFINE_CHUNK = 250_000

_MODULE = None
_CHECKED = False


def _module():
    """Return the open3d module, or None when it cannot be imported."""
    global _MODULE, _CHECKED
    if _MODULE is not None:
        return _MODULE
    if _CHECKED:
        return None
    _CHECKED = True
    try:
        import open3d as o3d
    except Exception:
        return None
    _MODULE = o3d
    return _MODULE


def available() -> bool:
    """True when Open3D can be imported and used for queries."""
    return _module() is not None


def _closest_on_triangles(points, a, b, c):
    """Closest point on each triangle (p, a, b, c) in float64.

    Vectorized version of the standard region-based triangle query
    (Ericson, "Real-Time Collision Detection").  Zero-length denominators
    are clamped so even a degenerate candidate cannot produce NaNs.
    """
    ab = b - a
    ac = c - a
    ap = points - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)

    bp = points - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)

    cp = points - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    out = np.empty_like(points)
    done = np.zeros(len(points), dtype=bool)

    # Vertex region A.
    m = (d1 <= 0.0) & (d2 <= 0.0)
    out[m] = a[m]
    done |= m

    # Vertex region B.
    m = ~done & (d3 >= 0.0) & (d4 <= d3)
    out[m] = b[m]
    done |= m

    # Edge AB.
    m = ~done & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    den = d1[m] - d3[m]
    den[den == 0.0] = 1.0
    t = d1[m] / den
    out[m] = a[m] + ab[m] * t[:, None]
    done |= m

    # Vertex region C.
    m = ~done & (d6 >= 0.0) & (d5 <= d6)
    out[m] = c[m]
    done |= m

    # Edge AC.
    m = ~done & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    den = d2[m] - d6[m]
    den[den == 0.0] = 1.0
    t = d2[m] / den
    out[m] = a[m] + ac[m] * t[:, None]
    done |= m

    # Edge BC.
    m = (~done & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0))
    den = (d4[m] - d3[m]) + (d5[m] - d6[m])
    den[den == 0.0] = 1.0
    t = (d4[m] - d3[m]) / den
    out[m] = b[m] + (c[m] - b[m]) * t[:, None]
    done |= m

    # Interior region.
    m = ~done
    den = va[m] + vb[m] + vc[m]
    den[den == 0.0] = 1.0
    v = vb[m] / den
    w = vc[m] / den
    out[m] = a[m] + ab[m] * v[:, None] + ac[m] * w[:, None]
    return out


def _nondegenerate_faces(v, f):
    """Triangles with distinct corners and positive area."""
    a = v[f[:, 0]]
    b = v[f[:, 1]]
    c = v[f[:, 2]]
    cross = np.cross(b - a, c - a)
    area2 = np.einsum("ij,ij->i", cross, cross)
    distinct = ((f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2])
                & (f[:, 2] != f[:, 0]))
    return distinct & np.isfinite(area2) & (area2 > 0.0)


def closest_points(vertices, faces, points):
    """Exact closest points on a triangle mesh for every query point.

    Returns ``(closest, distance, triangle_index)`` with the same meaning
    and precision as ``reproject._closest_points_vtk``.  Embree picks the
    candidate triangle in parallel; the float64 refinement below makes the
    final result independent of Open3D's float32 storage.
    """
    o3d = _module()
    if o3d is None:  # pragma: no cover - guarded by available()
        raise ImportError("open3d is not available")

    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int64)
    p = np.ascontiguousarray(points, dtype=np.float64)
    if not len(v) or not len(f):
        raise ValueError("source mesh has no triangles")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("source faces must be triangles")
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError("source faces reference missing vertices")
    if not len(p):
        return (np.zeros((0, 3), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.int64))

    valid = _nondegenerate_faces(v, f)
    if valid.all():
        query_faces = f
        remap = None
    else:
        remap = np.flatnonzero(valid)
        if not len(remap):
            raise ValueError("source mesh has no non-degenerate triangles")
        query_faces = f[remap]

    # Center before the float32 cast.
    center = v.mean(axis=0)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.ascontiguousarray(v - center, dtype=np.float32)),
        o3d.core.Tensor(np.ascontiguousarray(query_faces,
                                            dtype=np.uint32)))

    count = len(p)
    closest = np.empty((count, 3), dtype=np.float64)
    distance = np.empty(count, dtype=np.float64)
    triangle = np.empty(count, dtype=np.int64)

    for start in range(0, count, QUERY_CHUNK):
        end = min(count, start + QUERY_CHUNK)
        query = np.ascontiguousarray(p[start:end] - center,
                                     dtype=np.float32)
        answer = scene.compute_closest_points(o3d.core.Tensor(query))
        tri = answer["primitive_ids"].numpy().astype(np.int64)
        if len(tri) and (tri.min() < 0 or tri.max() >= len(query_faces)):
            raise RuntimeError("Embree returned an invalid triangle index")
        if remap is not None:
            tri = remap[tri]

        # Float64 refinement of candidate triangles.
        for rs in range(start, end, REFINE_CHUNK):
            re = min(end, rs + REFINE_CHUNK)
            local = slice(rs - start, re - start)
            ft = f[tri[local]]
            cp = _closest_on_triangles(
                p[rs:re], v[ft[:, 0]], v[ft[:, 1]], v[ft[:, 2]])
            closest[rs:re] = cp
            distance[rs:re] = np.linalg.norm(cp - p[rs:re], axis=1)
            triangle[rs:re] = tri[local]

    return closest, distance, triangle
