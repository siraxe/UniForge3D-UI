"""Mesh array helpers."""


def as_triangles(vertices, faces):
    """Split quad faces into triangles."""
    if faces is not None and getattr(faces, "ndim", 2) == 2 \
            and faces.shape[1] == 4:
        import numpy as np
        faces = np.concatenate(
            [faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)
    return vertices, faces


def welded_smooth_normals(vertices, faces):
    """Maya-style vertex normals: face normals averaged per unique
    *position* (UV-seam duplicates share one normal) with corner-angle
    weights -- the same math the viewer's display smoothing uses."""
    import numpy as np

    v, f = as_triangles(np.asarray(vertices, dtype=np.float64),
                        np.asarray(faces))
    diag = float(np.linalg.norm(v.max(0) - v.min(0)))
    eps = max(diag * 1e-6, 1e-9)
    keys = np.round(v / eps).astype(np.int64)
    _uniq, inverse = np.unique(keys, axis=0, return_inverse=True)

    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    lens = np.linalg.norm(fn, axis=1)
    good = lens > 1e-12              # skip degenerate triangles
    fn, tg, inv = fn[good], f[good], inverse[f[good]]

    # corner-angle weights (the Maya / trimesh / Blender default)
    weights = np.empty((len(tg), 3), dtype=np.float64)
    for k in range(3):
        e0 = v[tg[:, (k + 1) % 3]] - v[tg[:, k]]
        e1 = v[tg[:, (k + 2) % 3]] - v[tg[:, k]]
        l0 = np.maximum(np.linalg.norm(e0, axis=1), 1e-30)
        l1 = np.maximum(np.linalg.norm(e1, axis=1), 1e-30)
        cosang = np.clip((e0 * e1).sum(1) / (l0 * l1), -1.0, 1.0)
        weights[:, k] = np.arccos(cosang)

    acc = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
    for k in range(3):
        np.add.at(acc, inv[:, k], fn * weights[:, k, None])
    norms = np.linalg.norm(acc, axis=1)
    ok = norms > 1e-12
    acc[ok] /= norms[ok, None]
    acc[~ok] = (0.0, 0.0, 1.0)       # degenerate corner: neutral normal
    return acc[inverse].astype(np.float32)
