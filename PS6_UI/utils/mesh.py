"""Mesh array helpers."""


def as_triangles(vertices, faces):
    """Split quad faces into triangles."""
    if faces is not None and getattr(faces, "ndim", 2) == 2 \
            and faces.shape[1] == 4:
        import numpy as np
        faces = np.concatenate(
            [faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)
    return vertices, faces
