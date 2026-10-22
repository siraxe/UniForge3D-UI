"""World-space symmetry for the sculpting brushes.

The Edit-mesh panel's X / Y / Z checkboxes enable mirror planes that are
AXIS-ALIGNED IN WORLD SPACE and pass through the actor's world origin
(the translation part of its matrix).  While a plane is enabled every
brush edit (smooth dabs, grabs) runs a second time at its mirrored
position, so sculpting one side sculpts both.

Mirroring the brush, not the vertices: real-world meshes are not
vertex-symmetric, so matching vertices across a plane is hopeless.
Instead each brush hit (a camera ray intersecting the surface) is
reflected: reflecting the ray's origin and direction gives the ray the
mirrored camera would cast, and whatever geometry it hits - with
whatever vertex layout happens to be there - receives the mirrored
deformation.  Equivalent, and cheaper: reflect the hit point and face
normal, then let the brush's radius query find the vertices around the
mirrored spot on its own.

For k enabled axes the reflections compose, so two checked axes also
deform the diagonal quadrant (2^k - 1 images per edit).

Strokes NEAR a plane get one extra guard (see ``SymmetryPlanes.fade_mult``):
the motion component that would carry a vertex ACROSS the plane fades to
zero as the plane approaches, so the two mirror images - which demand
opposite cross-plane motion there - blend continuously instead of
tearing the centerline apart.
"""

from __future__ import annotations

import numpy as np


class SymmetryPlanes:
    """Reflection across world-space axis planes through a pivot.

    Cheap value object built per stroke from the actor's world matrix
    and the checked axes (0 = X, 1 = Y, 2 = Z); the brushes call
    ``reflect_point`` / ``reflect_dir`` once per dab / grab.
    """

    def __init__(self, matrix, axes) -> None:
        self.axes = tuple(sorted({int(a) for a in axes}))
        if matrix is None:
            self.pivot = np.zeros(3)
        else:
            self.pivot = np.array(
                [matrix.GetElement(3, 0), matrix.GetElement(3, 1),
                 matrix.GetElement(3, 2)], dtype=np.float64)

    def reflect_point(self, point):
        """Mirror a world-space *point* across every enabled plane."""
        q = np.asarray(point, dtype=np.float64).copy()
        for axis in self.axes:
            q[axis] = 2.0 * self.pivot[axis] - q[axis]
        return q

    def reflect_dir(self, direction):
        """Mirror a world-space direction (delta or normal).

        Directions flip the mirrored components outright; points fold
        about the pivot (see :meth:`reflect_point`).
        """
        q = np.asarray(direction, dtype=np.float64).copy()
        for axis in self.axes:
            q[axis] = -q[axis]
        return q

    def fade_mult(self, world_points, band):
        """Near-plane motion falloff, per vertex and world component.

        Returns an (n, 3) multiplier: 1.0 for every component once the
        vertex sits more than *band* world units from every enabled
        plane, smoothly (smootherstep) damping the component ACROSS an
        enabled plane to 0.0 right on it.  Motion ALONG a plane is never
        damped - disabled axes stay 1.0 everywhere.

        Why: a vertex on the plane is (almost) its own mirror image, so
        the two brush instances around it demand opposite cross-plane
        motions; the only demand they agree on is tangential.  Damping
        the cross-plane component to zero at the plane (and back to full
        beyond the brush radius) keeps the two sides continuous instead
        of fighting - the same "center falloff" idea ZBrush / 3D-Coat /
        Mudbox apply around the symmetry line when sculpting across it.
        """
        pts = np.asarray(world_points, dtype=np.float64)
        mult = np.ones_like(pts)
        for axis in self.axes:
            s = np.clip(np.abs(pts[:, axis] - self.pivot[axis]) / band,
                        0.0, 1.0)
            mult[:, axis] = s * s * (3.0 - 2.0 * s)
        return mult
