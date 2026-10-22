"""Grab-and-move brush ('snake hook') on the displayed mesh.

Per stroke: on press, every vertex inside the brush radius is grabbed with
a fixed spinning-top weight (a wide, low body out at the rim focusing
into a needle-sharp tip at the grab centre); every mouse move then
displaces the grabbed set by the screen drag converted to world units on
the camera plane.  The grabbed set and its weights are frozen for the
whole stroke, so geometry pulled off the surface keeps following the
cursor exactly like a hook - no re-picking.  An optional per-vertex mask
restricts the grab to one polygroup's faces (see face_group_mask).

Design notes:
* Reuses :class:`~tools.mesh_edit.smooth_operator.MeshDeformer`'s stroke
  infrastructure (frozen point locator, first-touch undo capture, region
  normal refresh) so the behaviour and perf characteristics match the
  smooth brush: one undo entry per stroke, single-digit ms per move on
  dense meshes.
* Falloff curve and stroke tuning are hardcoded module constants.
"""

from __future__ import annotations

import math

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from tools.mesh_edit.smooth_operator import MeshDeformer

#: below this screen-drag distance (px) a move is skipped entirely
_MIN_MOVE_PX = 0.01

#: falloff angle, in radians; the curve is tan((1-t)*a)/tan(a), so a
#: just under pi/2 focuses the weight into a needle-sharp tip at the
#: grab centre, while smaller a widens the spike toward a cone
_TAN_SHARPNESS = 0.9

#: fraction of the cursor drag converted into deformation (0.9 = a
#: 100x100 px cursor move deforms the grabbed tip by 90x90 of its
#: world-equivalent; 1.0 = the tip tracks the cursor exactly)
_MOVE_FOLLOW = 0.9


def _falloff(t, sharpness=None):
    """Grab weight for normalised radius *t* (0 at centre, 1 at rim).

    Spinning-top profile: ``tan((1-t)*a)/tan(a)`` - like the first tan
    curve mirrored.  Wide, flat body out at the rim, rising slowly and
    then curving up into a near-vertical sharp tip at the grab centre
    (the tangent's asymptote lands at t = 0), so pulled geometry tapers
    into a point instead of a rounded dome.  *sharpness* overrides the
    module default (exposed in the brush popup); higher a = thinner
    needle, lower a = wider cone.

    The tail is multiplied by a smootherstep feather so weight *and*
    slope reach exactly zero at the rim: on its own the tan lands
    linearly, and the last vertex ring moving while its outside
    neighbours stand still reads as a hard cut around the affected area.
    """
    tc = np.clip(t, 0.0, 1.0)
    a = _TAN_SHARPNESS if sharpness is None else float(sharpness)
    spike = np.tan(a * (1.0 - tc)) / math.tan(a)
    fade = 1.0 - tc * tc * tc * (tc * (tc * 6.0 - 15.0) + 10.0)
    return spike * fade


class MeshGrabber:
    """Grabs a weighted vertex set on press and displaces it on move."""

    def __init__(self, deformer: MeshDeformer | None = None):
        # share the smooth brush's MeshDeformer so the heavy per-mesh
        # caches (CSR topology) are built once, not once per brush;
        # per-stroke state is exclusive (only one stroke runs at a time)
        self._deformer = deformer if deformer is not None else MeshDeformer()
        self._actor = None
        self._verts = None                # grabbed vertex ids (mesh space)
        self._weights = None              # per-vertex bell weights
        self._active = False
        # symmetry (stroke-frozen): _verts is the UNION of the primary
        # grab and the mirrored grab; _wa / _wb hold each vertex's two
        # falloff weights (0 where a set doesn't reach it - shared
        # near-plane vertices carry BOTH, so their opposing cross-plane
        # motions cancel while their agreeing tangential ones add), and
        # _fade is the per-vertex (n, 3) near-plane multiplier cleaning
        # the residual (see SymmetryPlanes.fade_mult).  All None without
        # symmetry.
        self._wa = None
        self._wb = None
        self._fade = None
        self._sym = None

    # -- stroke lifecycle ------------------------------------------------------
    def begin_stroke(self, actor, center_world, radius_world: float,
                     sharpness=None, allowed_verts=None) -> bool:
        """Freeze the grabbed set at *center_world*; False when empty.

        *sharpness* is the falloff spike focus (see :func:`_falloff`);
        None keeps the module default.  *allowed_verts* optionally limits
        the grab to a boolean per-vertex mask (polygroup-only mode).
        """
        deformer = self._deformer
        polydata = deformer._polydata(actor)
        if polydata is None:
            return False
        # topology caches + frozen point locator + fresh undo capture
        if not deformer.begin_stroke(actor):
            return False
        mesh_center, mesh_radius = deformer._to_mesh(actor, center_world,
                                                     radius_world)
        if mesh_center is None or mesh_radius <= 0.0:
            deformer.end_stroke()
            return False
        ids = vtk.vtkIdList()
        deformer._locator.FindPointsWithinRadius(float(mesh_radius),
                                                 tuple(mesh_center), ids)
        count = ids.GetNumberOfIds()
        if count == 0:
            deformer.end_stroke()
            return False
        candidates = np.fromiter((ids.GetId(i) for i in range(count)),
                                 np.int64, count)
        points = vtk_to_numpy(polydata.GetPoints().GetData())
        dist = np.linalg.norm(points[candidates] - mesh_center, axis=1)
        # exact re-test: locator buckets can predate earlier edits
        inside = dist <= mesh_radius
        if allowed_verts is not None:
            inside &= np.asarray(allowed_verts, dtype=bool)[candidates]
        verts = candidates[inside]
        if len(verts) == 0:
            deformer.end_stroke()
            return False
        t = np.clip(dist[inside] / mesh_radius, 0.0, 1.0)
        weights = _falloff(t, sharpness)
        self._weights = weights
        self._verts = verts
        # symmetry: detect BOTH sides' affected vertices once, here at
        # stroke start - the mirrored set is grabbed exactly like the
        # primary one and the two are merged into one union with per-set
        # weights (shared near-plane vertices end up in both lists)
        self._wa = None
        self._wb = None
        self._fade = None
        self._sym = deformer.symmetry
        if self._sym is not None:
            self._grab_mirror(deformer, actor, polydata, center_world,
                              mesh_radius, radius_world, sharpness,
                              allowed_verts)
        deformer._capture_before(polydata, self._verts)
        self._actor = actor
        self._active = True
        return True

    def _grab_mirror(self, deformer, actor, polydata, center_world,
                     radius_mesh, radius_world, sharpness, allowed_verts):
        """Grab the mirrored set and MERGE it with the primary one.

        The primary grab's centre is reflected across the enabled world
        planes - the point the mirrored camera ray would hit - and
        re-grabbed with the same falloff.  The two sets are then merged
        into one vertex union that keeps BOTH weights per vertex:

        * a vertex only the primary reach moves by the primary delta,
        * a vertex only the mirror reach moves by the reflected delta,
        * a SHARED vertex (the near-plane region both spheres cover)
          receives both contributions - and because the two deltas are
          mirror images, their cross-plane components oppose and cancel
          while their tangential components agree, which is exactly the
          behaviour a vertex on the symmetry seam must have.  (The same
          "brush runs twice" overlap ZBrush uses; averaging the pair
          instead would halve the strength with a visible discontinuity
          at the overlap rim.)

        The merged set's frozen near-plane fade then removes the small
        residual cross-plane difference between the two weights, so the
        centerline stays clean even when the falloffs are lopsided.
        """
        sym = self._sym
        mirror_center, _ = deformer._to_mesh(
            actor, sym.reflect_point(center_world), radius_mesh)
        if mirror_center is None:
            return
        ids = vtk.vtkIdList()
        deformer._locator.FindPointsWithinRadius(float(radius_mesh),
                                                 tuple(mirror_center), ids)
        count = ids.GetNumberOfIds()
        if count == 0:
            return
        candidates = np.fromiter((ids.GetId(i) for i in range(count)),
                                 np.int64, count)
        points = vtk_to_numpy(polydata.GetPoints().GetData())
        dist = np.linalg.norm(points[candidates] - mirror_center, axis=1)
        inside = dist <= radius_mesh
        if allowed_verts is not None:
            inside &= np.asarray(allowed_verts, dtype=bool)[candidates]
        verts2 = candidates[inside]
        if len(verts2) == 0:
            return
        t2 = np.clip(dist[inside] / radius_mesh, 0.0, 1.0)
        wb2 = _falloff(t2, sharpness)
        # merge: union of both id sets, weights aligned by lookup
        union = np.union1d(self._verts, verts2)
        wa = np.zeros(len(union))
        wa[np.searchsorted(union, self._verts)] = self._weights
        wb = np.zeros(len(union))
        wb[np.searchsorted(union, verts2)] = wb2
        full, trans = deformer._world_xform(actor)
        world_u = points[union] @ full.T + trans
        self._verts = union
        self._wa = wa
        self._wb = wb
        self._fade = sym.fade_mult(world_u, radius_world)

    def face_group_mask(self, actor, face_groups, group):
        """Per-vertex mask of every face in polygroup *group*, or None.

        *face_groups* is the per-polygon group-id array of *actor*
        (``PolygroupController.cell_groups``).  A vertex is included when
        it belongs to at least one face of the group; vertices shared
        with neighbouring groups still move those neighbours as well,
        since the mesh is not split at group borders.
        """
        groups = np.asarray(face_groups)
        wanted = groups == int(group)
        if not wanted.any():
            return None
        return self._deformer.vertex_mask_for_cells(actor, wanted)

    def move(self, delta_world) -> int:
        """Displace the grabbed set by *delta_world*; returns vertex count.

        The delta is applied once per call with the frozen per-vertex
        weights, so dragging back and forth is exactly reversible and
        speed is governed by the caller's mouse-path granularity.
        """
        if (not self._active or self._actor is None
                or self._verts is None
                or float(np.linalg.norm(delta_world)) < 1e-12):
            return 0
        deformer = self._deformer
        actor = self._actor
        polydata = deformer._polydata(actor)
        if polydata is None:
            return 0
        # world delta -> mesh delta (inverse linear part; same convention
        # as _to_mesh's uniform-scale radius mapping).  Row form so the
        # per-vertex symmetry fades can scale world components first.
        world = np.asarray(delta_world, dtype=np.float64)
        lin = np.eye(3)
        matrix = actor.GetMatrix()
        if matrix is not None:
            m = np.array([[matrix.GetElement(r, c) for c in range(4)]
                          for r in range(4)])
            try:
                inv = np.linalg.inv(m)
            except np.linalg.LinAlgError:
                return 0
            lin = inv[:3, :3]
        rows = lin.T
        points = vtk_to_numpy(polydata.GetPoints().GetData())
        if self._wb is not None:
            # Both brush instances at once, per vertex: wa * delta +
            # wb * mirrored(delta).  On shared near-plane vertices the
            # mirrored cross-plane components oppose and cancel while
            # the tangential ones add; the frozen fade then damps what
            # is left of the cross-plane difference.  Everything is
            # combined in world space first, then mapped to mesh space.
            disp = (self._wa[:, None] * world[None, :]
                    + self._wb[:, None]
                    * self._sym.reflect_dir(world)[None, :])
            disp *= self._fade
            points[self._verts] += disp @ rows
        else:
            points[self._verts] += self._weights[:, None] * (world @ rows)
        data = polydata.GetPoints().GetData()
        data.Modified()
        polydata.GetPoints().Modified()
        polydata.Modified()
        topo = deformer._topology_of(polydata)
        deformer._refresh_normals(polydata, topo, self._verts)
        return len(self._verts)

    def end_stroke(self):
        """Finish the stroke; returns an undo record or None (same shape
        as :meth:`MeshDeformer.end_stroke`)."""
        if not self._active:
            return None
        self._active = False
        self._verts = None
        self._weights = None
        self._actor = None
        self._wa = None
        self._wb = None
        self._fade = None
        self._sym = None
        return self._deformer.end_stroke()

    def abort(self):
        """Drop the stroke without finishing (model swapped mid-drag)."""
        self._deformer.end_stroke()
        self._active = False
        self._verts = None
        self._weights = None
        self._actor = None
        self._wa = None
        self._wb = None
        self._fade = None
        self._sym = None
