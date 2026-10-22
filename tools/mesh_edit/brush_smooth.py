"""Smooth-brush cursor overlay for the 3D view (tools/mesh_edit).

Draws a thin circle in *screen space* - the same trick as the lasso line:
a single 2D ``vtkActor2D`` in display coordinates, always crisp, always on
top of the 3D scene, no fill, no world-space geometry.  What makes it a
surface brush is the deformation: each update projects a world-space
circle lying in the plane of the hovered face, so facing the camera
straight on it is a perfect circle while tilted faces foreshorten it into
an ellipse whose minor axis follows the face normal - the circle rotates
with the surface exactly as a 3D disc would.

viewer.py owns *when* it shows (hold S over model geometry); this module
only draws.  :meth:`SmoothBrushCursor.place` returns whether anything
changed, so the viewer can skip renders while the mouse rests.
"""

from __future__ import annotations

import math

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

_COLOR = (0.55, 0.85, 1.00)      # light cyan
_OPACITY = 0.9
_WIDTH = 1.5                     # screen pixels
_SIDES = 64                      # ring smoothness
_QUANTUM = 0.75                  # px; sub-step changes don't redraw
_CENTER_SIDES = 12               # centre marker: small ring at the midpoint
_CENTER_RADIUS = 4.0             # px; constant screen size, never scaled
_AXIS_FLOOR = 0.35               # min minor axis as a fraction of the
                                 # nominal radius: tight spots stop the
                                 # ring from visually shrinking away
_GHOST_COLOR = (1.00, 0.62, 0.20)   # orange: the mirrored brush preview
_GHOST_OPACITY = 0.6
_GHOST_WIDTH = 1.0
_GHOST_STIPPLE = 0xAAAA             # dashed so it reads as a preview


def _plane_tangents(normal):
    """Orthonormal (tangent, bitangent) spanning the plane of *normal*.

    The tangent is built from the reference axis least aligned with the
    normal, which keeps the circle's spin stable and avoids a degenerate
    cross product when the normal points straight along a coordinate axis.
    Returns None for a zero-length normal.
    """
    nx, ny, nz = normal
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-9:
        return None
    nx, ny, nz = nx / length, ny / length, nz / length

    ax, ay, az = (1.0, 0.0, 0.0)
    if abs(nx) > 0.9:
        ax, ay, az = (0.0, 1.0, 0.0)
    tx = ay * nz - az * ny           # tangent = reference x normal
    ty = az * nx - ax * nz
    tz = ax * ny - ay * nx
    tl = math.sqrt(tx * tx + ty * ty + tz * tz)
    if tl < 1e-9:
        return None
    tx, ty, tz = tx / tl, ty / tl, tz / tl
    bx = ny * tz - nz * ty           # bitangent = normal x tangent
    by = nz * tx - nx * tz
    bz = nx * ty - ny * tx
    return (tx, ty, tz), (bx, by, bz)


class SmoothBrushCursor:
    """Screen-space brush ring hugging the surface under the mouse.

    The ring is the projection of a world-space circle of *world_radius*
    centred on the hit point and lying parallel to the face normal: its
    two projected radius vectors (tangent, bitangent) are conjugate
    semi-diameters of the ellipse ``center + A*cos(t) + B*sin(t)``, which
    is sampled directly - no ellipse fitting, correct perspective for free.
    """

    def __init__(self, renderer, color=_COLOR):
        self._ren = renderer
        self._visible = False
        self._last = None               # quantized (center, A, B) signature
        self._smooth = None             # (center, A, B) last drawn, for lerp
        self._ghost_visible = False
        self._ghost_last = None

        # Fixed topology: big ring (closed) + small centre marker (closed);
        # only the point coordinates change per update.
        self._polydata, self._points = self._make_ring_polydata()

        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(self._polydata)
        mapper.SetTransformCoordinate(coordinate)

        self._actor = vtk.vtkActor2D()
        self._actor.SetMapper(mapper)
        prop = self._actor.GetProperty()
        prop.SetColor(*color)
        prop.SetLineWidth(_WIDTH)
        prop.SetOpacity(_OPACITY)
        self._actor.PickableOff()      # never blocks picking / lasso
        self._actor.SetVisibility(False)
        # VTK 9.7 has no vtkRenderer.AddActor2D; AddViewProp takes 2D props.
        self._ren.AddViewProp(self._actor)

        # Ghost marker: the mirrored brush preview (symmetry on).  Just
        # the small constant-size centre circle - dashed and dimmer so it
        # reads as a preview next to the real ring.
        self._ghost_polydata, self._ghost_points = (
            self._make_ring_polydata(marker_only=True))
        ghost_mapper = vtk.vtkPolyDataMapper2D()
        ghost_mapper.SetInputData(self._ghost_polydata)
        ghost_mapper.SetTransformCoordinate(coordinate)
        self._ghost_actor = vtk.vtkActor2D()
        self._ghost_actor.SetMapper(ghost_mapper)
        ghost_prop = self._ghost_actor.GetProperty()
        ghost_prop.SetColor(*_GHOST_COLOR)
        ghost_prop.SetLineWidth(_GHOST_WIDTH)
        ghost_prop.SetOpacity(_GHOST_OPACITY)
        ghost_prop.SetLineStipplePattern(_GHOST_STIPPLE)
        self._ghost_actor.PickableOff()
        self._ghost_actor.SetVisibility(False)
        self._ren.AddViewProp(self._ghost_actor)

    @staticmethod
    def _make_ring_polydata(marker_only: bool = False):
        """Ring (closed) + centre marker (closed) line topology.

        *marker_only* drops the big ring: the ghost preview only ever
        draws the small constant-size centre circle.
        """
        polydata = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        polydata.SetPoints(points)
        lines = vtk.vtkCellArray()
        if marker_only:
            ids = (np.arange(_CENTER_SIDES + 1) % _CENTER_SIDES).astype(
                np.int64)
            offsets = np.array([0, _CENTER_SIDES + 1], dtype=np.int64)
        else:
            ids = np.empty(_SIDES + 1 + _CENTER_SIDES + 1, dtype=np.int64)
            ids[:_SIDES + 1] = np.arange(_SIDES + 1) % _SIDES
            base = _SIDES + 1
            ids[base:base + _CENTER_SIDES + 1] = base + (
                np.arange(_CENTER_SIDES + 1) % _CENTER_SIDES)
            offsets = np.array([0, _SIDES + 1,
                                _SIDES + 1 + _CENTER_SIDES + 1],
                               dtype=np.int64)
        lines.SetData(
            numpy_to_vtkIdTypeArray(offsets, deep=True),
            numpy_to_vtkIdTypeArray(ids, deep=True))
        polydata.SetLines(lines)
        return polydata, points

    @property
    def visible(self) -> bool:
        return self._visible

    def place(self, position, normal, world_radius: float,
              smooth: float = 0.0, nominal_px: float | None = None) -> bool:
        """Project the face-parallel ring; True when it moved on screen.

        With *smooth* (0..1) the drawn ring chases its target geometry
        (temporal lerp of center and both radius vectors), which removes
        the jitter dense or noisy meshes otherwise cause.  The change
        signature is computed from the raw target, so smoothing never
        masks real movement.

        With *nominal_px* (the brush's constant screen radius) the drawn
        ring is size-normalised after the lerp: the major semi-axis is
        pinned to it and the minor one is clamped between a floor
        fraction and the nominal.  Tight crevices swing the surface
        normal between neighbouring faces every pick, and both the
        foreshortening and the temporal lerp then shrink the raw ellipse
        - the clamp keeps the outer circle at the size the user set with
        grow/shrink, no matter what the surface does underneath.
        """
        tangents = _plane_tangents(normal)
        if tangents is None or world_radius <= 0.0:
            self.hide()
            return False
        center = self._to_display(position)
        t, b = tangents
        tip_t = self._to_display([position[i] + t[i] * world_radius
                                  for i in range(3)])
        tip_b = self._to_display([position[i] + b[i] * world_radius
                                  for i in range(3)])
        a = tip_t - center             # projected radius along the tangent
        c = tip_b - center             # projected radius along the bitangent

        signature = (tuple(np.round(center / _QUANTUM)),
                     tuple(np.round(a / _QUANTUM)),
                     tuple(np.round(c / _QUANTUM)))
        if self._visible and signature == self._last:
            return False
        self._last = signature

        if 0.0 < smooth < 1.0 and self._smooth is not None:
            k = float(smooth)
            center = self._smooth[0] + (center - self._smooth[0]) * k
            a = self._smooth[1] + (a - self._smooth[1]) * k
            c = self._smooth[2] + (c - self._smooth[2]) * k
        self._smooth = (center, a, c)

        if nominal_px is not None and nominal_px > 0.0:
            m = np.array([[a @ a, a @ c], [a @ c, c @ c]])
            evals, evecs = np.linalg.eigh(m)      # ascending eigenvalues
            if evals[1] > 1e-12:
                minor = float(np.sqrt(max(evals[0], 0.0)))
                minor = min(max(minor, _AXIS_FLOOR * nominal_px),
                            float(nominal_px))
                a = nominal_px * evecs[:, 1]      # major semi-axis: constant
                c = minor * evecs[:, 0]

        theta = np.linspace(0.0, 2.0 * np.pi, _SIDES, endpoint=False)
        ring = (center
                + np.outer(np.cos(theta), a)
                + np.outer(np.sin(theta), c))
        self._draw(center, ring)
        return True

    def place_screen(self, x_disp, y_disp, radius_px: float) -> bool:
        """Flat screen-space circle at a display point; True when it moved.

        Draw-mode idle cursor over empty space: with no surface to hug the
        ring degenerates to a perfect circle in display coordinates - same
        radius, colour and centre marker as the surface ring, but it never
        foreshortens ("stays straight").  Same quantized-signature skip as
        :meth:`place` so resting the mouse costs no redraws.
        """
        if radius_px <= 0.0:
            self.hide()
            return False
        center = np.array([float(x_disp), float(y_disp)])
        a = np.array([radius_px, 0.0])
        c = np.array([0.0, radius_px])
        signature = (tuple(np.round(center / _QUANTUM)),
                     tuple(np.round(a / _QUANTUM)),
                     tuple(np.round(c / _QUANTUM)))
        if self._visible and signature == self._last:
            return False
        self._last = signature
        theta = np.linspace(0.0, 2.0 * np.pi, _SIDES, endpoint=False)
        ring = (center
                + radius_px * np.column_stack((np.cos(theta),
                                               np.sin(theta))))
        self._draw(center, ring)
        return True

    def _draw(self, center, ring):
        """Stamp one ring frame (centre marker included) and show it."""
        # centre marker: constant-size small ring so the exact midpoint of
        # the brush is visible even on a heavily foreshortened ring
        theta_s = np.linspace(0.0, 2.0 * np.pi, _CENTER_SIDES, endpoint=False)
        marker = (center
                  + _CENTER_RADIUS
                  * np.column_stack((np.cos(theta_s), np.sin(theta_s))))
        # both rings are drawn closed: repeat the first point of each so the
        # topology's closing index always points at a real point
        points = np.vstack((ring, ring[:1], marker, marker[:1]))
        self._points.SetData(numpy_to_vtk(
            np.column_stack((points, np.zeros(len(points)))), deep=True))
        self._polydata.Modified()
        if not self._visible:
            self._visible = True
            self._actor.SetVisibility(True)

    # -- mirrored brush preview (symmetry) ---------------------------------
    def place_ghost(self, position, normal, world_radius: float) -> bool:
        """Stamp the mirrored-side centre marker; True when it moved.

        A live preview of where the second brush instance (symmetry on)
        is going to work: the small constant-size centre circle, dashed
        and dimmer, projected at the reflected brush hit.  *normal* and
        *world_radius* are kept for interface symmetry with
        :meth:`place` (and the degenerate-normal hide) - the marker
        itself stays a constant screen-size circle, so it stays readable
        even when the mirrored spot faces away or is heavily
        foreshortened.
        """
        tangents = _plane_tangents(normal)
        if tangents is None or world_radius <= 0.0:
            self.hide_ghost()
            return False
        center = self._to_display(position)
        signature = tuple(np.round(center / _QUANTUM))
        if self._ghost_visible and signature == self._ghost_last:
            return False
        self._ghost_last = signature
        theta_s = np.linspace(0.0, 2.0 * np.pi, _CENTER_SIDES, endpoint=False)
        marker = (center
                  + _CENTER_RADIUS
                  * np.column_stack((np.cos(theta_s), np.sin(theta_s))))
        points = np.vstack((marker, marker[:1]))
        self._ghost_points.SetData(numpy_to_vtk(
            np.column_stack((points, np.zeros(len(points)))), deep=True))
        self._ghost_polydata.Modified()
        if not self._ghost_visible:
            self._ghost_visible = True
            self._ghost_actor.SetVisibility(True)
        return True

    def hide_ghost(self):
        self._ghost_last = None
        if not self._ghost_visible:
            return
        self._ghost_visible = False
        self._ghost_actor.SetVisibility(False)

    def hide(self):
        self._last = None
        self._smooth = None
        if self._visible:
            self._visible = False
            self._actor.SetVisibility(False)
        self.hide_ghost()

    def snap(self):
        """Drop the temporal-lerp history, keeping the ring visible.

        Called when a stroke ends: the lerp leaves the drawn ring trailing
        behind its target after a fast drag, and without this the leftover
        distance is eaten up on the next hover - which reads as the ring
        jumping off in the stroke direction.  The next place() positions
        exactly instead of easing in from stale state.
        """
        self._smooth = None

    def rehook(self):
        """Re-add the actors after RemoveAllViewProps (model switch/clear)."""
        self._ren.AddViewProp(self._actor)
        self._ren.AddViewProp(self._ghost_actor)
        self._visible = False
        self._actor.SetVisibility(False)
        self._last = None
        self._smooth = None
        self.hide_ghost()

    def _to_display(self, world_point):
        """World-space point -> display coordinates (pixels, bottom-left)."""
        self._ren.SetWorldPoint(world_point[0], world_point[1],
                                world_point[2], 1.0)
        self._ren.WorldToDisplay()
        return np.asarray(self._ren.GetDisplayPoint()[:2], dtype=np.float64)
