"""reSym: align a model's symmetry plane with the brush symmetry axis.

The user clicks SEVEN points on the mesh:

* 1, 2 (red)        - two OPPOSITE features (e.g. left and right wing
  tip); the line between them runs ACROSS the body, so its component
  perpendicular to the body line is the mirror-plane normal,
* 3, 4 (blue)       - two points ALONG the body (e.g. middle of the
  scales, a bit further down); this line removes the body-tilt
  component from the wing line,
* 5, 6, 7 (green)   - three points on the BOTTOM of the model; their
  plane is the model's floor, and its normal is the UP axis.

The floor's UP axis is what keeps the model honest: the alignment is a
PURE YAW about that axis - the yaw that swings the plane normal's
horizontal component onto the checked symmetry axis (X/Y/Z).  The model
never pitches or rolls, so its stance on the "ground" is preserved
exactly; the dialog reports the small residual tilt when the hand-picked
floor and plane don't manage a perfect axis alignment.  Afterwards the
model slides along the checked axis so the p1/p2 midpoint plane passes
exactly through the pivot - the same plane the brush symmetry and
MirrorAndMerge use.

On confirmation the world-space transform is applied to the displayed
mesh through the normal edit stack, so it shows up as an unsaved edit
the existing "Bake changes" button writes into the model file.
"""

from __future__ import annotations

import math

import numpy as np
import vtk
from PySide6.QtWidgets import QMessageBox

_MARKER_COLORS = ((0.90, 0.16, 0.16), (0.90, 0.16, 0.16),
                  (0.18, 0.24, 0.72), (0.18, 0.24, 0.72),
                  (0.05, 0.42, 0.16), (0.05, 0.42, 0.16),
                  (0.05, 0.42, 0.16))     # dark green: floor points
_MARKER_FRACTION = 0.006      # marker radius, of the model bounds diagonal

# guidance per placed point, in order; colour identifies the point, no
# numbering
_PROMPTS = (
    "click one wing/feature tip (red)",
    "click the OPPOSITE tip, eyeballed (red)",
    "click the middle of the body (dark blue)",
    "click a bit further along the body (dark blue)",
    "click a first point on the BOTTOM of the model "
    "(dark green, floor)",
    "click a second bottom point, spread apart (dark green)",
    "last: a third bottom spot, not in line with the others "
    "(dark green)",
)


class ResymPicker:
    """Four-click symmetry alignment wizard owned by the viewer."""

    def __init__(self, viewer):
        self.v = viewer
        self.active = False
        self.points = []
        self._markers = []

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        win = self.v.window()
        if self.active:
            self.cancel("reSym: restarted; click point 1/4.")
            return
        if self.v.current_path is None:
            win.log("reSym: no model is loaded.", "warn")
            return
        self.active = True
        self.points = []
        self._drop_markers()
        win.log("reSym: place two red wing tips, two blue body points, "
                "three green floor points.", "info")
        win.log("reSym: a plain click places a point, left-drag orbits, "
                "wheel zooms; Esc cancels.", "info")
        win.log("reSym: " + _PROMPTS[0], "info")

    def cancel(self, message: str = "reSym: cancelled; nothing changed."):
        self.active = False
        self.points = []
        self._drop_markers()
        self.v.render_now()
        self.v.window().log(message, "warn")

    def reset(self):
        """Drop state without touching the renderer (model switch: the
        viewer's RemoveAllViewProps already removed the markers)."""
        self.active = False
        self.points = []
        self._markers = []

    # -- picking -------------------------------------------------------------
    def pick_point(self, x, y):
        """Record one clicked surface point (viewer left-click handler)."""
        hit = self.v.brush.pick(x, y)
        if hit is None:
            return                       # clicked empty space; keep waiting
        position, _normal = hit
        self.points.append(np.asarray(position, dtype=np.float64))
        self._add_marker(position, len(self.points) - 1)
        self.v.render_now()
        count = len(self.points)
        if count < 7:
            self.v.window().log("reSym: " + _PROMPTS[count], "info")
        else:
            self._finish()

    # -- markers -------------------------------------------------------------
    def _add_marker(self, position, index):
        bounds = self.v.model_bounds()
        if bounds is not None:
            diagonal = math.dist(bounds[::2], bounds[1::2])
        else:
            diagonal = 1.0
        radius = max(diagonal * _MARKER_FRACTION, 1e-6)
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*position)
        sphere.SetRadius(radius)
        sphere.SetPhiResolution(24)
        sphere.SetThetaResolution(24)
        sphere.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*_MARKER_COLORS[index])
        actor.GetProperty().LightingOff()
        actor.PickableOff()
        self.v._ren.AddViewProp(actor)
        self._markers.append(actor)

    def _drop_markers(self):
        for actor in self._markers:
            self.v._ren.RemoveViewProp(actor)
        self._markers = []

    # -- the math ------------------------------------------------------------
    def _frame(self):
        """Orthonormal (across, along, normal) frame from the 4 points.

        Returns (across_hat, along_hat, normal, midpoint) or raises
        ValueError with a user-facing message.
        """
        p1, p2, p3, p4 = self.points[:4]
        across = p2 - p1
        along = p4 - p3
        alen = float(np.linalg.norm(across))
        blen = float(np.linalg.norm(along))
        if alen < 1e-9 or blen < 1e-9:
            raise ValueError("two of the points coincide; nothing to "
                             "align with")
        along_hat = along / blen
        normal = across - (across @ along_hat) * along_hat
        nlen = float(np.linalg.norm(normal))
        if nlen < 1e-4 * max(alen, blen):
            raise ValueError("the two picked lines are parallel; the "
                             "wing line needs a cross-body component")
        normal /= nlen
        along_hat = along_hat - (along_hat @ normal) * normal
        along_hat /= float(np.linalg.norm(along_hat))
        across_hat = np.cross(along_hat, normal)
        return across_hat, along_hat, normal, 0.5 * (p1 + p2)

    def _floor_up(self):
        """World UP axis from the three floor points (points 5-7).

        The floor plane's normal, sign-fixed to point INTO the model
        (away from the floor, toward the body points).
        """
        f1, f2, f3 = self.points[4:7]
        up = np.cross(f2 - f1, f3 - f1)
        length = float(np.linalg.norm(up))
        if length < 1e-9:
            raise ValueError("the three floor points are collinear; "
                             "spread them apart on the model's bottom")
        up /= length
        body = np.mean(self.points[:4], axis=0)
        if float((body - f1) @ up) < 0.0:
            up = -up
        return up

    def _finish(self):
        win = self.v.window()
        try:
            _across, _along, normal, midpoint = self._frame()
            up = self._floor_up()
        except ValueError as ex:
            self.cancel(f"reSym: {ex}")
            return

        panel = win.right_panel
        axes = [axis for axis, checkbox in
                enumerate((panel.sym_x, panel.sym_y, panel.sym_z))
                if checkbox.isChecked()]
        if len(axes) != 1:
            self.cancel("reSym: check exactly ONE symmetry axis (X, Y or "
                        "Z) - the wing line is aligned with it.")
            return
        axis = axes[0]
        target_normal = np.zeros(3)
        target_normal[axis] = 1.0

        # PURE YAW about the floor's UP axis: swing the plane normal's
        # horizontal component onto the checked axis's horizontal
        # component.  No pitch, no roll - the model keeps its stance.
        e_horiz = target_normal - (target_normal @ up) * up
        e_len = float(np.linalg.norm(e_horiz))
        n_horiz = normal - (normal @ up) * up
        n_len = float(np.linalg.norm(n_horiz))
        if e_len < 1e-6:
            self.cancel("reSym: the checked axis is parallel to the "
                        "floor normal; pick a horizontal axis.")
            return
        if n_len < 1e-6:
            self.cancel("reSym: the picked plane normal is nearly "
                        "vertical; the wing line needs a horizontal "
                        "component.")
            return
        e_horiz /= e_len
        n_horiz /= n_len
        cos = float(np.clip(n_horiz @ e_horiz, -1.0, 1.0))
        sin = float(np.clip(up @ np.cross(n_horiz, e_horiz), -1.0, 1.0))
        yaw = math.atan2(sin, cos)
        # the plane's normal sign is arbitrary, so n and -n define the
        # same mirror plane: always take the SHORTER spin
        if yaw > math.pi / 2.0:
            yaw -= math.pi
        elif yaw < -math.pi / 2.0:
            yaw += math.pi
        ct, st = math.cos(yaw), math.sin(yaw)
        K = np.array([[0.0, -up[2], up[1]],
                      [up[2], 0.0, -up[0]],
                      [-up[1], up[0], 0.0]])
        rotation = (np.eye(3) * ct + K * st
                    + np.outer(up, up) * (1.0 - ct))
        # bookkeeping check: yaw must not tilt the model
        if not np.allclose(rotation @ up, up, atol=1e-9):
            self.cancel("reSym: internal error (yaw moved the UP axis).")
            return

        # pivot = the model's world origin (actor matrix translation)
        actors = self.v._split_actors()
        pivot = np.zeros(3)
        if actors:
            matrix = actors[0].GetMatrix()
            if matrix is not None:
                pivot = np.array([matrix.GetElement(3, c) for c in range(3)])

        # how tilted is the model relative to the world vertical?
        world_up = np.array([0.0, 1.0, 0.0])
        floor_tilt = math.degrees(math.acos(
            float(np.clip(up @ world_up, -1.0, 1.0))))

        box = QMessageBox(self.v)
        box.setWindowTitle("reSym")
        box.setText(
            f"Align the picked symmetry plane with the {'XYZ'[axis]} "
            "axis?\n\n"
            f"Yaw: {math.degrees(yaw):.2f} degrees\n"
            f"Slide along {'XYZ'[axis]}: shown after you choose\n"
            + (f"Floor tilt vs world Y: {floor_tilt:.2f} degrees\n"
               if floor_tilt > 0.05 else "")
            + "\nYaw only: the model spins around its own UP axis and "
            "keeps its stance and floor height.\n"
            "Yaw + upright: ALSO stands the model up (floor normal -> "
            "world Y), removing the tilt, and keeps its floor height.")
        yaw_btn = box.addButton("Yaw only", QMessageBox.YesRole)
        upright_btn = box.addButton("Yaw + upright", QMessageBox.YesRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.NoRole)
        box.setDefaultButton(yaw_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is cancel_btn or clicked is None:
            self.cancel("reSym: discarded (not applied).")
            return
        upright = clicked is upright_btn
        if upright:
            # full frame alignment: plane normal -> axis, floor UP ->
            # world Y (this also nails the plane exactly - the residual
            # the yaw-only mode leaves is the model's own lean)
            rotation = np.column_stack(
                (target_normal, world_up,
                 np.cross(target_normal, world_up))) @ np.column_stack(
                    (n_horiz, up, np.cross(n_horiz, up))).T
            up_eff = world_up
        else:
            up_eff = up

        # Yaw about a TILTED up axis lifts one side like a spinning top,
        # and a plain slide along the axis would graze the floor height
        # (the axis isn't perfectly horizontal when the floor picks are
        # noisy).  Solve the translation T = a*axis + b*up_eff so BOTH
        # hold exactly: the midpoint plane passes through the pivot along
        # the axis, and the floor centroid keeps its height along UP.
        floor_pts = np.asarray(self.points[4:7])
        floor_after = (floor_pts - pivot) @ rotation.T + pivot
        drop = float((floor_after - floor_pts).mean(axis=0) @ up_eff)
        plane_off = float(((midpoint - pivot) @ rotation.T)
                          @ target_normal)
        e_dot_u = float(target_normal @ up_eff)
        a, b = np.linalg.solve(
            np.array([[1.0, e_dot_u], [e_dot_u, 1.0]]),
            np.array([-plane_off, -drop]))
        slide = a * target_normal + b * up_eff
        # the plane normal's sign is free (the short yaw may align -n),
        # so the residual tilt is the angle to either direction
        residual = math.degrees(math.acos(float(np.clip(
            abs((rotation @ normal) @ target_normal), -1.0, 1.0))))
        win.log(f"reSym: slide along {'XYZ'[axis]} = {-a:.6g}; residual "
                f"tilt {residual:.2f} degrees"
                + ("" if residual <= 1.0 else " (pick noise / lean "
                   "remains in yaw-only mode)"), "info")

        geometry = self.v.edited_geometry()
        if geometry is None:
            self.cancel("reSym: could not read the displayed geometry.")
            return
        vertices, faces = geometry
        moved = (vertices - pivot) @ rotation.T + pivot + slide
        result = {
            "mode_label": "reSym",
            "target_path": str(self.v.current_path),
            "vertices": np.ascontiguousarray(moved),
            "faces": np.ascontiguousarray(faces),
            "vertex_count": int(len(moved)),
            "face_count": int(len(faces)),
        }
        self.active = False
        self.points = []
        self._drop_markers()
        win.edits.accept_result("reSym", result)
