from __future__ import annotations

import math

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from PySide6.QtCore import QTimer, Qt

from debug_trace import cam_info, trace, vec3
from tools.mesh_edit.brush_smooth import SmoothBrushCursor, _SIDES as _RING_SIDES
from tools.history import VertexEditEntry
from tools.mesh_edit.smooth_operator import MeshDeformer
from tools.mesh_edit.move_operator import (
    MeshGrabber, _MIN_MOVE_PX, _MOVE_FOLLOW,
)
from tools.mesh_edit.symmetry import SymmetryPlanes
from brush_settings import BrushSettings, BrushSettingsPopup


_BRUSH_SMOOTHING = 0.35
_BRUSH_GROW = 1.15                    # per press of the grow/shrink hotkeys
_BRUSH_RADIUS_MIN_PX = 4.0
_SMOOTH_INNER_RATIO = 0.6             # invisible inner core, fraction of R
_SMOOTH_INNER_BOOST = 1.2             # falloff multiplier inside the core

# Hover ring normal stabilisation (see BrushController._ring_normal): the
# picked face's normal is averaged with the faces around its vertices,
# oriented toward the camera, then low-passed across picks.  The ring waits
# for the hit cell to change before recomputing the neighbourhood average,
# so the extra work stays a handful of vector ops on triangle crossings.
_RING_NORMAL_SAMPLES = 16             # max faces averaged per hit cell
_RING_NORMAL_LERP = 0.35              # temporal follow rate for the normal
_RING_NORMAL_CACHE = 4096             # per-actor cached cell averages


class BrushController:
    def __init__(self, viewer):
        self.v = viewer
        self.cursor = SmoothBrushCursor(viewer._ren)
        self.held = False
        self.shift = False              # brush activated by a Shift hold
        self.draw_enabled = False       # Draw toggle (outliner panel) gate
        self.cursor_saved = None        # cursor hidden while the circle shows
        self.cursor_hidden = False      # True between hide/restore pairs
        self.normal_cache = {}          # actor -> world-space face normals
        self.ring_normal_cache = {}     # actor -> {cell: averaged normal}
        self._ring_normal_prev = None   # (actor, low-passed hover normal)
        self.locators = {}              # actor -> vtkStaticCellLocator
        self._locator_snaps = {}        # actor -> frozen polydata locator() built on
        self.last_hit = None            # cached triangle for the cheap ray test
        self.hit = None                 # last (position, normal) on the mesh
        self.hit_actor = None           # source actor of the last pick
        self.last_event = None          # (x, y) of the last processed move
        self.render_pending = False
        # exposed brush values (right-click popup in Draw mode); the
        # controller reads them live on every dab / spacing / radius use
        self.settings = BrushSettings.load()
        self.settings.changed.connect(self._on_settings_changed)
        self._popup = None
        self.radius_px = float(self.settings.radius_px)   # screen-size

        # Localized Laplacian smoothing - see tools/mesh_edit/smooth_operator.py
        # One MeshDeformer shared by BOTH stroke engines so the heavy
        # per-mesh topology cache is built once per polydata, not once
        # per brush (strokes never run simultaneously).
        self.deformer = MeshDeformer()
        self.stroke_actor = None        # actor the stroke edits
        self.stroke_pos = None          # last dab position (world space)
        self.stroke_moved = 0           # vertices moved by this stroke
        self.stroke_allowed = None      # vertex mask: visible polygroups
        self.stroke_cells = None        # cell mask for filtered stroke picks

        # Grab-and-move ("snake hook") stroke: plain left-drag on the mesh
        # in Draw mode; Shift keeps the smooth brush (hardcoded tuning,
        # see tools/mesh_edit/move_operator.py)
        self.grabber = MeshGrabber(deformer=self.deformer)
        self._warmed_path = None        # model the caches were built for
        self.move_actor = None          # actor the move stroke edits
        self.move_center = None         # world centre of the grabbed set
        self.move_normal = None         # face normal at grab time (for ring)
        self.move_last = None           # last processed screen position
        self.move_moved = 0             # vertices displaced by this stroke

    # -- symmetry ----------------------------------------------------------
    def _active_symmetry(self, actor):
        """SymmetryPlanes from the panel switches, or None.

        Shared by the stroke gate (``_setup_symmetry``) and the mirrored
        preview ring, so what you see is exactly what a stroke will use.
        """
        panel = getattr(self.v.window(), "right_panel", None)
        switch = getattr(panel, "symmetry_enable", None)
        if switch is None or not switch.isChecked():
            return None
        checks = (getattr(panel, "sym_x", None),
                  getattr(panel, "sym_y", None),
                  getattr(panel, "sym_z", None))
        axes = tuple(axis for axis, cb in enumerate(checks)
                     if cb is not None and cb.isChecked())
        if not axes:
            return None
        return SymmetryPlanes(actor.GetMatrix(), axes)

    def _setup_symmetry(self, actor):
        """Attach a SymmetryPlanes to the deformer for the next stroke.

        Reads the right panel's Symmetry switch plus its X/Y/Z axis
        checkboxes (world-space planes through the model's world
        origin); with the switch off or no axis checked the stroke
        simply stays asymmetric (deformer.symmetry = None).
        """
        self.deformer.symmetry = self._active_symmetry(actor)

    def _refresh_ghost(self, position=None, normal=None, actor=None):
        """Re-pose the mirrored preview ring (or hide it).

        *position*/*normal* default to the current hover/stroke hit;
        drawn only while symmetry is on for the hit's actor, so the user
        always sees where the second brush instance will land BEFORE
        starting a deformation.
        """
        if position is None:
            hit = self.hit
            if hit is None:
                self.cursor.hide_ghost()
                return
            position, normal = hit
        actor = actor if actor is not None else self.hit_actor
        sym = self._active_symmetry(actor) if actor is not None else None
        if sym is None:
            self.cursor.hide_ghost()
            return
        ghost_pos = sym.reflect_point(position)
        self.cursor.place_ghost(ghost_pos, sym.reflect_dir(normal),
                                self.world_radius(ghost_pos))

    # -- live-state helpers ---------------------------------------------------
    @property
    def shift_active(self) -> bool:
        return self.shift

    @property
    def stroking(self) -> bool:
        return self.stroke_actor is not None

    def on_model_cleared(self):
        self.close_settings_popup()
        self.hide()
        self.held = False
        self.shift = False
        if self.draw_enabled:
            self.draw_enabled = False
            self.v.drawModeChanged.emit(False)
        self.v._orbit_pivot = None   # pivot belonged to the old model
        self.normal_cache = {}
        self.ring_normal_cache = {}
        self._ring_normal_prev = None
        self.locators = {}
        self._locator_snaps = {}
        self.last_hit = None
        self.hit = None
        self.hit_actor = None
        self.last_event = None
        self.stroke_actor = None
        self.stroke_pos = None
        self.stroke_moved = 0
        self.stroke_allowed = None
        self.stroke_cells = None
        self.deformer = MeshDeformer()   # fresh topology caches per model
        self.grabber = MeshGrabber(deformer=self.deformer)   # shared again
        self._warmed_path = None
        self.move_actor = None
        self.move_center = None
        self.move_cursor = None
        self.move_normal = None
        self.move_face_screen = None
        self.move_last = None
        self.move_moved = 0
        self.cursor.rehook()             # renderer lost its view props

    def forget_actor(self, actor):
        if self.last_hit is not None:
            self.last_hit = None         # its triangle may be gone
        self.normal_cache.pop(actor, None)
        self.ring_normal_cache.pop(actor, None)
        if (self._ring_normal_prev is not None
                and self._ring_normal_prev[0] is actor):
            self._ring_normal_prev = None
        self.locators.pop(actor, None)
        self._locator_snaps.pop(actor, None)

    def invalidate(self, actor=None):
        self.last_hit = None
        self.hit = None
        if actor is None:
            self.ring_normal_cache.clear()
        else:
            self.ring_normal_cache.pop(actor, None)
        if (actor is None or (self._ring_normal_prev is not None
                              and self._ring_normal_prev[0] is actor)):
            self._ring_normal_prev = None
        if actor is not None:
            self.normal_cache.pop(actor, None)
            self.locators.pop(actor, None)
            self._locator_snaps.pop(actor, None)

    # -- key handling ----------------------------------------------------------
    def warm_up(self):
        for source in self.v._polygroup_actors():
            self.world_normals(source)      # hover ring + flatten normals
            self.locator(source)            # per-actor pick locators
            polydata = self.deformer._polydata(source)
            if polydata is not None:
                # shared by smooth dabs AND move grabs; connectivity never
                # changes, so this stays valid for the mesh's lifetime
                self.deformer._topology_of(polydata)
        self._warmed_path = self.v._current_path

    def set_draw_enabled(self, enabled: bool):
        trace("brush.draw", enabled=bool(enabled), mode=self.v._mode,
              mouse=tuple(self.v._mouse_pos))
        self.draw_enabled = bool(enabled)
        if enabled:
            if self._warmed_path != self.v._current_path:
                self.warm_up()
            if self.v._current_path is not None:
                self.last_event = None
                self.hover(*self.v._mouse_pos)
            return
        if self.shift:
            self.shift = False
            self.held = False
            if self.stroking:
                self.end_stroke()   # Draw turned off mid-stroke
        self.hide()

    def key(self, event, pressed: bool) -> bool:
        combo = bool(event.modifiers() & (Qt.ControlModifier
                                          | Qt.AltModifier | Qt.MetaModifier))
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            if pressed:
                if (self.draw_enabled and not combo
                        and self.v._mode is None and not self.stroking):
                    self.open_settings_popup()      # next to the cursor
                    return True
                return False
            if self._popup is not None:
                self.close_settings_popup()         # Space released
                return True
            return False
        if event.key() == Qt.Key_Shift and not event.isAutoRepeat():
            if pressed:
                # Activate the brush only when hovering the model and the
                # Draw toggle is on; over empty space Shift keeps its
                # camera roles untouched, and Ctrl/Alt combos (lasso,
                # shortcuts) never start it.
                hit = (self.draw_enabled
                       and not combo
                       and not self.shift
                       and self.v._current_path is not None
                       and self.pick(*self.v._mouse_pos) is not None)
                trace("brush.shift", pressed=True, activate=bool(hit),
                      mouse=tuple(self.v._mouse_pos), draw=self.draw_enabled,
                      combo=combo)
                if hit:
                    self.shift = True
                    self.held = True
                    self.hover(*self.v._mouse_pos)
                    return True
                return False
            if self.shift:
                trace("brush.shift", pressed=False, mouse=tuple(self.v._mouse_pos))
                self.shift = False
                self.held = False
                if self.stroking:
                    self.end_stroke()   # Shift released mid-stroke
                self.hide()
                # the ring stays on as the Draw-mode cursor after a session
                if (self.draw_enabled
                        and self.v._current_path is not None):
                    self.last_event = None
                    self.hover(*self.v._mouse_pos)
                return True
            return False
        if pressed and self.draw_enabled and not combo:
            # auto-repeat allowed: holding W / S steadily resizes the ring
            if event.key() == Qt.Key_W:
                self.resize(_BRUSH_GROW)
                return True
            if event.key() == Qt.Key_S:
                self.resize(1.0 / _BRUSH_GROW)
                return True
        return False

    # -- ring cursor ------------------------------------------------------------
    def resize(self, factor: float):
        limit = max(_BRUSH_RADIUS_MIN_PX + 1.0,
                    min(self.v.width(), self.v.height()) * 0.45)
        self.radius_px = min(max(self.radius_px * factor,
                                 _BRUSH_RADIUS_MIN_PX), limit)
        # keep the settings store in sync (Save must capture hotkey resizing);
        # silent - resize already re-poses the ring below
        self.settings.set("radius_px", self.radius_px, emit=False)
        if self.hit is not None and self.cursor.visible:
            position, normal = self.hit
            if self.cursor.place(position, normal,
                                 self.world_radius(position),
                                 smooth=_BRUSH_SMOOTHING,
                                 nominal_px=self.radius_px):
                self.schedule_render()
            self._refresh_ghost()   # keep the mirrored preview in step
        elif self.draw_enabled and not self.held:
            # idle Draw-mode cursor over empty space: just resize the
            # flat screen circle where the mouse currently is
            self.last_event = None
            self.hover(*self.v._mouse_pos)

    def _on_settings_changed(self):
        self.radius_px = max(float(self.settings.radius_px),
                             _BRUSH_RADIUS_MIN_PX)
        if self.hit is not None and self.cursor.visible:
            position, normal = self.hit
            if self.cursor.place(position, normal,
                                 self.world_radius(position),
                                 smooth=_BRUSH_SMOOTHING,
                                 nominal_px=self.radius_px):
                self.schedule_render()
            self._refresh_ghost()   # keep the mirrored preview in step
        elif self.draw_enabled and not self.held:
            self.last_event = None
            self.hover(*self.v._mouse_pos)

    def open_settings_popup(self, global_pos=None):
        if self._popup is not None:
            self._popup.close()
        popup = BrushSettingsPopup(self.settings, parent=self.v)
        # The popup can also close itself (outside click, Space release
        # caught by its application event filter): keep _popup in sync.
        popup.closed.connect(self._on_popup_closed)
        self._popup = popup
        popup.open_at(global_pos)

    def _on_popup_closed(self):
        self._popup = None

    def close_settings_popup(self):
        if self._popup is not None:
            self._popup.close()
            self._popup = None

    def world_radius(self, position) -> float:
        cam = self.v._ren.GetActiveCamera()
        d = cam.GetDirectionOfProjection()
        up = cam.GetViewUp()
        right = (d[1] * up[2] - d[2] * up[1],
                 d[2] * up[0] - d[0] * up[2],
                 d[0] * up[1] - d[1] * up[0])
        length = math.sqrt(sum(c * c for c in right)) or 1.0
        right = [c / length for c in right]
        base = np.asarray(self.project_to_display(position))
        offset = np.asarray(self.project_to_display(
            [position[i] + right[i] for i in range(3)]))
        pixels = float(np.linalg.norm(offset - base))
        if pixels < 1e-9:
            return float(self.radius_px * self.v._world_per_pixel())
        return float(self.radius_px / pixels)

    def project_to_display(self, world_point):
        ren = self.v._ren
        ren.SetWorldPoint(world_point[0], world_point[1],
                          world_point[2], 1.0)
        ren.WorldToDisplay()
        return ren.GetDisplayPoint()[:2]

    def _trace_ring(self) -> dict:
        try:
            if not self.cursor.visible:
                return {"ring": None}
            pts = vtk_to_numpy(self.cursor._points.GetData())[:_RING_SIDES,
                                                             :2]
            center = pts.mean(0)
            radii = np.linalg.norm(pts - center, axis=1)
            smooth = self.cursor._smooth
            return {
                "ring": (round(float(center[0]), 1),
                         round(float(center[1]), 1)),
                "ring_major": round(float(radii.max()), 1),
                "ring_minor": round(float(radii.min()), 1),
                "ring_target": (None if smooth is None
                                else (round(float(smooth[0][0]), 1),
                                      round(float(smooth[0][1]), 1))),
            }
        except Exception:  # noqa: BLE001 - tracing only
            return {"ring": "err"}

    def hide_mouse_cursor(self):
        if self.cursor_hidden:
            return
        self.cursor_saved = self.v.cursor()
        self.v.setCursor(Qt.BlankCursor)
        self.cursor_hidden = True

    def restore_mouse_cursor(self):
        if not self.cursor_hidden:
            return
        self.cursor_hidden = False
        saved = self.cursor_saved
        self.cursor_saved = None
        self.v.setCursor(saved if saved is not None else Qt.ArrowCursor)

    def hover(self, x, y):
        if self.v._current_path is None:
            return
        if not self.held and not self.draw_enabled:
            return
        last = self.last_event
        if self.cursor.visible and last is not None:
            if (x - last[0]) ** 2 + (y - last[1]) ** 2 < 4.0:
                trace("brush.hover.skip", xy=(x, y), last=last)
                return                    # under the hysteresis threshold
        self.last_event = (x, y)
        before = self._trace_ring()
        hit = self.pick(x, y)
        if hit is None:
            if self.held:
                self.hide()     # brush session moved off the mesh
                return
            # idle Draw-mode cursor over empty space: flat screen circle
            self.hit = None
            self.cursor.hide_ghost()
            self.hide_mouse_cursor()
            x_disp, y_disp = self.v._display_point(x, y)
            if self.cursor.place_screen(x_disp, y_disp, self.radius_px):
                self.schedule_render()
            trace("brush.hover.miss", xy=(x, y), disp=(x_disp, y_disp),
                  **{**before, **self._trace_ring()},
                  **cam_info(self.v._ren.GetActiveCamera()))
            return
        position, normal = hit
        entry = self.last_hit or {}
        normal = self._ring_normal(entry.get("actor"), entry.get("cell_id"),
                                   position, normal)
        self.hit = (position, normal)
        newly_shown = not self.cursor.visible
        changed = self.cursor.place(
            position, normal, self.world_radius(position),
            smooth=_BRUSH_SMOOTHING, nominal_px=self.radius_px)
        trace("brush.hover", xy=(x, y), pick=tuple(round(v, 3)
                                                   for v in position),
              normal=tuple(round(float(v), 3) for v in normal),
              newly_shown=newly_shown, changed=changed,
              **{**before, **self._trace_ring()},
              **cam_info(self.v._ren.GetActiveCamera()))
        if changed or newly_shown:
            self.hide_mouse_cursor()
            self.schedule_render()
        # mirrored preview ring follows the hover hit (symmetry on)
        self._refresh_ghost(position, normal)

    def drag_cursor(self, x, y):
        if not self.draw_enabled or self.stroking:
            return
        self.hit = None
        self.cursor.hide_ghost()
        self.hide_mouse_cursor()
        x_disp, y_disp = self.v._display_point(x, y)
        if self.cursor.place_screen(x_disp, y_disp, self.radius_px):
            self.schedule_render()

    def schedule_render(self):
        if self.render_pending:
            return
        self.render_pending = True
        QTimer.singleShot(0, self.render_now_pending)

    def render_now_pending(self):
        self.render_pending = False
        if self.v.isVisible():
            self.v.render_now()

    def hide(self):
        if self.cursor.visible:
            self.cursor.hide()
            self.v.render_now()
        self.restore_mouse_cursor()

    # -- picking ------------------------------------------------------------
    def pick(self, x, y):
        hit = self.ray_hits_cached_triangle(x, y)
        if hit is not None:
            self.hit_actor = self.last_hit.get("actor")
            return hit
        ray = self.ray(*self.v._display_point(x, y))
        if ray is None:
            return None
        origin, direction = ray
        _, far = self.v._ren.GetActiveCamera().GetClippingRange()
        best = None
        for actor, source, cell_map in self.pick_candidates():
            locator = self.locator(actor)
            if locator is None:
                continue
            mesh_origin, mesh_direction = self.world_to_mesh(
                actor, origin, direction)
            if mesh_origin is None:
                continue
            hits = vtk.vtkPoints()
            cell_ids = vtk.vtkIdList()
            locator.IntersectWithLine(tuple(mesh_origin),
                                      tuple(mesh_origin
                                            + mesh_direction * far),
                                      1e-9, hits, cell_ids)
            if cell_ids.GetNumberOfIds() == 0:
                continue
            world_hit = self.mesh_to_world(actor, hits.GetPoint(0))
            if world_hit is None:
                continue
            distance = float(np.linalg.norm(world_hit - origin))
            if best is not None and distance >= best[0]:
                continue
            source_id = (int(cell_map[cell_ids.GetId(0)])
                         if cell_map is not None else int(cell_ids.GetId(0)))
            best = (distance, world_hit, source, source_id)
        if best is None:
            self.last_hit = None
            return None
        _, position, source, cell_id = best
        self.hit_actor = source
        normals = self.world_normals(source)
        if normals is None or cell_id >= len(normals):
            return None
        verts = self.triangle_verts(source, cell_id)
        if verts is None:
            return None
        normal = normals[cell_id]
        self.last_hit = {"verts": verts, "normal": normal,
                         "actor": source, "cell_id": cell_id}
        return position, normal

    def pick_candidates(self):
        sources = self.v._polygroup_actors()
        candidates = []
        actors = self.v._ren.GetActors()
        actors.InitTraversal()
        while True:
            actor = actors.GetNextActor()
            if actor is None:
                break
            if not actor.GetVisibility():
                continue
            if getattr(actor, "_polygroup_clone", False):
                source = getattr(actor, "_polygroup_source", None)
                cell_map = getattr(actor, "_polygroup_cell_map", None)
                if source is not None and cell_map is not None:
                    candidates.append((actor, source, cell_map))
                continue
            if any(actor is src for src in sources):
                candidates.append((actor, actor, None))
        return candidates

    def locator(self, actor):
        locator = self.locators.get(actor)
        if locator is not None:
            return locator
        mapper = actor.GetMapper()
        polydata = mapper.GetInput() if mapper is not None else None
        if polydata is None or polydata.GetNumberOfPolys() == 0:
            return None
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(polydata)
        locator.BuildLocator()
        self.locators[actor] = locator
        return locator

    @staticmethod
    def _actor_matrix(actor):
        matrix = actor.GetMatrix()
        if matrix is None:
            return None
        return np.array([[matrix.GetElement(r, c) for c in range(4)]
                         for r in range(4)])

    def world_to_mesh(self, actor, origin, direction):
        m = self._actor_matrix(actor)
        if m is None:
            return origin, direction
        try:
            inv = np.linalg.inv(m)
        except np.linalg.LinAlgError:
            return None, None
        o = inv @ np.append(origin, 1.0)
        d = inv[:3, :3] @ direction
        length = np.linalg.norm(d)
        if length < 1e-12:
            return None, None
        return o[:3], d / length

    def mesh_to_world(self, actor, point):
        m = self._actor_matrix(actor)
        p = np.append(np.asarray(point, dtype=float), 1.0)
        if m is not None:
            p = m @ p
        if abs(p[3]) < 1e-12:
            return None
        return p[:3] / p[3]

    def _stroke_locator(self, actor):
        snap = self._locator_snaps.get(actor)
        if snap is not None:
            return self.locators.get(actor)
        mapper = actor.GetMapper()
        polydata = mapper.GetInput() if mapper is not None else None
        if polydata is None or polydata.GetNumberOfPolys() == 0:
            return None
        snap = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.DeepCopy(polydata.GetPoints())
        snap.SetPoints(points)
        snap.SetPolys(polydata.GetPolys())   # shared; cells never change
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(snap)
        locator.BuildLocator()
        self.locators[actor] = locator
        self._locator_snaps[actor] = snap
        return locator

    def _exact_ray_cell_hit(self, actor, cell_id, origin, direction):
        verts = self.triangle_verts(actor, cell_id)
        if verts is None:
            return None
        e1 = verts[1] - verts[0]
        e2 = verts[2] - verts[0]
        pvec = np.cross(direction, e2)
        det = e1 @ pvec
        if abs(det) < 1e-12:
            return None
        inv_det = 1.0 / det
        tvec = origin - verts[0]
        u = (tvec @ pvec) * inv_det
        qvec = np.cross(tvec, e1)
        v = (direction @ qvec) * inv_det
        if u < -1e-9 or v < -1e-9 or u + v > 1.0 + 1e-9:
            return None
        t = (e2 @ qvec) * inv_det
        if t <= 1e-9:
            return None
        return origin + direction * t

    def stroke_pick(self, x, y):
        actor = self.stroke_actor
        if actor is None:
            return None
        ray = self.ray(*self.v._display_point(x, y))
        if ray is None:
            return None
        locator = self._stroke_locator(actor)
        if locator is None:
            return None
        origin, direction = ray
        mesh_origin, mesh_direction = self.world_to_mesh(actor,
                                                         origin, direction)
        if mesh_origin is None:
            return None
        _, far = self.v._ren.GetActiveCamera().GetClippingRange()
        hits = vtk.vtkPoints()
        cell_ids = vtk.vtkIdList()
        locator.IntersectWithLine(tuple(mesh_origin),
                                  tuple(mesh_origin + mesh_direction * far),
                                  1e-9, hits, cell_ids)
        count = cell_ids.GetNumberOfIds()
        if count == 0:
            return None
        # the locator does not guarantee near-to-far hit order, so scan
        # every intersection and keep the nearest visible cell
        visible = self.stroke_cells
        index = -1
        nearest = float("inf")
        for i in range(count):
            candidate = int(cell_ids.GetId(i))
            if visible is not None and not (
                    0 <= candidate < len(visible) and visible[candidate]):
                continue
            point = hits.GetPoint(i)
            along = ((point[0] - mesh_origin[0]) * mesh_direction[0]
                     + (point[1] - mesh_origin[1]) * mesh_direction[1]
                     + (point[2] - mesh_origin[2]) * mesh_direction[2])
            if along < nearest:
                nearest = along
                index = i
        if index < 0:
            return None
        cell_id = int(cell_ids.GetId(index))
        # exact hit on the live triangle; fall back to the snapshot point
        # when the moved surface dodged the ray for that triangle
        position = self._exact_ray_cell_hit(actor, cell_id,
                                            mesh_origin, mesh_direction)
        if position is None:
            position = np.asarray(hits.GetPoint(index))
        position = self.mesh_to_world(actor, position)
        normal = self.face_normal_world(actor, cell_id)
        if position is None or normal is None:
            return None
        return position, normal

    def face_normal_world(self, actor, cell_id):
        mapper = actor.GetMapper()
        polydata = mapper.GetInput() if mapper is not None else None
        if polydata is None or polydata.GetPoints() is None:
            return None
        ids = vtk.vtkIdList()
        polydata.GetCellPoints(cell_id, ids)
        if ids.GetNumberOfIds() < 3:
            return None
        points = polydata.GetPoints()
        local = np.array([points.GetPoint(ids.GetId(k)) for k in range(3)])
        normal = np.cross(local[1] - local[0], local[2] - local[0])
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            return None
        matrix = actor.GetMatrix()
        if matrix is not None:
            normal = np.array([[matrix.GetElement(r, c) for c in range(3)]
                               for r in range(3)]) @ (normal / length)
            length = float(np.linalg.norm(normal))
            if length < 1e-12:
                return None
        return normal / length

    def ray_hits_cached_triangle(self, x, y):
        entry = self.last_hit
        if entry is None:
            return None
        ray = self.ray(*self.v._display_point(x, y))
        if ray is None:
            return None
        origin, direction = ray
        v0, v1, v2 = entry["verts"]
        e1 = v1 - v0
        e2 = v2 - v0
        pvec = np.cross(direction, e2)
        det = e1 @ pvec
        if abs(det) < 1e-12:
            return None
        inv_det = 1.0 / det
        tvec = origin - v0
        u = (tvec @ pvec) * inv_det
        qvec = np.cross(tvec, e1)
        v = (direction @ qvec) * inv_det
        if u < -1e-9 or v < -1e-9 or u + v > 1.0 + 1e-9:
            return None
        t = (e2 @ qvec) * inv_det
        if t <= 1e-9:
            return None
        return origin + direction * t, entry["normal"]

    def ray(self, x_disp, y_disp):
        ren = self.v._ren
        near, far = None, None
        for z, out in ((0.0, "near"), (1.0, "far")):
            ren.SetDisplayPoint(x_disp, y_disp, z)
            ren.DisplayToWorld()
            w = ren.GetWorldPoint()
            if abs(w[3]) < 1e-12:
                return None
            if out == "near":
                near = np.asarray(w[:3]) / w[3]
            else:
                far = np.asarray(w[:3]) / w[3]
        direction = far - near
        length = np.linalg.norm(direction)
        if length < 1e-12:
            return None
        return near, direction / length

    def triangle_verts(self, actor, cell_id):
        mapper = actor.GetMapper()
        polydata = mapper.GetInput() if mapper is not None else None
        if polydata is None or polydata.GetPoints() is None:
            return None
        ids = vtk.vtkIdList()
        polydata.GetCellPoints(cell_id, ids)
        if ids.GetNumberOfIds() < 3:
            return None
        points = polydata.GetPoints()
        local = np.array([points.GetPoint(ids.GetId(k)) for k in range(3)])
        matrix = actor.GetMatrix()
        if matrix is None:
            return local
        m = np.array([[matrix.GetElement(r, c) for c in range(4)]
                      for r in range(4)])
        hom = np.vstack((local.T, np.ones(3)))     # points, not directions
        world = m @ hom
        return (world[:3] / world[3]).T

    def world_normals(self, actor):
        cached = self.normal_cache.get(actor)
        if cached is not None:
            return cached
        mapper = actor.GetMapper()
        polydata = mapper.GetInput() if mapper is not None else None
        if (polydata is None or polydata.GetPoints() is None
                or polydata.GetNumberOfPolys() == 0):
            return None
        points = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()))
        cells = polydata.GetPolys()
        count = cells.GetNumberOfCells()
        normals = np.zeros((count, 3), dtype=np.float64)
        conn = np.asarray(vtk_to_numpy(cells.GetConnectivityArray()))
        offsets = np.asarray(vtk_to_numpy(cells.GetOffsetsArray()))
        sizes = np.diff(offsets)
        if sizes.size == count and np.all(sizes == 3):
            tris = conn.reshape(count, 3)
            p0, p1, p2 = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
            normals = np.cross(p1 - p0, p2 - p0)
        else:
            # mixed polygon sizes: per-cell Newell fallback (still once)
            ids = vtk.vtkIdList()
            for i in range(count):
                polydata.GetCellPoints(i, ids)
                m = ids.GetNumberOfIds()
                if m < 3:
                    continue
                ring = points[[ids.GetId(k) for k in range(m)]]
                acc = np.zeros(3)
                for k in range(m):
                    acc = acc + np.cross(ring[k], ring[(k + 1) % m])
                normals[i] = acc
        # mesh space -> world space (importer node / alignment matrices)
        matrix = actor.GetMatrix()
        if matrix is not None:
            rot = np.array([[matrix.GetElement(r, c) for c in range(3)]
                            for r in range(3)])
            normals = normals @ rot.T
        lengths = np.linalg.norm(normals, axis=1)
        lengths[lengths < 1e-12] = 1.0
        normals /= lengths[:, None]
        self.normal_cache[actor] = normals
        return normals

    # -- hover ring normal ----------------------------------------------------
    def _ring_normal(self, actor, cell_id, position, face_normal):
        face = np.asarray(face_normal, dtype=np.float64)
        if actor is None or cell_id is None:
            return face
        cell_id = int(cell_id)
        cache = self.ring_normal_cache.get(actor)
        if cache is None:
            cache = {}
            self.ring_normal_cache[actor] = cache
        average = cache.get(cell_id)
        if average is None:
            average = self._neighbour_normal(actor, cell_id, face)
            if len(cache) >= _RING_NORMAL_CACHE:
                cache.clear()           # bounded; recomputing is cheap
            cache[cell_id] = average

        # orient toward the camera: the visible side is unambiguous, so a
        # winding flip cannot send the ring (or the blend below) through
        # a degenerate, vanished normal
        cam = self.v._ren.GetActiveCamera()
        if cam is not None:
            view = np.asarray(cam.GetPosition(), dtype=np.float64) - position
            if float(view @ average) < 0.0:
                average = -average

        previous = self._ring_normal_prev
        if previous is not None and previous[0] is actor:
            last = previous[1]
            if float(last @ average) > 0.0:
                average = last + (average - last) * _RING_NORMAL_LERP
                length = float(np.linalg.norm(average))
                if length > 1e-12:
                    average = average / length
                else:
                    average = last
            # else: crossed more than 90 degrees between picks; snap
            # instead of blending through zero (which would hide the ring)
        self._ring_normal_prev = (actor, average)
        return average

    def _neighbour_normal(self, actor, cell_id, face):
        normals = self.world_normals(actor)
        polydata = self.deformer._polydata(actor)
        if (normals is None or polydata is None
                or not 0 <= cell_id < len(normals)):
            return face
        topo = self.deformer._topology_of(polydata)
        offs = topo["offs"]
        if cell_id + 1 >= len(offs):
            return face
        verts = topo["conn"][int(offs[cell_id]):int(offs[cell_id + 1])]
        if verts.size:
            ptr = topo["vcell_ptr"]
            owner = topo["vcell_ids"]
            rings = [owner[int(ptr[v]):int(ptr[v + 1])] for v in verts]
            incident = np.unique(np.concatenate(rings))
            incident = incident[incident != cell_id]
            incident = incident[:_RING_NORMAL_SAMPLES - 1]
            cells = np.concatenate(([cell_id], incident))
        else:
            cells = np.array([cell_id], dtype=np.int64)
        samples = normals[cells]
        flip = (samples @ face < 0.0)[:, None]
        samples = np.where(flip, -samples, samples)
        weights = np.ones(len(cells))
        weights[0] = 2.0
        average = (samples * weights[:, None]).sum(axis=0)
        length = float(np.linalg.norm(average))
        if length < 1e-12:
            return face                  # exact cancellation: no info
        return average / length

    # -- strokes ------------------------------------------------------------
    def _visible_masks(self, actor):
        visible = self.v._polygroups.visible_groups
        if visible is None:
            return None, None
        groups = self.v._polygroups.cell_groups(actor)
        if groups is None or not len(groups):
            return None, None
        cells = np.isin(groups, np.asarray(visible))
        if cells.all():
            return None, None
        return self.deformer.vertex_mask_for_cells(actor, cells), cells

    def try_begin_stroke(self, x, y) -> bool:
        if self.v._current_path is None:
            return False
        hit = self.pick(x, y)
        actor = self.hit_actor
        if hit is None or actor is None:
            return False
        # Draw-mode camera pivot: strokes aim orbit/zoom at where they
        # started (viewer._orbit_pivot; F / model swap clears it)
        self.v._orbit_pivot = np.asarray(hit[0], dtype=float)
        allowed, cells = self._visible_masks(actor)
        self._setup_symmetry(actor)     # panel X/Y/Z -> mirror map
        if not self.deformer.begin_stroke(actor):
            return False
        self.stroke_allowed = allowed
        self.stroke_cells = cells
        # Pin the stroke-pick locator to a frozen snapshot BEFORE the first
        # dab marks the polydata modified (see _stroke_locator).
        self._stroke_locator(actor)
        self.stroke_actor = actor
        self.stroke_pos = hit[0]
        self.hit = hit
        self.stroke_moved = 0
        self.v._mode = "brush_smooth"   # drags keep smoothing until release
        # The press handler ran hide() before dispatching; while the stroke
        # owns the drag the ring stands in for the cursor again
        self.hide_mouse_cursor()
        self.dab(hit[0], hit[1])
        self.schedule_render()
        return True

    def press_begin_stroke(self, x, y) -> bool:
        claimed = bool(self.held and not self.stroking
                       and self.try_begin_stroke(x, y))
        trace("smooth.press", xy=(x, y), held=self.held,
              stroking=self.stroking, claimed=claimed,
              **self._trace_ring(),
              **cam_info(self.v._ren.GetActiveCamera()))
        return claimed

    def stroke_move(self, x, y):
        hit = self.stroke_pick(x, y)
        if hit is None:
            return
        position, normal = hit
        self.hit = (position, normal)
        radius = self.world_radius(position)
        spacing = max(radius * float(self.settings.spacing), 1e-12)
        seg = position - self.stroke_pos
        dist = float(np.linalg.norm(seg))
        while dist >= spacing:
            step = self.stroke_pos + seg * (spacing / dist)
            self.dab(step, normal)
            self.stroke_pos = step
            seg = position - self.stroke_pos
            dist = float(np.linalg.norm(seg))
        self.cursor.place(position, normal, radius,
                          smooth=_BRUSH_SMOOTHING,
                          nominal_px=self.radius_px)
        # mirrored preview follows the smooth stroke's next dab spot
        self._refresh_ghost(position, normal, self.stroke_actor)
        self.schedule_render()

    def dab(self, position, normal=None):
        if self.stroke_actor is None:
            return
        moved = self.deformer.dab(self.stroke_actor, position,
                                  self.world_radius(position),
                                  strength=float(self.settings.strength),
                                  inner_ratio=_SMOOTH_INNER_RATIO,
                                  inner_boost=_SMOOTH_INNER_BOOST,
                                  normal_world=normal,
                                  flatten=float(self.settings.flatten),
                                  flatten_depth=float(
                                      self.settings.flatten_depth),
                                  allowed_verts=self.stroke_allowed)
        if moved:
            self.stroke_moved += moved
            self.last_hit = None   # geometry moved under the cache

    def _refresh_after_stroke(self, actor, record):
        cache = self.normal_cache.get(actor)
        cells = record.get("cells") if record is not None else None
        if (cache is None or cells is None
                or len(cache) <= int(cells.max(initial=-1))):
            self.normal_cache.pop(actor, None)   # nothing to patch
        else:
            normals = np.asarray(record["cell_normals"], dtype=float)
            matrix = actor.GetMatrix()
            if matrix is not None:
                rot = np.array([[matrix.GetElement(r, c) for c in range(3)]
                                for r in range(3)])
                normals = normals @ rot.T
            lengths = np.linalg.norm(normals, axis=1)
            lengths[lengths < 1e-12] = 1.0
            cache[cells] = normals / lengths[:, None]
        # neighbourhood averages can include a touched cell; the next hover
        # rebuilds only the hit cell it lands on
        self.ring_normal_cache.pop(actor, None)
        if (self._ring_normal_prev is not None
                and self._ring_normal_prev[0] is actor):
            self._ring_normal_prev = None
        # Hover picks go through the filtered clone actors while polygroups
        # are hidden; their locators are built on the shared (now deformed)
        # points and are stale too, so drop and rebuild them with the source.
        stale = [actor]
        stale.extend(clone for clone in self.v._polygroups.clone_actors
                     if getattr(clone, "_polygroup_source", None) is actor)
        path = self.v._current_path
        for target in stale:
            self.locators.pop(target, None)   # buckets went stale
            self._locator_snaps.pop(target, None)
            QTimer.singleShot(0, lambda t=target: (
                self.v._current_path is path and self.locator(t)))

    def end_stroke(self):
        self.v._mode = None
        actor, self.stroke_actor = self.stroke_actor, None
        self.stroke_pos = None
        self.stroke_allowed = None
        self.stroke_cells = None
        self.deformer.symmetry = None   # symmetry is per-stroke only
        record = self.deformer.end_stroke()
        if actor is not None and record is not None:
            # one undo entry per stroke: exactly the vertices it touched
            self.v._history.push(VertexEditEntry(
                "Smooth brush", actor, record["ids"],
                record["old_positions"], record["new_positions"],
                record["old_normals"], record["new_normals"]))
        if actor is not None:
            self._refresh_after_stroke(actor, record)
        moved, self.stroke_moved = self.stroke_moved, 0
        if moved:
            self.v._ren.ResetCameraClippingRange()
            self.v.smoothEdited.emit()   # queue the edit for Bake changes
        # drop the ring's lerp history so the next hover doesn't visibly
        # finish catching up after a fast stroke
        self.cursor.snap()
        self.v.render_now()

    # -- move ("snake hook") strokes -------------------------------------------
    def _move_group_mask(self, actor):
        entry = self.last_hit or {}
        if entry.get("actor") is not actor:
            return None, None
        cell_id = entry.get("cell_id")
        groups = self.v._polygroups.cell_groups(actor)
        if (cell_id is None or groups is None
                or cell_id >= len(groups) or not groups.any()):
            return None, None
        group = int(groups[cell_id])
        return self.grabber.face_group_mask(actor, groups, group), group

    def press_begin_move(self, x, y) -> bool:
        if (not self.draw_enabled or self.shift
                or self.v._current_path is None):
            trace("move.press.reject", xy=(x, y), draw=self.draw_enabled,
                  shift=self.shift, filtered=self.v._polygroups.is_filtered,
                  path=self.v._current_path)
            return False
        hit = self.pick(x, y)
        actor = self.hit_actor
        if hit is None or actor is None:
            trace("move.press.miss", xy=(x, y), actor=actor)
            return False
        position, normal = hit
        visible, _cells = self._visible_masks(actor)
        if self.settings.apply_to_polygroup:
            group_mask, group = self._move_group_mask(actor)
            allowed = group_mask
            if allowed is not None and visible is not None:
                allowed = allowed & visible
            elif allowed is None:
                allowed = visible       # no group info: visible groups only
        else:
            allowed, group = visible, None
        trace("move.press", xy=(x, y),
              pick=tuple(round(v, 3) for v in position),
              normal=tuple(round(v, 3) for v in normal),
              pivot=vec3(self.v._orbit_pivot),
              group=group,
              filtered=self.v._polygroups.is_filtered,
              allowed=None if allowed is None else int(allowed.sum()),
              **self._trace_ring(),
              **cam_info(self.v._ren.GetActiveCamera()))
        # Draw-mode camera pivot: after release, orbit/zoom aims at the
        # point that was grabbed (viewer._orbit_pivot; F clears it)
        self.v._orbit_pivot = np.asarray(position, dtype=float)
        self._setup_symmetry(actor)     # panel X/Y/Z -> mirror map
        if not self.grabber.begin_stroke(actor, position,
                                         self.world_radius(position),
                                         sharpness=float(
                                             self.settings.hook_sharpness),
                                         allowed_verts=allowed):
            return False
        self.move_actor = actor
        self.move_center = np.asarray(position, dtype=float)
        self.move_cursor = np.asarray(position, dtype=float)  # ring anchor
        self.move_normal = np.asarray(normal, dtype=float)
        self.move_face_screen = self._face_screen_dir(position,
                                                      self.move_normal)
        self.move_last = (x, y)
        self.move_moved = 0
        self.hit = (position, normal)
        self.v._mode = "brush_move"
        self.hide_mouse_cursor()
        # exact placement (no temporal lerp) for the whole move stroke: the
        # ring tracks the dead-reckoned cursor anchor, which has no pick
        # jitter to filter - and a lerped ring would be left lagging behind
        # the cursor at release, so the first hover after the stroke read
        # as the ring jumping forward to catch up.
        if self.cursor.place(position, normal, self.world_radius(position),
                             nominal_px=self.radius_px):
            self.schedule_render()
        return True

    def screen_delta_world(self, dx, dy, at) -> np.ndarray:
        cam = self.v._ren.GetActiveCamera()
        d = np.asarray(cam.GetDirectionOfProjection(), dtype=float)
        up = np.asarray(cam.GetViewUp(), dtype=float)
        right = np.cross(d, up)
        length = float(np.linalg.norm(right))
        if length < 1e-12:
            return np.zeros(3)
        right /= length
        sx, sy = self.v._display_scale()
        world_per_px = self.world_radius(at) / self.radius_px
        return (right * (dx * sx * world_per_px)
                + up * (-dy * sy * world_per_px))

    def _face_screen_dir(self, position, normal):
        offset = self.world_radius(position) * 0.1
        p0 = np.asarray(self.project_to_display(position), dtype=float)
        p1 = np.asarray(self.project_to_display(
            [position[i] + normal[i] * offset for i in range(3)]), dtype=float)
        d = p1 - p0
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            return np.array([0.0, 1.0])    # screen up = outward
        return d / length

    def move_move(self, x, y, alt: bool = False):
        if self.move_actor is None or self.move_center is None:
            return
        dx = x - self.move_last[0]
        dy = y - self.move_last[1]
        self.move_last = (x, y)
        if (dx * dx + dy * dy) < _MIN_MOVE_PX * _MIN_MOVE_PX:
            return
        # full cursor motion, on the camera plane at the tip's depth -
        # the ring always tracks the real cursor position
        raw = self.screen_delta_world(dx, dy, self.move_center)
        self.move_cursor = self.move_cursor + raw
        if alt and self.move_normal is not None:
            # face normalized: signed drag component along the normal's
            # screen direction, applied along the world-space normal
            # (direction is in display space, y up; mouse dy is y down)
            dirx, diry = self.move_face_screen
            sx, sy = self.v._display_scale()
            world_per_px = (self.world_radius(self.move_center)
                            / self.radius_px)
            amount = (((dx * sx) * dirx + (-dy * sy) * diry)
                      * world_per_px * _MOVE_FOLLOW)
            delta = self.move_normal * amount
        else:
            # the tip follows only _MOVE_FOLLOW of the cursor motion
            delta = raw * _MOVE_FOLLOW
        moved = self.grabber.move(delta)
        if moved:
            self.move_moved += moved
            self.move_center = self.move_center + delta
            self.last_hit = None       # geometry moved under the cache
        if self.cursor.visible:
            # exact placement: see press_begin_move - no lerp, so the drawn
            # ring never trails the cursor during a grab stroke
            self.cursor.place(self.move_cursor, self.move_normal,
                              self.world_radius(self.move_cursor),
                              nominal_px=self.radius_px)
        # mirrored preview: the second grab centre is the reflection of
        # the dragged tip (reflection is affine, so it tracks exactly)
        actor = self.move_actor
        sym = self._active_symmetry(actor) if actor is not None else None
        if sym is not None and self.move_center is not None:
            ghost_pos = sym.reflect_point(self.move_center)
            self.cursor.place_ghost(ghost_pos,
                                    sym.reflect_dir(self.move_normal),
                                    self.world_radius(ghost_pos))
        else:
            self.cursor.hide_ghost()
        trace("move.move", xy=(x, y), d=(dx, dy), moved=moved,
              cursor=tuple(round(v, 3) for v in self.move_cursor),
              center=tuple(round(v, 3) for v in self.move_center),
              **self._trace_ring(),
              **cam_info(self.v._ren.GetActiveCamera()))
        self.schedule_render()

    def end_move(self):
        trace("move.end.enter", mode=self.v._mode,
              pivot=vec3(self.v._orbit_pivot),
              **self._trace_ring(),
              **cam_info(self.v._ren.GetActiveCamera()))
        self.v._mode = None
        actor, self.move_actor = self.move_actor, None
        self.move_center = None
        self.move_cursor = None
        self.move_normal = None
        self.move_face_screen = None
        self.move_last = None
        self.deformer.symmetry = None   # symmetry is per-stroke only
        record = self.grabber.end_stroke()
        if actor is not None and record is not None:
            # one undo entry per stroke: exactly the vertices it grabbed
            self.v._history.push(VertexEditEntry(
                "Move brush", actor, record["ids"],
                record["old_positions"], record["new_positions"],
                record["old_normals"], record["new_normals"]))
        if actor is not None:
            self._refresh_after_stroke(actor, record)
        moved, self.move_moved = self.move_moved, 0
        if moved:
            self.v._ren.ResetCameraClippingRange()
            self.v.smoothEdited.emit()   # queue the edit for Bake changes
        # leave the ring where the cursor left it (move_cursor already
        # tracks the mouse 1:1): teleporting it onto the lagging mesh tip
        # read as a backwards snap on release.  The temporal-lerp history
        # is deliberately kept (no cursor.snap() here): the next hover's
        # exact surface placement then eases in over a few frames instead
        # of landing in one frame, which read as a second jump.
        self.v.render_now()
        trace("move.end.done", moved=moved,
              pivot=vec3(self.v._orbit_pivot),
              **self._trace_ring(),
              **cam_info(self.v._ren.GetActiveCamera()))
