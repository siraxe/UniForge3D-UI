"""Native VTK 3D viewer widget for PySide6.

Loads GLB / glTF / OBJ / STL / PLY with full PBR materials (glTF) and embeds
as a native Windows widget - no browser involved.

Maya-style camera: left-drag tumble   shift+left horizon-locked horizontal
spin with a 90-degree compass magnet (Shift toggles live, even mid-drag)
middle-drag pan   right-drag / wheel dolly (zoom)

Smooth brush (see viewer_brush.py): in Draw mode (Outliner panel's toggle
or Q) the ring cursor is always on - over the mesh it hugs the surface,
over empty space it is a flat screen circle. Holding Shift while the
cursor is over the mesh activates the brush until Shift is released -
smoothing is only applied while the left button is held on the surface.
Over empty space Shift keeps its camera role (level horizon + turntable).
W / S resize the ring while Draw mode is on; holding Space (idle) opens
the brush-settings popup and releasing Space closes it.

Ctrl+Shift+left-drag draws a near-white lasso over the 3D view. On release,
every face whose projection touches it is added to the light selection (a
neutral colour that cannot be confused with the polygroup palette). Holding
Alt as well changes the lasso to red and subtracts the touched faces instead;
Alt can be pressed or released mid-drag, and the band recolours immediately.
Ctrl+Shift+left-click on empty space, or Ctrl+D anywhere, clears everything.
Ctrl+W paints the selected faces with the first free polygroup colour and
clears the selection so the result is visible. With nothing selected and the
whole mesh visible, Ctrl+W puts every face back in the uncoloured group 0;
with nothing selected while some groups are hidden, it paints every visible
face into a new group instead. Ctrl+Shift+W splits the visible faces (or the
lasso selection) into edge-connected islands and gives each island its own
polygroup colour, largest island first.

Ctrl+Shift+left-click (without dragging) on a face solos its polygroup: only
that group stays visible. While groups are already hidden, such a click adds
the clicked face's group to the hidden set instead. Clicking empty space while
filtered inverts the visible set, revealing exactly the hidden groups (this
works for any number of visible groups). A second visibility flip in a row
returns to all groups.
A drag that touches visible faces only selects them (hidden groups stay
hidden); an empty drag while filtered flips the view to the *other* groups
on release. These actions all take effect on release.

The footer's Split and save button walks the polygroups: it solos each one,
asks for a file-name suffix, and finally exports <model>_<suffix>.glb per
group (unpainted faces included as group 0).

Material swatches: four spheres rendered in an overlay viewport at the top
right. Clicking one applies its material preset to the loaded model:
  1 current/original   2 dark + specular + slight metal
  3 bronze (metallic)  4 original texture / attached textured mesh
                          (disabled when the model has none)

Backface culling (toolbar checkbox): hides polygons facing away from the
camera, which exposes double-sided sheets, flipped patches and holes.
"""

import math
from pathlib import Path

import numpy as np
import vtk
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut

import hotkeys
from debug_trace import cam_info, qt, trace, vec3
from vtk.util.numpy_support import (
    numpy_to_vtk,
    numpy_to_vtkIdTypeArray,
    vtk_to_numpy,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from tools.lasso_polygroups.lasso_select import (
    LASSO_COLOR,
    LASSO_SUBTRACT_COLOR,
    FaceSelection,
    LassoOverlay,
    _poly_cells,
    build_selection_actors,
    combine_selection_cells,
    faces_in_lasso,
)
from tools.lasso_polygroups.polygroups import (
    DEFAULT_POLYGROUP,
    PolygroupController,
    split_geometry,
    visible_geometry,
)
from tools.history import HistoryStack, StateEntry, VertexEditEntry
from viewer_brush import BrushController
from resym import ResymPicker

_BG_BOTTOM = (0x2b / 255, 0x2e / 255, 0x37 / 255)   # #2b2e37
_BG_TOP = (0x17 / 255, 0x18 / 255, 0x1e / 255)      # #17181e

#: max drag distance (px) for a reSym left-press to still count as a
#: point-picking click instead of a camera orbit
_RESYM_CLICK_PX = 5.0

# glTF texture slots the importer assigns on vtkProperty
_TEXTURE_SLOTS = ("albedoTex", "normalTex", "rmTex", "aoTex", "emissiveTex")

# material presets applied to the model: (color, metallic, roughness)
_PRESETS = {
    "dark_specular": ((0.41, 0.43, 0.46), 0.35, 0.45),   # matte-ish dark
    "bronze": ((1.00, 0.72, 0.44), 0.70, 0.35),          # less metallic
}
_SWATCH_LOOKS = {                       # how each ball itself looks
    "original": ((0.80, 0.80, 0.80), 0.0, 0.45),
    "bronze": ((1.00, 0.72, 0.44), 0.70, 0.35),
    "dark_specular": ((0.41, 0.43, 0.46), 0.35, 0.45),
    "textured": ((0.55, 0.55, 0.55), 0.0, 0.60),   # + albedo texture
}
_SWATCH_ORDER = ("dark_specular", "bronze", "original", "textured")
_SWATCH_RADIUS = 0.19                  # world units, ball radius
_SWATCH_SPACING = 0.60                 # world units, center-to-center

# Plain keys VTK's stock interactor style reacts to (wireframe, surface,
# reset, pick, exit, ...); see keyPressEvent.
_VTK_STYLE_KEYS = frozenset((
    Qt.Key_W, Qt.Key_S, Qt.Key_R, Qt.Key_P, Qt.Key_Q, Qt.Key_E,
    Qt.Key_F, Qt.Key_3, Qt.Key_U, Qt.Key_L, Qt.Key_J,
))


def file_bounds(path):
    """World-space bounds of a model file as (xmin, xmax, ymin, ymax,
    zmin, zmax), or None if it cannot be read.

    Used to bring the refined mesh into the coarse mesh's coordinate frame so
    the before/after toggle switches in place instead of jumping to a
    different scale. Trimesh is already a dependency (only imported lazily
    here, so the UI stays light until a comparison pair is opened).
    """
    try:
        import trimesh
        mesh = trimesh.load(str(path), force="mesh", merge_primitives=True)
        b = mesh.bounds
        if b is None:
            return None
        (x0, y0, z0), (x1, y1, z1) = b
        return (float(x0), float(x1), float(y0), float(y1),
                float(z0), float(z1))
    except Exception:
        return None


def _face_position_keys(vertices, faces):
    """Order-independent per-face keys built from float32 positions.

    Returns ``(keys, owner_rows)``: one key per (fan) triangle, plus the
    face row each key came from - quads contribute two keys (their fan
    triangles), so a quad survives the triangulation a GLB write applies.
    """
    corners = np.asarray(vertices, dtype=np.float32)[
        np.asarray(faces, dtype=np.int64)]
    rows = np.arange(len(faces), dtype=np.int64)
    if corners.ndim == 3 and corners.shape[1] == 4:
        corners = np.concatenate([corners[:, (0, 1, 2)],
                                  corners[:, (0, 2, 3)]])
        rows = np.concatenate([rows, rows])
    corners = np.sort(corners, axis=1)   # face identity ignores winding
    flat = np.ascontiguousarray(corners).reshape(len(corners), -1)
    return [row.tobytes() for row in flat], rows


class ModelViewer(QVTKRenderWindowInteractor):
    """Render-window widget; call load() with any supported model file."""

    SUPPORTED = {".glb", ".gltf", ".obj", ".stl", ".ply"}

    #: Emitted after a smooth-brush stroke (or an undo) changed the mesh,
    #: so the app can queue the edited geometry for Bake changes.
    smoothEdited = Signal()

    #: One-line status for the log panel (message, level).
    notice = Signal(str, str)

    #: Draw mode switched from inside the viewer (model swap turns it
    #: off); the Outliner panel syncs its Draw button to this.
    drawModeChanged = Signal(bool)

    #: Shift-drag turntable: while the free drag angle is within this many
    #: degrees of a multiple of 90 the camera is held exactly on that
    #: compass heading, so it does not detach easily.
    _TT_SNAP_ZONE = 20.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ren = vtk.vtkRenderer()
        self.GetRenderWindow().AddRenderer(self._ren)
        self._ren.SetBackground(*_BG_BOTTOM)     # gradient: bottom #2b2e37
        self._ren.SetBackground2(*_BG_TOP)       #            top    #17181e
        self._ren.GradientBackgroundOn()

        # Maya-style camera rig, implemented entirely at the Qt level (see
        # mouse*Event). The VTK interactor never receives mouse events, so
        # the stock trackball state machinery cannot interfere.
        # None|swatch|turntable|tumble|pan|dolly|lasso
        self._mode = None
        self._last = (0, 0)
        self._lasso_cursor = None
        self._lasso_subtract = False
        self._lasso_press_group = None    # polygroup under the press point
        self._polygroup_flip_streak = 0   # consecutive visibility flips

        self._current_path = None
        self._model_actors = []          # vtkActor list of the loaded model
        self._originals = []             # per-actor saved material state
        self._texture_actors = []        # textured variant for the 4th swatch
        self._texture_originals = []     # material state of that variant
        self._has_textures = False
        self._preset = "original"
        self._backface_cull = True

        # gray transparent "ghost" overlay (middle-click in the Outliner
        # list): a reference mesh shown next to the loaded one purely for
        # visual fit comparison; it is never picked, styled or exported
        self._ghost_path = None
        self._ghost_actors = []
        self._ghost_visible = True

        # lasso face selection (tools/lasso_polygroups/lasso_select.py)
        self._lasso = None               # LassoOverlay, rebuilt by clear()
        self._selection_actors = []      # selection highlight actors
        self._selection_sources = []     # source actor per highlight actor
        self._selected_cells = {}        # actor -> selected mesh cell ids
        self._selected_faces = []        # FaceSelection list
        self._selection_count = 0

        # persistent per-face polygroups (see polygroups.py)
        self._polygroups = PolygroupController(
            self._ren, self._backface_cull,
            on_changed=self._polygroups_changed,
            on_visibility_changed=self._polygroup_visibility_changed,
            actors_provider=self._polygroup_actors)

        # top-left overlay: file name + vertex/triangle counts in small
        # gray text; vtkCornerAnnotation keeps it pinned across resizes
        self._info = vtk.vtkCornerAnnotation()
        self._info.SetMinimumFontSize(9)
        self._info.SetMaximumFontSize(11)
        self._info.SetNonlinearFontScaleFactor(1.0)
        prop = self._info.GetTextProperty()
        prop.SetColor(0.62, 0.63, 0.66)
        prop.SetOpacity(0.9)
        self._info.SetText(vtk.vtkCornerAnnotation.UpperLeft, "")
        self._ren.AddViewProp(self._info)

        self._build_swatches()
        self._picker = vtk.vtkPropPicker()
        self._lasso = LassoOverlay(self._ren)

        # Smooth brush: everything ring/pick/stroke related lives in
        # BrushController (viewer_brush.py).  In Draw mode the ring is the
        # standing cursor (flat circle over empty space); holding Shift
        # while hovering the mesh activates the brush (W / S resize it);
        # smoothing itself always needs a left-drag on the surface.
        self.brush = BrushController(self)
        self._mouse_pos = (0, 0)          # last seen cursor position

        # reSym wizard: 7-click symmetry alignment (panels/right.py's
        # reSym button starts it; clicks place points, drags orbit)
        self.resym = ResymPicker(self)
        self._resym_press = None         # press pos while orbiting

        # Draw-mode orbit pivot: the world point of the last smooth stroke
        # or plain left-click on the mesh (see mousePressEvent /
        # BrushController.try_begin_stroke).  While set, orbit/zoom revolve
        # around it instead of the focal point; F (reset_camera) clears it.
        self._orbit_pivot = None

        self._history = HistoryStack()    # Ctrl+Z (tools/history.py)

        # F = frame/reset the view (works anywhere in the app)
        QShortcut(QKeySequence(hotkeys.seq("reset_view")), self,
                  activated=self.reset_camera,
                  context=Qt.ApplicationShortcut)
        # Ctrl+D = clear the lasso face selection from anywhere in the app.
        QShortcut(QKeySequence(hotkeys.seq("clear_selection")), self,
                  activated=self.clear_selection,
                  context=Qt.ApplicationShortcut)
        # Ctrl+W = paint the selected faces with the next free polygroup.
        QShortcut(QKeySequence(hotkeys.seq("assign_polygroup")), self,
                  activated=self.assign_polygroup,
                  context=Qt.ApplicationShortcut)
        # Ctrl+Shift+W = one polygroup per connected island of visible faces.
        QShortcut(QKeySequence(hotkeys.seq("assign_polygroup_islands")), self,
                  activated=self.assign_polygroup_islands,
                  context=Qt.ApplicationShortcut)
        # Ctrl+Z = undo the last smooth-brush stroke.
        QShortcut(QKeySequence(hotkeys.seq("undo")), self,
                  activated=self.undo_last_stroke,
                  context=Qt.ApplicationShortcut)
        # Ctrl+Shift+Z = redo the last undone smooth-brush stroke.
        QShortcut(QKeySequence(hotkeys.seq("redo")), self,
                  activated=self.redo_last_stroke,
                  context=Qt.ApplicationShortcut)
        # Q = toggle Draw mode (the Outliner panel's square Draw button).
        QShortcut(QKeySequence(hotkeys.seq("draw_mode")), self,
                  activated=self.toggle_draw_mode,
                  context=Qt.ApplicationShortcut)
        # X = toggle brush symmetry (the Edit mesh panel's Symmetry switch).
        QShortcut(QKeySequence(hotkeys.seq("symmetry_toggle")), self,
                  activated=self.toggle_symmetry,
                  context=Qt.ApplicationShortcut)

        self.setFocusPolicy(Qt.StrongFocus)
        self.Initialize()
        self.Start()

    # ------------------------------------------------------------------
    @property
    def current_path(self) -> Path | None:
        return self._current_path

    def toggle_draw_mode(self):
        """Q hotkey: flip the Outliner panel's Draw toggle.

        Going through the button (instead of the brush gate directly) keeps
        the check state, styling and ``_on_draw_toggled`` forwarding in sync.
        """
        btn = self.window().outliner.draw_button
        btn.setChecked(not btn.isChecked())

    def toggle_symmetry(self):
        """X hotkey: flip the Edit mesh panel's Symmetry switch.

        Going through the checkbox keeps the panel UI and the brush gate
        (BrushController._setup_symmetry reads it at stroke start) in
        sync; the axis checkboxes decide which planes mirror.
        """
        panel = self.window().right_panel
        panel.symmetry_enable.setChecked(
            not panel.symmetry_enable.isChecked())

    def retarget_path(self, old_path, new_path) -> bool:
        """Keep the loaded geometry but report it under *new_path*.

        Used when the displayed file is renamed on disk.  Returns True when
        the viewer was actually showing *old_path*.
        """
        if self._current_path is None:
            return False
        old = Path(old_path)
        try:
            same = self._current_path.resolve() == old.resolve()
        except OSError:
            same = self._current_path == old
        if same:
            self._current_path = Path(new_path)
        return same

    @property
    def has_textures(self) -> bool:
        return self._has_textures

    # ------------------------------------------------------------------
    # ghost overlay (middle-click in the Outliner list)
    @property
    def ghost_path(self) -> Path | None:
        return self._ghost_path

    @property
    def ghost_visible(self) -> bool:
        """Whether the ghost overlay actors are currently drawn."""
        return self._ghost_visible

    def set_ghost_visible(self, visible: bool):
        """Show/hide the ghost overlay without dropping the selection.

        Used by the Ctrl+1 shortcut: the ghost stays picked (reproject can
        still use it, the Outliner dot stays on) while the 3D overlay itself
        is hidden."""
        visible = bool(visible)
        if self._ghost_visible == visible:
            return
        self._ghost_visible = visible
        for actor in self._ghost_actors:
            actor.SetVisibility(visible)
        self.render_now()

    def toggle_ghost(self, path) -> bool:
        """Middle-click action: overlay *path* as a gray transparent ghost.

        Clicking the mesh that is already the ghost removes it instead
        (that case returns False).  The ghost is display-only: not
        pickable, excluded from material presets, camera fits and every
        file-based operation."""
        path = Path(path)
        if self._ghost_path is not None:
            try:
                same = (self._ghost_path.resolve() == path.resolve())
            except OSError:
                same = self._ghost_path == path
            if same:
                self._remove_ghost()
                self.render_now()
                return False
        actors = self._read_reference_actors(path)
        if not actors:
            raise RuntimeError(f"VTK could not read {path.name}")
        self._remove_ghost()
        for actor in actors:
            prop = actor.GetProperty()
            prop.SetColor(0.58, 0.60, 0.63)      # grayish
            prop.SetOpacity(0.35)                # transparent
            prop.SetLighting(True)
            prop.SetInterpolationToGouraud()
            actor.SetPickable(False)             # never blocks lasso/picking
            actor.SetProperty(prop)
            actor._ghost = True
            self._ren.AddActor(actor)
        self._ghost_actors = actors
        self._ghost_path = path
        self._ghost_visible = True    # a fresh ghost starts visible
        self.render_now()
        return True

    def _remove_ghost(self):
        """Drop the ghost overlay (actors and bookkeeping)."""
        for actor in self._ghost_actors:
            self._ren.RemoveActor(actor)
        self._ghost_actors = []
        self._ghost_path = None
        self._ghost_visible = True

    def _rehook_ghost(self):
        """Re-add the ghost after a load/clear wiped the renderer's props.

        The actors survive RemoveAllViewProps as Python objects, so the
        ghost persists across model switches until it is toggled off or
        the scene is closed."""
        for actor in self._ghost_actors:
            self._ren.AddActor(actor)

    def _read_reference_actors(self, path: Path) -> list:
        """Read *path* into plain, unstyled actors (ghost overlay source)."""
        path = Path(path)
        ext = path.suffix.lower()
        try:
            if ext in (".glb", ".gltf"):
                return self._read_gltf_actors(path)
            if ext == ".obj":
                importer = vtk.vtkOBJImporter()
                importer.SetFileName(str(path))
                importer.Update()
                return self._collect_importer_actors(importer)
            if ext == ".stl":
                reader = vtk.vtkSTLReader()
            elif ext == ".ply":
                reader = vtk.vtkPLYReader()
            else:
                return []
            reader.SetFileName(str(path))
            reader.Update()
            polydata = reader.GetOutput()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            return [actor]
        except Exception:
            return []

    def clear(self):
        self.resym.reset()      # markers die with RemoveAllViewProps below
        self._resym_press = None
        self._cancel_lasso()
        # RemoveAllViewProps also drops the highlight actors, but reset the
        # bookkeeping so no stale faces carry over to the next model.
        self._selection_actors = []
        self._selection_sources = []
        self._selected_cells = {}
        self._selected_faces = []
        self._selection_count = 0
        self._lasso_subtract = False
        self._lasso_press_group = None
        self._polygroup_flip_streak = 0
        self._polygroups.clear()
        self._ren.RemoveAllViewProps()
        self._ren.AddViewProp(self._info)   # RemoveAllViewProps dropped it
        self._info.SetText(vtk.vtkCornerAnnotation.UpperLeft, "")
        self._current_path = None
        self._model_actors = []
        self._originals = []
        self._texture_actors = []
        self._texture_originals = []
        self._has_textures = False
        # NOTE: the ghost overlay deliberately survives clear()/model
        # switches - RemoveAllViewProps only unhooks the actors; the
        # ghost is re-added by _rehook_ghost() after the next load and
        # only a middle-click toggle (or replacement) removes it.
        # RemoveAllViewProps also removed the lasso overlay actor.
        self._lasso = LassoOverlay(self._ren)
        # ...and the brush cursor actors: re-add them hidden, and drop any
        # held-S state and cached face normals so the next model starts clean.
        self.brush.on_model_cleared()
        self._history = HistoryStack()    # strokes never cross model loads
        self._set_preset("dark_specular", rebuild_swatch_texture=True)
    def load(self, path, preserve_camera: bool = False,
             preserve_material: bool = False, align_bounds=None) -> bool:
        """Load a model file; returns True on success.

        With preserve_camera the existing view orientation/zoom is kept,
        which makes before/after toggling seamless. preserve_material keeps
        the currently selected swatch preset instead of resetting to the
        default dark material. align_bounds maps the loaded model uniformly
        into the given (xmin, xmax, ymin, ymax, zmin, zmax) box; the coarse
        source's bounds are used so the refined mesh appears at the same
        scale/position when toggling.
        """
        path = Path(path)
        ext = path.suffix.lower()
        had_model = self._current_path is not None
        camera_state = (self._camera_state()
                        if preserve_camera and had_model else None)
        preset = self._preset if preserve_material and had_model else None
        self.clear()
        try:
            if ext in (".glb", ".gltf"):
                if not self._load_gltf(path):
                    raise RuntimeError(f"VTK could not read {path.name}")
            elif ext == ".obj":
                importer = vtk.vtkOBJImporter()
                importer.SetFileName(str(path))
                importer.Update()
                self._adopt(importer)
            elif ext == ".stl":
                reader = vtk.vtkSTLReader()
                reader.SetFileName(str(path))
                reader.Update()
                self._add_polydata(reader.GetOutput())
            elif ext == ".ply":
                reader = vtk.vtkPLYReader()
                reader.SetFileName(str(path))
                reader.Update()
                self._add_polydata(reader.GetOutput())
            else:
                return False
        except Exception:
            self.clear()
            if camera_state is not None:
                self._restore_camera(camera_state)
            self.render_now()
            return False

        self._current_path = path
        self._update_swatch_texture()    # also snapshots _model_actors
        if align_bounds is not None:
            self._align_model(align_bounds)
        if camera_state is not None:
            self._restore_camera(camera_state)
        else:
            self.reset_camera()
        if preset is not None:
            # "textured" needs an albedo texture on the new mesh; fall back
            # to its original material when the refined mesh has none.
            self._set_preset("original"
                             if preset == "textured" and not self._has_textures
                             else preset)
        else:
            self._set_preset("dark_specular")   # 1st swatch = default
        self._apply_backface_culling()   # re-apply on the freshly loaded actors
        self._rehook_ghost()             # ghost survives model switches
        self._fit_swatches()             # fresh camera scale for this window size
        self._update_info_text()
        self.render_now()
        return True

    def load_arrays(self, vertices, faces, *, preserve_camera: bool = False,
                    preserve_material: bool = False,
                    display_path=None) -> bool:
        """Display in-memory triangle or quad geometry; no file is written.

        ``display_path`` is what :attr:`current_path` reports (the file the
        geometry came from), so Show buttons and the repair/refine actions
        keep referring to that model even though the viewer shows edited
        vertices.  Materials/UVs are not available on this path; the normal
        material presets still work.
        """
        v = np.ascontiguousarray(vertices, dtype=np.float64)
        f = np.ascontiguousarray(faces, dtype=np.int64)
        if not len(v) or not len(f):
            return False

        had_model = self._current_path is not None
        camera_state = (self._camera_state()
                        if preserve_camera and had_model else None)
        preset = self._preset if preserve_material and had_model else None
        self.clear()

        polydata = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(v, deep=True))
        polydata.SetPoints(points)
        cells = vtk.vtkCellArray()
        stride = f.shape[1] if f.ndim == 2 and f.shape[1] in (3, 4) else 3
        offsets = np.arange(0, f.size + 1, stride, dtype=np.int64)
        cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                      numpy_to_vtkIdTypeArray(f.ravel(), deep=True))
        polydata.SetPolys(cells)
        self._add_polydata(polydata)

        self._current_path = Path(display_path) if display_path else None
        self._update_swatch_texture()
        if camera_state is not None:
            self._restore_camera(camera_state)
        else:
            self.reset_camera()
        if preset is not None:
            self._set_preset(preset)
        else:
            self._set_preset("dark_specular")
        self._apply_backface_culling()
        self._rehook_ghost()             # ghost survives model switches
        self._fit_swatches()
        self._update_info_text()
        self.render_now()
        return True

    # ------------------------------------------------------------------
    def set_backface_culling(self, enabled: bool) -> None:
        """Toggle back-face culling on the loaded model.

        With culling on, polygons whose normals point away from the camera
        are not drawn. A cleanly closed, consistently oriented mesh looks
        unchanged (the far side's front faces still cover it), while
        double-sided sheets, flipped patches and holes lose their back
        faces and become see-through - an easy way to spot them.
        """
        self._backface_cull = bool(enabled)
        self._apply_backface_culling()
        self.render_now()

    def _apply_backface_culling(self):
        for actor in (*self._model_actors, *self._texture_actors,
                      *self._selection_actors):
            prop = actor.GetProperty()
            if self._backface_cull:
                prop.BackfaceCullingOn()
            else:
                prop.BackfaceCullingOff()
        self._polygroups.set_backface_culling(self._backface_cull)

    def _load_gltf(self, path: Path) -> bool:
        actors = self._read_gltf_actors(path)
        if not actors:
            return False
        for actor in actors:
            self._ren.AddActor(actor)
        return True
    def _read_gltf_actors(self, path: Path) -> list:
        """Actors read from a GLB/glTF, detached from the importer.

        Falls back to a trimesh re-export when VTK silently yields nothing
        (e.g. EXT_texture_webp on older outputs).
        """
        actors = self._collect_importer_actors(self._gltf_importer(path))
        if actors:
            return actors
        temp = self._export_via_trimesh(path)
        if temp is None:
            return []
        try:
            return self._collect_importer_actors(self._gltf_importer(temp))
        finally:
            try:  # VTK may still hold the file open briefly
                temp.unlink(missing_ok=True)
            except PermissionError:
                pass
    @staticmethod
    def _gltf_importer(path: Path):
        importer = vtk.vtkGLTFImporter()
        importer.SetFileName(str(path))
        importer.Update()
        return importer
    @staticmethod
    def _collect_importer_actors(importer) -> list:
        src = importer.GetRenderer()
        actors = src.GetActors()
        actors.InitTraversal()
        found = []
        while True:
            actor = actors.GetNextActor()
            if actor is None:
                break
            src.RemoveActor(actor)
            found.append(actor)
        return found
    @staticmethod
    def _export_via_trimesh(path: Path) -> Path | None:
        import os
        import tempfile

        try:
            import trimesh
            scene = trimesh.load(path, force="scene")
            fd, name = tempfile.mkstemp(prefix=".u3d_view_", suffix=".glb")
            os.close(fd)
            temp = Path(name)
            scene.export(temp, file_type="glb", extension_webp=False)
            return temp
        except Exception:
            return None
    def _adopt(self, importer):
        """Move the actors an importer built into our renderer."""
        actors = self._collect_importer_actors(importer)
        if not actors:
            raise RuntimeError("importer produced no actors")
        for actor in actors:
            self._ren.AddActor(actor)
    def _add_polydata(self, polydata):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        self._ren.AddActor(actor)
    def _camera_state(self):
        camera = self._ren.GetActiveCamera()
        return {
            "position": camera.GetPosition(),
            "focal_point": camera.GetFocalPoint(),
            "view_up": camera.GetViewUp(),
            "clipping_range": camera.GetClippingRange(),
            "parallel_scale": camera.GetParallelScale(),
            "parallel_projection": camera.GetParallelProjection(),
            "view_angle": camera.GetViewAngle(),
        }

    def _restore_camera(self, state):
        camera = self._ren.GetActiveCamera()
        camera.SetPosition(state["position"])
        camera.SetFocalPoint(state["focal_point"])
        camera.SetViewUp(state["view_up"])
        camera.SetClippingRange(state["clipping_range"])
        camera.SetParallelScale(state["parallel_scale"])
        camera.SetParallelProjection(state["parallel_projection"])
        camera.SetViewAngle(state["view_angle"])
        # Fit near/far planes to the new mesh without moving the camera.
        self._ren.ResetCameraClippingRange()

    @staticmethod
    def _actors_bounds(actors):
        if not actors:
            return None
        lo = [float("inf")] * 3
        hi = [float("-inf")] * 3
        for actor in actors:
            b = actor.GetBounds()
            for i in range(3):
                lo[i] = min(lo[i], b[2 * i])
                hi[i] = max(hi[i], b[2 * i + 1])
        if not all(math.isfinite(v) for v in lo + hi):
            return None
        return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])

    def model_bounds(self):
        """Current world-space bounds of the primary model actors."""
        return self._actors_bounds(self._model_actors)

    def _mesh_stats(self):
        """Total (vertices, triangles) across the primary model actors."""
        verts = tris = 0
        for actor in self._model_actors:
            mapper = actor.GetMapper()
            if mapper is None:
                continue
            pd = mapper.GetInput()
            if pd is None:
                continue
            verts += pd.GetNumberOfPoints()
            tris += pd.GetNumberOfPolys()
        return verts, tris

    def _update_info_text(self):
        """Refresh the top-left overlay (name / verts / faces) or hide it."""
        if self._current_path is None:
            self._info.SetText(vtk.vtkCornerAnnotation.UpperLeft, "")
            return
        verts, faces = self._mesh_stats()
        text = (f"{self._current_path.name}\n"
                f"  verts: {verts:,}\n"
                f"  tris: {faces:,}")
        if self._selection_count:
            text += f"\n  sel: {self._selection_count:,}"
        if self._polygroups.used_count:
            text += (f"\n  groups: {self._polygroups.used_count}"
                     f"/{self._polygroups.capacity}")
        groups = self._polygroups.visible_groups
        if groups is not None:
            if len(groups) == 1:
                label = ("default" if groups[0] == DEFAULT_POLYGROUP
                         else f"group {groups[0]}")
                text += f"\n  solo: {label}"
            else:
                text += f"\n  visible groups: {len(groups)}"
        self._info.SetText(vtk.vtkCornerAnnotation.UpperLeft, text)

    def _align_model(self, target_bounds):
        self._align_actors(self._model_actors, target_bounds)

    def _align_actors(self, actors, target_bounds):
        """Uniformly scale/translate actors into target_bounds.

        A user matrix applies the adjustment in world space and preserves
        any node transforms the importer set on actors.
        """
        cur = self._actors_bounds(actors)
        if cur is None or target_bounds is None:
            return
        cur_ext = max(cur[1] - cur[0], cur[3] - cur[2], cur[5] - cur[4])
        tgt_ext = max(target_bounds[1] - target_bounds[0],
                      target_bounds[3] - target_bounds[2],
                      target_bounds[5] - target_bounds[4])
        if cur_ext <= 0 or tgt_ext <= 0:
            return
        scale = tgt_ext / cur_ext
        tx = ((target_bounds[0] + target_bounds[1]) / 2
              - scale * (cur[0] + cur[1]) / 2)
        ty = ((target_bounds[2] + target_bounds[3]) / 2
              - scale * (cur[2] + cur[3]) / 2)
        tz = ((target_bounds[4] + target_bounds[5]) / 2
              - scale * (cur[4] + cur[5]) / 2)
        matrix = vtk.vtkMatrix4x4()
        matrix.SetElement(0, 0, scale)
        matrix.SetElement(1, 1, scale)
        matrix.SetElement(2, 2, scale)
        matrix.SetElement(0, 3, tx)
        matrix.SetElement(1, 3, ty)
        matrix.SetElement(2, 3, tz)
        for actor in actors:
            actor.SetUserMatrix(matrix)
        self._ren.ResetCameraClippingRange()

    def reset_camera(self):
        self._orbit_pivot = None    # F also forgets a Draw-mode pivot
        self._ren.ResetCamera()
        self._ren.ResetCameraClippingRange()
        self.render_now()

    def toggle_projection(self) -> str:
        """Switch between perspective and orthographic and return the new mode.

        Both modes share one zoom value, so switching never changes the
        apparent size of the model: the view angle (perspective) and the
        parallel scale (orthographic) are converted into each other via
        ``parallel_scale = distance * tan(view_angle / 2)``, which is the
        half-height of the view frustum at the focal point in both cases.
        """
        cam = self._ren.GetActiveCamera()
        if cam.GetParallelProjection():
            distance = cam.GetDistance()
            half = (math.atan(cam.GetParallelScale() / distance)
                    if distance > 1e-9 else math.radians(30.0))
            cam.ParallelProjectionOff()
            cam.SetViewAngle(max(2.0, min(150.0,
                2.0 * math.degrees(half))))
        else:
            half = math.radians(cam.GetViewAngle()) * 0.5
            cam.ParallelProjectionOn()
            cam.SetParallelScale(cam.GetDistance() * math.tan(half))
        self._ren.ResetCameraClippingRange()
        self.render_now()
        return "Orthographic" if cam.GetParallelProjection() else "Perspective"

    def render_now(self):
        self.GetRenderWindow().Render()

    # -- swatch layout helpers --------------------------------------------
    def _fit_swatches(self):
        """Keep the swatch row fully visible and hugging the top edge."""
        size = self._swatch_ren.GetSize()
        if size[0] <= 0 or size[1] <= 0:
            return                       # not laid out yet
        aspect = size[0] / max(1.0, float(size[1]))
        half_span = _SWATCH_SPACING * 1.5 + _SWATCH_RADIUS
        scale = max(_SWATCH_RADIUS * 1.45, half_span / aspect)
        cam = self._swatch_ren.GetActiveCamera()
        cam.SetParallelScale(scale)
        # Center the view just above the ball row so the balls sit near the
        # viewport's top edge instead of floating in the vertical middle
        # (a small margin of 0.35 radii stays as breathing room).
        pad = _SWATCH_RADIUS * 0.35
        cy = _SWATCH_RADIUS + pad - scale
        cam.SetFocalPoint(0.0, cy, 0.0)
        cam.SetPosition(0.0, cy, 5.0)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_swatch_ren"):
            self._fit_swatches()
            self.render_now()

    def showEvent(self, event):
        # The VTK render window only syncs its pixel size to the widget once
        # shown; without this the swatch balls render with a stale camera
        # scale (too big) until the user resizes the window.
        super().showEvent(event)
        if hasattr(self, "_swatch_ren"):
            self._fit_swatches()
            self.render_now()
            # Render once more after the event loop has mapped the window:
            # the very first frame otherwise stays black (gradient + swatches
            # missing) until some other event forces a redraw.
            QTimer.singleShot(0, self._initial_render)

    def _initial_render(self):
        if not self.isVisible():
            return
        self._fit_swatches()
        self.render_now()

    # -- material presets --------------------------------------------------
    @staticmethod
    def _snapshot_actor(actor) -> dict:
        prop = actor.GetProperty()
        textures = {}
        for slot in _TEXTURE_SLOTS:
            tex = prop.GetTexture(slot)
            if tex is not None:
                textures[slot] = tex
        return {
            "actor": actor,
            "color": prop.GetColor(),
            "metallic": prop.GetMetallic(),
            "roughness": prop.GetRoughness(),
            "interpolation": prop.GetInterpolation(),
            "textures": textures,
        }
    @staticmethod
    def _restore_material(entry):
        prop = entry["actor"].GetProperty()
        prop.SetColor(*entry["color"])
        prop.SetMetallic(entry["metallic"])
        prop.SetRoughness(entry["roughness"])
        prop.SetInterpolation(entry["interpolation"])
        for slot, tex in entry["textures"].items():
            if hasattr(prop, "RemoveTexture"):
                prop.RemoveTexture(slot)     # replaces without a VTK warning
            prop.SetTexture(slot, tex)
    def _snapshot_originals(self):
        """Remember each primary actor's material so presets can restore it."""
        self._model_actors = []
        self._originals = []
        self._has_textures = False
        actors = self._ren.GetActors()
        actors.InitTraversal()
        while True:
            actor = actors.GetNextActor()
            if actor is None:
                break
            if any(actor is alt for alt in self._texture_actors):
                continue        # textured variant lives in its own list
            if any(actor is alt for alt in self._selection_actors):
                continue        # selection highlight is not model geometry
            if any(actor is alt for alt in self._polygroups.actors):
                continue        # polygroup colour is not model geometry
            if getattr(actor, "_polygroup_clone", False):
                continue        # filtered stand-in, not model geometry
            if getattr(actor, "_ghost", False):
                continue        # ghost overlay is not model geometry
            if getattr(actor, "_brush_cursor", False):
                continue        # brush cursor is not model geometry
            self._model_actors.append(actor)
            entry = self._snapshot_actor(actor)
            if entry["textures"]:
                self._has_textures = True
            self._originals.append(entry)
    def _set_preset(self, name: str, rebuild_swatch_texture: bool = False):
        """Apply a material preset (or restore original) to the model.

        When a textured variant exists, the 4th swatch shows that alternate
        mesh and the first three keep operating on the primary geometry.
        """
        self._preset = name
        use_textured = name == "textured" and bool(self._texture_actors)
        for entry in self._originals:
            entry["actor"].SetVisibility(not use_textured)
        for entry in self._texture_originals:
            entry["actor"].SetVisibility(use_textured)
        self._sync_selection_visibility()
        self._polygroups.sync_visibility()

        if use_textured:
            for entry in self._texture_originals:
                self._restore_material(entry)
        else:
            restore = name in ("original", "textured")
            for entry in self._originals:
                if restore:
                    self._restore_material(entry)
                    continue
                prop = entry["actor"].GetProperty()
                color, metallic, roughness = _PRESETS[name]
                for slot in entry["textures"]:
                    if hasattr(prop, "RemoveTexture"):
                        prop.RemoveTexture(slot)
                prop.SetInterpolationToPBR()
                prop.SetColor(*color)
                prop.SetMetallic(metallic)
                prop.SetRoughness(roughness)
        if rebuild_swatch_texture:
            self._refresh_textured_swatch()
        self.render_now()

    def add_textured(self, path) -> bool:
        """Attach a textured variant, shown only by the 4th swatch.

        Geometry-only GLBs carry no UVs, so texture generation remeshes and
        UV-unwraps the result. Keep it as an alternate actor set instead of
        replacing the primary mesh: the first three presets then keep their
        exact materials, and the 4th swatch displays this textured mesh.
        """
        if not self._model_actors:
            return False
        path = Path(path)
        if path.suffix.lower() not in (".glb", ".gltf"):
            return False
        try:
            actors = self._read_gltf_actors(path)
        except Exception:
            return False
        if not actors:
            return False

        target = self.model_bounds()
        self._remove_textured()
        self._texture_actors = actors
        self._texture_originals = [self._snapshot_actor(a) for a in actors]
        for actor in actors:
            self._ren.AddActor(actor)
            actor.SetVisibility(False)
        self._align_actors(actors, target)
        self._has_textures = True
        self._apply_backface_culling()
        self._refresh_textured_swatch()
        if self._preset == "textured":
            self._set_preset("textured")
        self.render_now()
        return True
    def _remove_textured(self):
        removed = list(self._texture_actors)
        stale_selection = any(actor in self._selected_cells
                              for actor in removed)
        for actor in removed:
            self.brush.forget_actor(actor)   # its triangle may be gone
            self._selected_cells.pop(actor, None)
            self._ren.RemoveActor(actor)
        self._texture_actors = []
        self._texture_originals = []
        self._has_textures = any(e["textures"] for e in self._originals)
        if stale_selection:
            self._rebuild_selection_actors()
        self._polygroups.forget_actors(removed)

    # -- swatch balls (top-right overlay) ------------------------------------
    def _build_swatches(self):
        self._swatch_ren = vtk.vtkRenderer()
        self._swatch_ren.SetViewport(0.72, 0.84, 0.99, 1.0)
        self._swatch_ren.SetBackground(0.16, 0.17, 0.19)
        self._swatch_ren.SetLayer(1)
        self._swatch_ren.SetInteractive(False)
        win = self.GetRenderWindow()
        win.SetNumberOfLayers(2)
        win.AddRenderer(self._swatch_ren)

        key = vtk.vtkLight(); key.SetPosition(1.0, 1.0, 2.0); key.SetIntensity(1.1)
        fill = vtk.vtkLight(); fill.SetPosition(-1.5, -0.5, 1.5)
        fill.SetIntensity(0.45)
        self._swatch_ren.AddLight(key)
        self._swatch_ren.AddLight(fill)

        # one row of four, even spacing; radius sized so balls keep a clear
        # gap between each other and the panel edges
        self._swatch_actors = {}
        radius = _SWATCH_RADIUS
        spacing = _SWATCH_SPACING            # center-to-center distance
        xs = [(i - 1.5) * spacing for i in range(4)]
        centers = dict(zip(_SWATCH_ORDER, ((x, 0.0) for x in xs)))
        for name in _SWATCH_ORDER:
            color, metallic, roughness = _SWATCH_LOOKS[name]
            source = vtk.vtkSphereSource()
            source.SetRadius(radius)
            source.SetPhiResolution(48)
            source.SetThetaResolution(48)
            source.Update()
            mapper = vtk.vtkOpenGLPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetInterpolationToPBR()
            prop.SetColor(*color)
            prop.SetMetallic(metallic)
            prop.SetRoughness(roughness)
            cx, cy = centers[name]
            actor.SetPosition(cx, cy, 0.0)
            actor._swatch_preset = name
            self._swatch_ren.AddActor(actor)
            self._swatch_actors[name] = actor

        cam = self._swatch_ren.GetActiveCamera()
        cam.ParallelProjectionOn()
        cam.SetFocalPoint(0.0, 0.0, 0.0)
        cam.SetPosition(0.0, 0.0, 5.0)
        cam.SetViewUp(0.0, 1.0, 0.0)
        self._swatch_ren.ResetCamera()
        self._fit_swatches()
    def _update_swatch_texture(self):
        """Snapshot the primary model and refresh the 'textured' ball."""
        if not self._originals:
            self._snapshot_originals()
        self._refresh_textured_swatch()
    def _refresh_textured_swatch(self):
        """Put the textured variant's albedo on the 'textured' ball.

        Falls back to the primary model's albedo for a fully textured mesh
        loaded directly (no separate variant).
        """
        actor = self._swatch_actors["textured"]
        albedo = next(
            (entry["textures"].get("albedoTex")
             for entry in (*self._texture_originals, *self._originals)
             if entry["textures"].get("albedoTex") is not None),
            None,
        )
        prop = actor.GetProperty()
        if albedo is not None:
            if hasattr(prop, "RemoveTexture"):
                prop.RemoveTexture("albedoTex")
            prop.SetTexture("albedoTex", albedo)
            prop.SetColor(1.0, 1.0, 1.0)
        else:
            if hasattr(prop, "RemoveTexture"):
                prop.RemoveTexture("albedoTex")
            prop.SetColor(0.55, 0.55, 0.55)
    def _swatch_at(self, qt_x: int, qt_y: int) -> str | None:
        """Preset name of the swatch ball under the cursor, if any."""
        # Qt logical pixels -> VTK window display pixels (they can differ
        # with DPI scaling or before the window catches up to a resize).
        rw_w, rw_h = self.GetRenderWindow().GetSize()
        sx = rw_w / max(1, self.width())
        sy = rw_h / max(1, self.height())
        x_vtk = int(qt_x * sx)
        y_vtk = int(rw_h - qt_y * sy)   # VTK display origin is bottom-left
        if self._picker.Pick(x_vtk, y_vtk, 0.0, self._swatch_ren):
            return getattr(self._picker.GetViewProp(), "_swatch_preset", None)
        return None
    def _world_per_pixel(self) -> float:
        cam = self._ren.GetActiveCamera()
        h = max(1, self.height())
        if cam.GetParallelProjection():
            return 2.0 * cam.GetParallelScale() / h
        return (2.0 * cam.GetDistance()
                * math.tan(math.radians(cam.GetViewAngle()) / 2.0) / h)
    def _pan(self, dx: float, dy: float):
        """Move the camera in the view plane; scene follows the cursor."""
        cam = self._ren.GetActiveCamera()
        wpp = self._world_per_pixel()
        d = cam.GetDirectionOfProjection()
        up = cam.GetViewUp()
        # screen-right = view direction x up
        right = (d[1] * up[2] - d[2] * up[1],
                 d[2] * up[0] - d[0] * up[2],
                 d[0] * up[1] - d[1] * up[0])
        vx = right[0] * (-dx * wpp) + up[0] * (dy * wpp)
        vy = right[1] * (-dx * wpp) + up[1] * (dy * wpp)
        vz = right[2] * (-dx * wpp) + up[2] * (dy * wpp)
        f = cam.GetFocalPoint()
        p = cam.GetPosition()
        cam.SetFocalPoint(f[0] + vx, f[1] + vy, f[2] + vz)
        cam.SetPosition(p[0] + vx, p[1] + vy, p[2] + vz)
    def _level_horizon(self):
        """Remove camera roll so the horizon sits level with the world
        ground; a horizontal (Azimuth) orbit then keeps it level. Skipped
        when looking almost straight up/down (nothing to level against).
        """
        cam = self._ren.GetActiveCamera()
        d = cam.GetDirectionOfProjection()
        ang = math.degrees(math.acos(max(-1.0, min(1.0,
            sum(a * b for a, b in zip(d, (0.0, 1.0, 0.0)))))))
        if not (2.0 < ang < 178.0):
            return
        cam.SetViewUp(0.0, 1.0, 0.0)
        cam.OrthogonalizeViewUp()
        self.render_now()

    def _orbit_world_y(self, degrees: float):
        """Rotate the camera around the WORLD Y axis through the focal point
        (or through the Draw-mode pivot when one is recorded).

        vtkCamera.Azimuth() orbits around the camera's own (orthogonalized)
        view-up axis, which is tilted whenever the camera looks up or down;
        every step then drifts the elevation and re-tilts the horizon. Doing
        the rotation manually around world Y keeps elevation frozen, so the
        horizon stays exactly level for the whole shift-drag orbit.
        """
        cam = self._ren.GetActiveCamera()
        pivot = self._orbit_pivot
        f = cam.GetFocalPoint()
        p = cam.GetPosition()
        if pivot is None:
            pivot = f
        rad = math.radians(degrees)
        cos, sin = math.cos(rad), math.sin(rad)
        x, y, z = p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2]
        # right-handed rotation about +Y (same sense vtkCamera.Azimuth uses
        # when view-up points along +Y); the focal point follows the same
        # rotation so the pivot keeps its screen position
        fx, fy, fz = f[0] - pivot[0], f[1] - pivot[1], f[2] - pivot[2]
        cam.SetPosition(pivot[0] + cos * x + sin * z,
                        pivot[1] + y,
                        pivot[2] - sin * x + cos * z)
        cam.SetFocalPoint(pivot[0] + cos * fx + sin * fz,
                          pivot[1] + fy,
                          pivot[2] - sin * fx + cos * fz)
        cam.SetViewUp(0.0, 1.0, 0.0)
        cam.OrthogonalizeViewUp()

    def _begin_turntable(self):
        """Straighten the horizon and arm the 90-degree snap before a
        shift-drag orbit.  Safe to call mid-gesture: an in-progress tumble
        hands over to the turntable without a jump."""
        self._level_horizon()    # straighten up before spinning
        cam = self._ren.GetActiveCamera()
        f = cam.GetFocalPoint()
        p = cam.GetPosition()
        self._tt_base = math.degrees(
            math.atan2(-(p[2] - f[2]), p[0] - f[0]))
        self._tt_free = 0.0
        self._tt_applied = self._tt_base

    def _turntable_drag(self, degrees: float):
        """Shift-drag orbit with lazy snapping to the compass headings.

        The cursor drives a free azimuth (1 px = 1 degree).  Whenever that
        free angle is within ``_TT_SNAP_ZONE`` degrees of a multiple of 90
        (north/south/east/west), the camera is pinned exactly onto that
        heading and stays there - it only breaks free once the drag
        carries the free angle out of the zone.  Rotating around world Y
        keeps the elevation frozen, so the horizon stays exactly level.
        """
        self._tt_free += degrees
        angle = self._tt_base + self._tt_free
        nearest = round(angle / 90.0) * 90.0
        target = (nearest if abs(angle - nearest) <= self._TT_SNAP_ZONE
                  else angle)
        delta = target - self._tt_applied
        if abs(delta) > 1e-9:
            self._orbit_world_y(delta)
            self._tt_applied = target

    def _tumble(self, dx: float, dy: float):
        """Free orbit around the focal point, clamped away from the poles.
        Directions are inverted: dragging right orbits the model left, etc.

        With a Draw-mode pivot recorded (last stroke / left-click on the
        mesh) the camera physically revolves around that point instead:
        position AND focal point are rotated by the same Rodrigues step, so
        the pivot keeps its screen position while the view swings around it.
        """
        pivot = self._orbit_pivot
        cam = self._ren.GetActiveCamera()
        if pivot is None:
            if dx:
                cam.Azimuth(-dx)
            if dy:
                cam.Elevation(dy)
                # keep the camera from flipping over the top/bottom
                d = cam.GetDirectionOfProjection()
                up = (0.0, 1.0, 0.0)
                ang = math.degrees(math.acos(max(-1.0, min(1.0, sum(
                    a * b for a, b in zip(d, up))))))
                if not (2.0 < ang < 178.0):
                    cam.Elevation(-dy)       # revert this step
            cam.OrthogonalizeViewUp()
            return

        up = np.asarray(cam.GetViewUp(), dtype=float)

        def _rodrigues(v, axis, degrees):
            rad = math.radians(degrees)
            c, s = math.cos(rad), math.sin(rad)
            return (v * c + np.cross(axis, v) * s
                    + axis * float(axis @ v) * (1.0 - c))

        f = np.asarray(cam.GetFocalPoint(), dtype=float) - pivot
        p = np.asarray(cam.GetPosition(), dtype=float) - pivot
        if dx:
            # azimuth: around the (orthogonalized) view-up axis; same sense
            # as cam.Azimuth(-dx) in the focal-point path
            axis = up
            n = float(np.linalg.norm(axis))
            if n > 1e-9:
                axis = axis / n
                f = _rodrigues(f, axis, -dx)
                p = _rodrigues(p, axis, -dx)
        if dy:
            # elevation: around the horizontal view-plane axis; the angle
            # is negated relative to cam.Elevation(dy) because VTK's call
            # orbits the camera around the focal point, while here the
            # whole camera frame revolves around the pivot - verified
            # against the stock path by screen-space scene motion
            d = np.asarray(cam.GetDirectionOfProjection(), dtype=float)
            axis = np.cross(d, up)
            n = float(np.linalg.norm(axis))
            if n > 1e-9:
                axis = axis / n
                f2 = _rodrigues(f, axis, -dy)
                p2 = _rodrigues(p, axis, -dy)
                d2 = p2 - f2
                length = float(np.linalg.norm(d2))
                if length > 1e-12:
                    d2 = d2 / length
                    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(
                        d2 @ np.asarray((0.0, 1.0, 0.0)))))))
                    if 2.0 < ang < 178.0:
                        f, p = f2, p2
        cam.SetFocalPoint(*(pivot + f))
        cam.SetPosition(*(pivot + p))
        cam.OrthogonalizeViewUp()

    # -- lasso face selection ------------------------------------------------
    @property
    def selection_count(self) -> int:
        """Number of faces currently highlighted by the selection."""
        return self._selection_count

    @property
    def selected_faces(self) -> list:
        """Current selection as FaceSelection(actor, cell_ids) entries."""
        return list(self._selected_faces)

    def _display_scale(self):
        """Qt logical -> VTK display pixel factors (HiDPI aware).

        Under Windows display scaling the render window works in physical
        pixels while Qt reports logical ones, so both axes can differ from
        1.0 (e.g. 1.25 at 125% scaling).  Anything converting between mouse
        deltas and VTK display pixels must go through this.
        """
        rw_w, rw_h = self.GetRenderWindow().GetSize()
        return (rw_w / max(1, self.width()),
                rw_h / max(1, self.height()))

    def _display_point(self, x, y):
        """Qt logical widget coords -> VTK display coords (bottom-left)."""
        rw_h = self.GetRenderWindow().GetSize()[1]
        sx, sy = self._display_scale()
        return x * sx, rw_h - y * sy

    def _polygroup_actors(self) -> list:
        """Source actors the active preset shows (never the filtered clones).

        The polygroup controller uses this to decide which actors to filter.
        It reflects the preset, not ``GetVisibility()``: hidden source actors
        are normal while filtered clones stand in for them.
        """
        if self._preset == "textured" and self._texture_actors:
            return list(self._texture_actors)
        return list(self._model_actors)

    def _selectable_actors(self) -> list:
        """Source actors selectable by the lasso for the active preset."""
        return self._polygroup_actors()

    def _pick_hits_model(self, x, y) -> bool:
        """True when a click lands on visible model geometry."""
        x_disp, y_disp = self._display_point(x, y)
        return self._polygroups.group_at(x_disp, y_disp) is not None

    # -- lasso drag lifecycle ------------------------------------------------
    def _begin_lasso(self, x, y, subtract=False):
        if self._lasso is None:
            return
        self._lasso_subtract = bool(subtract)
        # Remembered on press, acted on only at release (see _finish_lasso).
        self._lasso_press_group = self._polygroups.group_at(
            *self._display_point(x, y))
        color = LASSO_SUBTRACT_COLOR if subtract else LASSO_COLOR
        self._lasso.begin(*self._display_point(x, y), color=color)
        self._mode = "lasso"
        self._last = (x, y)
        self._lasso_cursor = self.cursor()
        self.setCursor(Qt.CrossCursor)
        self.render_now()

    def _update_lasso(self, x, y):
        if (self._lasso is not None
                and self._lasso.add(*self._display_point(x, y))):
            self.render_now()

    def _set_lasso_subtract(self, subtract):
        """Flip a live lasso between near-white/add and red/subtract."""
        subtract = bool(subtract)
        if subtract == self._lasso_subtract:
            return
        self._lasso_subtract = subtract
        if self._lasso is not None:
            self._lasso.set_color(
                LASSO_SUBTRACT_COLOR if subtract else LASSO_COLOR)
        self.render_now()

    def _flip_polygroups(self, group=None, invert=False):
        """Flip visibility to the hidden groups.

        With *invert* the whole visible set is swapped (an empty-space click
        while filtered); otherwise *group* is hidden and every other group
        shown (a lasso drag), or None means show everything.  The second
        visibility flip in a row always returns to all groups.
        """
        self._polygroup_flip_streak += 1
        if self._polygroup_flip_streak >= 2:
            self._polygroup_flip_streak = 0
            self._polygroups.show_all()
        elif invert:
            self._polygroups.invert_visibility()
        elif group is None:
            self._polygroups.show_all()
        else:
            self._polygroups.show_others(group)

    def _finish_lasso(self, x, y):
        subtract = self._lasso_subtract
        was_filtered = self._polygroups.is_filtered
        press_group = self._lasso_press_group
        self._lasso_subtract = False
        self._lasso_press_group = None
        if self._lasso is None:
            points = []
        else:
            self._lasso.add(*self._display_point(x, y))
            points = self._lasso.end()
        self._mode = None
        if self._lasso_cursor is not None:
            self.setCursor(self._lasso_cursor)
            self._lasso_cursor = None
        self._last = (0, 0)
        self.render_now()                # hide the rubber band

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        dragged = (len(points) >= 3 and xs
                   and max(max(xs) - min(xs), max(ys) - min(ys)) >= 3.0)
        if not dragged:
            if subtract:
                # A red click on a face is a no-op; empty still clears.
                self._polygroup_flip_streak = 0
                if press_group is None:
                    self._flip_polygroups(None)
                    self.clear_selection()
                return
            if press_group is not None:
                # Click on a face: usually solo its polygroup (act only on
                # release).  When some groups are already hidden, add this
                # one to the hidden set instead.
                self._polygroup_flip_streak = 0
                if was_filtered:
                    self._polygroups.hide_group(press_group)
                else:
                    self._polygroups.solo(press_group)
                return
            # Empty space: when filtered, reveal exactly the hidden groups;
            # when everything is visible there is nothing to flip.  Either
            # way the click deselects everything.
            self._flip_polygroups(invert=was_filtered)
            self.clear_selection()
            return

        selections = faces_in_lasso(self._ren, self._selectable_actors(),
                                    points)
        touched = self._install_selection(selections, subtract=subtract)

        if was_filtered and not subtract and not touched:
            # A drag that hits no visible face is an empty-space gesture:
            # flip to the other groups.  Selecting visible faces must never
            # reveal hidden groups.
            if press_group is None:
                self._flip_polygroups(invert=True)
            else:
                self._flip_polygroups(press_group)
        else:
            self._polygroup_flip_streak = 0

    def _cancel_lasso(self):
        """Abort an in-progress lasso (also used while loading a model)."""
        if self._lasso is not None:
            self._lasso.cancel()
        self._lasso_subtract = False
        self._lasso_press_group = None
        if self._mode == "lasso":
            self._mode = None
        if self._lasso_cursor is not None:
            self.setCursor(self._lasso_cursor)
            self._lasso_cursor = None
        self._last = (0, 0)

    # -- selection state -----------------------------------------------------
    def clear_selection(self, record: bool = True):
        """Remove all highlighted faces (also bound to Ctrl+D).

        *record* is False for clears that are part of a bigger recorded
        operation (polygroup assign), so one gesture stays one undo step.
        """
        if not self._selection_actors and not self._selected_cells:
            return
        before = self._selection_snapshot() if record else None
        self._selected_cells = {}
        self._remove_selection_actors()
        self._update_info_text()
        self.render_now()
        if record:
            self._record_history("Clear selection", before,
                                 self._apply_selection_snapshot,
                                 self._selection_snapshot)

    def _selection_snapshot(self) -> dict:
        """Cheap copy of the per-actor selected-cell map (undo support)."""
        return {actor: cells.copy()
                for actor, cells in self._selected_cells.items()}

    def _apply_selection_snapshot(self, snapshot) -> bool:
        """Restore a selection snapshot (Ctrl+Z / Ctrl+Shift+Z).

        An empty snapshot means "nothing was selected" and is applied as a
        clear; a snapshot whose actors all belong to a dropped mesh is
        stale and fails without touching anything.
        """
        known = {id(actor) for actor in
                 (*self._model_actors, *self._texture_actors)}
        live = {actor: cells for actor, cells in (snapshot or {}).items()
                if id(actor) in known}
        if snapshot and not live:
            return False
        self._selected_cells = {actor: cells.copy()
                                for actor, cells in live.items()}
        self._rebuild_selection_actors()
        return True

    # -- undo/redo records ----------------------------------------------------
    def _record_history(self, label, before, apply, after_fn):
        """Push a history entry when the fresh *after_fn()* differs.

        Shared tail for lasso selections and polygroup paints: *before* is
        the snapshot captured before the gesture, *apply* restores a
        snapshot.  No-op (nothing pushed) when the gesture changed nothing.
        """
        after = after_fn()
        if self._snapshots_equal(before, after):
            return
        self._history.push(StateEntry(label, apply, before, after))

    @staticmethod
    def _snapshots_equal(a, b) -> bool:
        """True when two snapshot dicts hold identical arrays."""
        a = a or {}
        b = b or {}
        if a.keys() != b.keys():
            return False
        return all(np.array_equal(a[key], b[key]) for key in a)

    def _record_polygroups(self, label, before):
        """Record a polygroup paint/reset for the undo history."""
        self._record_history(label, before,
                             self._polygroups.apply_snapshot,
                             self._polygroups.snapshot)

    def _remove_selection_actors(self):
        """Drop the highlight actors; the selected-cell map is untouched."""
        for actor in self._selection_actors:
            self._ren.RemoveActor(actor)
        self._selection_actors = []
        self._selection_sources = []
        self._selected_faces = []
        self._selection_count = 0

    def _install_selection(self, selections, subtract=False) -> int:
        """Union (additive) or remove (red) the lassoed cells, then redraw.

        Returns how many *visible* cells the lasso touched; cells in hidden
        polygroups are ignored.  This lets the caller tell a real selection
        drag from a drag over empty space.
        """
        if not selections:
            return 0
        before = self._selection_snapshot()
        touched = 0
        for selection in selections:
            actor = selection.actor
            # Hidden polygroups cannot be selected through the filter.
            cell_ids = self._polygroups.filter_cell_ids(
                actor, selection.cell_ids)
            touched += int(cell_ids.size)
            if cell_ids.size == 0:
                continue
            merged = combine_selection_cells(
                self._selected_cells.get(actor), cell_ids,
                subtract=subtract)
            if merged.size:
                self._selected_cells[actor] = merged
            else:
                self._selected_cells.pop(actor, None)
        self._rebuild_selection_actors()
        self._record_history(
            "Lasso subtract" if subtract else "Lasso select", before,
            self._apply_selection_snapshot, self._selection_snapshot)
        return touched

    def _rebuild_selection_actors(self):
        """Rebuild highlight overlays from the per-actor selected-cell map.

        The *state* keeps every selected face; the *overlays* only draw the
        ones whose polygroup is currently visible.  Hiding a group therefore
        does not lose its selection when it is shown again later.
        """
        self._remove_selection_actors()
        known = {id(actor) for actor in
                 (*self._model_actors, *self._texture_actors)}
        self._selected_cells = {
            actor: cells for actor, cells in self._selected_cells.items()
            if id(actor) in known and len(cells)
        }
        self._selected_faces = [
            FaceSelection(actor, cells)
            for actor, cells in self._selected_cells.items()
        ]
        self._selection_count = sum(len(selection.cell_ids)
                                    for selection in self._selected_faces)
        drawn = []
        for selection in self._selected_faces:
            cell_ids = self._polygroups.filter_cell_ids(
                selection.actor, selection.cell_ids)
            if cell_ids.size:
                drawn.append(FaceSelection(selection.actor, cell_ids))
        overlays = build_selection_actors(drawn, self._backface_cull)
        for actor in overlays:
            actor.SetPickable(False)   # highlights never block picking
            self._ren.AddActor(actor)
        self._selection_actors = overlays
        self._selection_sources = [actor._selection_source
                                   for actor in overlays]
        self._sync_selection_visibility()
        self._update_info_text()
        self.render_now()

    def _sync_selection_visibility(self):
        """Show highlights with the actor set of the active preset.

        Filtered sources are physically hidden in favour of clones, so the
        source's own visibility cannot be used here.
        """
        active = {id(actor) for actor in self._polygroup_actors()}
        for actor, source in zip(self._selection_actors,
                                 self._selection_sources):
            actor.SetVisibility(id(source) in active)

    # -- polygroups ----------------------------------------------------------
    @property
    def polygroup_count(self) -> int:
        """Number of polygroup colours currently in use (0..25)."""
        return self._polygroups.used_count

    def assign_polygroup(self) -> int | None:
        """Paint faces with the next free polygroup, bound to Ctrl+W.

        With a selection, the selected (visible) faces are painted.  With
        nothing selected and every group visible, the mesh returns to the
        uncoloured default group 0.  With nothing selected and some groups
        hidden, every visible face is painted into a new group and only that
        group stays visible.  Returns the group id (0 for the reset) or None
        when nothing can be assigned / all 25 colours are in use.
        """
        if not self._selected_faces:
            if not self._polygroups.is_filtered:
                if not self._selectable_actors():
                    return None
                before = self._polygroups.snapshot()
                self._polygroups.reset_to_default()
                self._record_polygroups("Reset polygroups", before)
                return DEFAULT_POLYGROUP
            selections = []
            for actor in self._selectable_actors():
                ids = self._polygroups.visible_cell_ids(actor)
                if ids is not None and ids.size:
                    selections.append(FaceSelection(actor, ids))
            if not selections:
                return None
            before = self._polygroups.snapshot()
            group = self._polygroups.assign(selections)
            if group is None:
                self.notice.emit("Polygroup: all colours are in use; reset "
                                 "a group first.", "warn")
                return None
            # Everything that was visible now belongs to the new group, so
            # keep exactly that group on screen (hidden groups stay hidden).
            self._polygroups.set_visible_groups((group,))
            self._record_polygroups("Assign polygroup", before)
            return group
        # While filtered, reveal the new group so the painted faces do not
        # disappear behind the old visibility set.
        before = self._polygroups.snapshot()
        group = self._polygroups.assign(self._selected_faces, reveal=True)
        if group is None:
            self.notice.emit("Polygroup: all colours are in use; reset "
                             "a group first.", "warn")
            return None
        self.clear_selection(record=False)
        self._record_polygroups("Assign polygroup", before)
        return group

    def assign_polygroup_islands(self) -> list | None:
        """One polygroup per connected island, bound to Ctrl+Shift+W.

        Only faces the filter currently shows take part (hidden groups stay
        untouched and never bridge islands).  With a lasso selection the
        islands are computed within that selection; with nothing selected
        every visible face is split.  Islands are painted largest-first into
        the free polygroup colours; if there are more islands than free
        colours, the smaller ones stay unpainted.  The new groups are always
        added to the visible set when a filter is active, so freshly painted
        islands never vanish.  Returns the list of new group ids, or None
        when nothing could be assigned.
        """
        if self._selected_faces:
            selections = self._selected_faces
        else:
            selections = []
            for actor in self._selectable_actors():
                ids = self._polygroups.visible_cell_ids(actor)
                if ids is not None and ids.size:
                    selections.append(FaceSelection(actor, ids))
        if not selections:
            self.notice.emit("Island split: there are no visible faces.",
                             "warn")
            return None
        before = self._polygroups.snapshot()
        groups, islands = self._polygroups.assign_islands(
            selections, reveal=True)
        if not groups:
            self.notice.emit("Island split: no free polygroups left; reset "
                             "a group first.", "warn")
            return None
        self.clear_selection(record=False)
        self._record_polygroups("Assign polygroup islands", before)
        if len(groups) < islands:
            self.notice.emit(
                f"Island split: painted the {len(groups)} largest of "
                f"{islands:,} islands; the rest stay unpainted "
                f"({self._polygroups.capacity} polygroups max).", "warn")
        else:
            self.notice.emit(
                f"Island split: painted {len(groups)} island(s).", "info")
        return groups

    # -- split and save ------------------------------------------------------
    @property
    def preset(self) -> str:
        """Name of the active material preset."""
        return self._preset

    def set_material_preset(self, name: str):
        """Apply a material preset by name (restored after a split wizard)."""
        if name in _SWATCH_LOOKS and name != self._preset:
            self._set_preset(name)

    def solo_polygroup(self, group) -> bool:
        """Show only *group* (used by the Split-and-save wizard)."""
        return self._polygroups.solo(group)

    def polygroup_visibility(self):
        """Visible group ids, or None when every group is visible."""
        return self._polygroups.visible_groups

    def set_polygroup_visibility(self, groups) -> bool:
        """Restore a visibility set captured before a split (None = all)."""
        return self._polygroups.set_visible_groups(groups)

    def _painted_polygroups(self, actors) -> tuple:
        groups = self._polygroups.face_groups(actors)
        return tuple(group for group in groups
                     if group != DEFAULT_POLYGROUP)

    def _split_target(self) -> str:
        """'model' or 'textured': the actor set holding the split groups.

        Groups are stored on whichever actor set was active when they were
        painted, so prefer the current preset and fall back to the other
        variant when it carries all of the painting.
        """
        textured = bool(self._preset == "textured" and self._texture_actors)
        active = self._texture_actors if textured else self._model_actors
        if self._painted_polygroups(active):
            return "textured" if textured else "model"
        other = self._model_actors if textured else self._texture_actors
        if self._painted_polygroups(other):
            return "model" if textured else "textured"
        return "textured" if textured else "model"

    def _split_actors(self) -> list:
        if self._split_target() == "textured":
            return list(self._texture_actors)
        return list(self._model_actors)

    def prepare_polygroup_split(self) -> bool:
        """Show the mesh that owns the polygroups.

        Returns True when the material preset was changed, so the caller can
        restore it when the wizard ends.
        """
        target = self._split_target()
        if target == "textured" and self._preset != "textured":
            self._set_preset("textured")
            return True
        if target == "model" and self._preset == "textured":
            self._set_preset("original")
            return True
        return False

    def polygroup_split_geometry(self) -> dict:
        """World-space triangles for Split-and-save, keyed by group id.

        Group 0 contains the unpainted faces.  The heavy lifting lives in
        :func:`polygroups.split_geometry`; this only picks the actor set.
        """
        actors = self._split_actors()
        return split_geometry(
            actors,
            {actor: self._polygroups.cell_groups(actor) for actor in actors})

    def polygroup_visible_geometry(self) -> tuple:
        """Geometry for Delete-hidden, from the polygroup-carrying actors.

        Returns ``(vertices, triangles, keep_mask, kept, total,
        tri_groups)`` covering every polygon of the actor set chosen like
        for Split-and-save; *keep_mask* marks the triangles whose
        polygroup is currently visible and *tri_groups* the per-triangle
        polygroup id, so the surviving faces' groups can be restored after
        the delete.  The heavy lifting lives in
        :func:`polygroups.visible_geometry`.
        """
        actors = self._split_actors()
        return visible_geometry(
            actors,
            {actor: self._polygroups.cell_groups(actor) for actor in actors},
            self._polygroups.visible_groups)

    def restore_polygroups(self, face_groups) -> bool:
        """Re-apply per-polygon polygroup ids to the displayed mesh.

        *face_groups* must have one entry per polygon of the freshly
        loaded mesh, in the same order (Delete-hidden keeps the surviving
        faces in order).  Polygroups that still have faces keep their
        original colours; groups left without faces disappear.  Returns
        True when at least one group was restored.
        """
        if face_groups is None:
            return False
        actors = self._split_actors()
        if len(actors) != 1:
            return False
        return self._polygroups.restore(actors[0], face_groups)

    def polygroups_face_groups(self):
        """Per-polygon group ids of the displayed mesh, or None.

        None when there is nothing to carry over: no model, several
        actors, or no painted polygroups.  The array is aligned with the
        polygon rows of the displayed (pending-edit) mesh, so Bake can
        match the reloaded file back to these groups by face.
        """
        actors = self._split_actors()
        if len(actors) != 1 or not self._polygroups.used_count:
            return None
        cells = self._polygroups.cell_groups(actors[0])
        if cells is None or not cells.any():
            return None
        return np.asarray(cells, dtype=np.int64).copy()

    def restore_polygroups_by_faces(self, old_vertices, old_faces,
                                    old_groups) -> int:
        """Re-apply polygroups after Bake by matching faces on positions.

        Bake writes the edited mesh to disk and reloads it; the reload can
        renumber vertices and GLB triangulates quads, so faces are matched
        by their vertex positions (float32 - exactly what a GLB stores)
        instead of indices.  Old quads participate through their fan
        triangles.  Faces without a match - genuinely new geometry, e.g.
        Close-holes caps - come back in the unpainted default group.
        Returns how many polygons got a group back (0 = nothing matched).
        """
        if old_vertices is None or old_faces is None or old_groups is None:
            return 0
        old_faces = np.asarray(old_faces, dtype=np.int64)
        old_groups = np.asarray(old_groups, dtype=np.int64).ravel()
        if old_faces.ndim != 2 or len(old_faces) != len(old_groups):
            return 0
        actors = self._split_actors()
        if len(actors) != 1:
            return 0
        mapper = actors[0].GetMapper()
        polydata = mapper.GetInput() if mapper is not None else None
        polys = _poly_cells(polydata) if polydata is not None else None
        if polys is None:
            return 0
        offsets, connectivity = polys
        stride = int(offsets[1] - offsets[0])
        if stride != 3 or not np.all(np.diff(offsets) == stride):
            return 0              # only uniform triangle meshes match
        points = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()),
                            dtype=np.float32)
        old_keys, owners = _face_position_keys(old_vertices, old_faces)
        lut = {}
        for key, row in zip(old_keys, owners):
            group = int(old_groups[row])
            if group:
                lut.setdefault(key, group)
        new_keys, owners = _face_position_keys(points,
                                               connectivity.reshape(-1, 3))
        groups = np.zeros(len(new_keys), dtype=np.uint8)
        matched = 0
        for key, row in zip(new_keys, owners):
            group = lut.get(key)
            if group:
                groups[row] = group
                matched += 1
        if not matched or not self._polygroups.restore(actors[0], groups):
            return 0
        return matched

    def polygroups_hidden(self) -> bool:
        """True when the visibility filter currently hides some groups."""
        return self._polygroups.is_filtered

    def _polygroups_changed(self):
        """Callback from the polygroup controller after overlays change."""
        self._update_info_text()
        self.render_now()

    def _polygroup_visibility_changed(self):
        """Solo/filter changed: redraw only the still-visible selection."""
        self.brush.last_hit = None       # cached triangle may now be hidden
        self._rebuild_selection_actors()

    # -- smooth brush cursor (hold Shift over the mesh; viewer_brush.py) -----
    def keyPressEvent(self, event):
        if self.resym.active and event.key() == Qt.Key_Escape:
            self.resym.cancel()
            event.accept()
            return
        if self.brush.key(event, pressed=True):
            event.accept()
            return
        # VTK's stock interactor style binds plain letters when they fall
        # through to it: w = wireframe, s = surface, r = reset camera,
        # p = pick, q/e = exit, f = fly-to, 3 = stereo, u/l/j = misc.
        # Swallow them unmodified so a stray keypress can never mangle the
        # view; combos (Ctrl+W, Ctrl+Z, ...) are QShortcuts and never arrive.
        if (event.modifiers() == Qt.NoModifier
                and event.key() in _VTK_STYLE_KEYS):
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self.brush.key(event, pressed=False):
            event.accept()
            return
        super().keyReleaseEvent(event)

    def undo_last_stroke(self):
        """Ctrl+Z: undo the newest edit (brush stroke, lasso, polygroup)."""
        entry = self._history.undo()
        return self._after_history_apply(entry)

    def redo_last_stroke(self):
        """Ctrl+Shift+Z: re-apply the most recently undone edit."""
        entry = self._history.redo()
        return self._after_history_apply(entry)

    def _after_history_apply(self, entry):
        """Shared tail of undo/redo: refresh caches, render, announce."""
        if entry is None:
            return False
        actor = getattr(entry, "actor", None)
        actor = actor() if callable(actor) else actor
        if actor is not None:
            self.brush.invalidate(actor)   # geometry changed underneath
        self._ren.ResetCameraClippingRange()
        self.render_now()
        if isinstance(entry, VertexEditEntry):
            self.smoothEdited.emit()     # only geometry edits re-Bake
        return True

    def edited_geometry(self):
        """(vertices, triangles) of the displayed mesh, world space.

        The same array format the edit operations produce, so Bake changes
        can save smooth-brush results like any other pending edit.
        Returns None when there is no displayed model.
        """
        actors = self._split_actors()
        if not actors:
            return None
        vertices, triangles, _mask, _kept, _total, _groups = \
            visible_geometry(
            actors,
            {actor: self._polygroups.cell_groups(actor) for actor in actors},
            None)
        if vertices is None or not len(vertices) or triangles is None:
            return None
        return vertices, triangles

    def mousePressEvent(self, event):
        # Camera/lasso gestures always reclaim the cursor; the brush only
        # lives on plain hover, and reappears on the next move after release.
        self.brush.hide()
        x, y = event.position().toTuple()
        btn = event.button()
        trace("viewer.press", pos=(x, y), button=qt(btn),
              mods=qt(event.modifiers()), mode=self._mode,
              draw=self.brush.draw_enabled, shift=self.brush.shift_active,
              pivot=vec3(self._orbit_pivot),
              **cam_info(self._ren.GetActiveCamera()))
        if self.resym.active:
            # the reSym wizard is active: a plain left-CLICK places a
            # point, but the press first starts a tumble so the camera
            # can be orbited to reach hidden spots - the pick only fires
            # on release when the drag stayed under _RESYM_CLICK_PX.
            # Esc cancels; right (dolly) and middle (pan) fall through.
            if btn == Qt.LeftButton:
                self._resym_press = (x, y)
                self._record_orbit_pivot(x, y)
                self._mode = "tumble"
                self._last = (x, y)
                event.accept()
                return
        if btn == Qt.LeftButton:
            mods = event.modifiers()
            if (mods & Qt.ControlModifier) and (mods & Qt.ShiftModifier):
                self._begin_lasso(x, y,
                                  subtract=bool(mods & Qt.AltModifier))
            elif self.brush.press_begin_stroke(x, y):
                pass                       # smoothing stroke owns this drag
            elif self._swatch_at(x, y):
                name = getattr(self._picker.GetViewProp(), "_swatch_preset",
                               None)
                if name and not (name == "textured"
                                 and not self._has_textures):
                    self._set_preset(name)
                self._mode = "swatch"
            elif self.brush.shift_active:
                pass        # Shift-brush session: the drag keeps smoothing
            elif event.modifiers() & Qt.ShiftModifier:
                self._mode = "turntable"
                self._last = (x, y)
                self._begin_turntable()
            else:
                # Draw mode: a plain left-drag on the mesh is the
                # grab-and-move ("snake hook") brush - it (re)aims the
                # orbit/zoom pivot at the grabbed point.  A press on empty
                # space (or with Draw off) falls through to tumbling.
                if self.brush.press_begin_move(x, y):
                    pass               # move stroke owns this drag
                else:
                    self._record_orbit_pivot(x, y)
                    self._mode = "tumble"
                    self._last = (x, y)
        elif btn == Qt.MiddleButton:
            self._mode = "pan"
            self._last = (x, y)
        elif btn == Qt.RightButton:
            self._mode = "dolly"
            self._last = (x, y)
        else:
            super().mousePressEvent(event)
            return
        # Draw mode: the circle stays on through pan/tumble/turntable/dolly
        if self._mode in ("turntable", "tumble", "pan", "dolly"):
            self.brush.drag_cursor(x, y)
        trace("viewer.press.done", pos=(x, y), mode=self._mode,
              draw=self.brush.draw_enabled)
        event.accept()
    def mouseMoveEvent(self, event):
        x, y = event.position().toTuple()
        self._mouse_pos = (x, y)
        trace("viewer.move", pos=(x, y), buttons=qt(event.buttons()),
              mods=qt(event.modifiers()), mode=self._mode,
              draw=self.brush.draw_enabled, pivot=vec3(self._orbit_pivot),
              **cam_info(self._ren.GetActiveCamera()))
        if self.brush.shift_active and self._mode is None:
            # Shift-hold session: the ring follows the mesh, but smoothing
            # only ever happens with the left button held (press handler).
            self.brush.hover(x, y)
            event.accept()
            return
        if self._mode in (None, "swatch"):
            if self._mode is None:
                self.brush.hover(x, y)
            super().mouseMoveEvent(event)
            return
        if self._mode == "brush_smooth":
            self.brush.stroke_move(x, y)   # ring keeps following the stroke
            event.accept()
            return
        if self._mode == "brush_move":
            # Alt held = face-normalized displacement (live toggle, so it
            # can flip mid-drag; release falls back to screen mode)
            self.brush.move_move(
                x, y, bool(event.modifiers() & Qt.AltModifier))
            event.accept()
            return
        self.brush.hide()      # drags pause the brush until release
        if self._mode == "lasso":
            # Alt can be pressed or released after the drag has started:
            # the band recolours immediately and release follows it.
            self._set_lasso_subtract(
                bool(event.modifiers() & Qt.AltModifier))
            self._update_lasso(x, y)
            event.accept()
            return
        dx = x - self._last[0]
        dy = y - self._last[1]
        self._last = (x, y)
        cam = self._ren.GetActiveCamera()
        # Shift is a live toggle during a left-drag: pressing it mid-tumble
        # hands the gesture over to the horizon-locked turntable; releasing
        # it returns to free tumbling.
        if (self._mode == "tumble"
                and (event.modifiers() & Qt.ShiftModifier)):
            self._begin_turntable()
            self._mode = "turntable"
        elif (self._mode == "turntable"
                and not (event.modifiers() & Qt.ShiftModifier)):
            self._mode = "tumble"
        if self._mode == "turntable":
            if dx:
                self._turntable_drag(-dx)
        elif self._mode == "tumble":
            self._tumble(dx, dy)
        elif self._mode == "pan":
            self._pan(dx, dy)
        elif self._mode == "dolly":
            if dx or dy:
                # drag right / down = zoom in; left / up = zoom out
                self._dolly(1.005 ** (dx + dy))
        # Draw mode: the flat circle tracks the mouse through the gesture
        self.brush.drag_cursor(x, y)
        self.render_now()
        trace("viewer.move.done", pos=(x, y), mode=self._mode, d=(dx, dy),
              pivot=vec3(self._orbit_pivot), **cam_info(cam))
        event.accept()
    def mouseReleaseEvent(self, event):
        x0, y0 = event.position().toTuple()
        trace("viewer.release", pos=(x0, y0), button=qt(event.button()),
              mods=qt(event.modifiers()), mode=self._mode,
              pivot=vec3(self._orbit_pivot),
              **cam_info(self._ren.GetActiveCamera()))
        if (self.resym.active and event.button() == Qt.LeftButton
                and self._mode == "tumble"):
            # reSym: a click that didn't drag into an orbit places a point
            press = getattr(self, "_resym_press", None)
            self._mode = None
            self._resym_press = None
            if press is not None and math.dist(press, (x0, y0)) \
                    < _RESYM_CLICK_PX:
                self.resym.pick_point(x0, y0)
            event.accept()
            return
        if self._mode == "brush_smooth":
            self.brush.end_stroke()
            event.accept()
            return
        if self._mode == "brush_move":
            self.brush.end_move()
            event.accept()
            return
        if self._mode == "lasso":
            x, y = event.position().toTuple()
            self._finish_lasso(x, y)
            event.accept()
            return
        if self._mode is None:
            super().mouseReleaseEvent(event)
            return
        self._mode = None
        self._last = (0, 0)
        event.accept()
    def leaveEvent(self, event):
        self.brush.hide()
        super().leaveEvent(event)

    def _record_orbit_pivot(self, x, y):
        """Draw mode: (re)aim the orbit/zoom pivot at the mesh under (x, y).

        Cheap on purpose: one brush pick (cached-triangle Moller-Trumbore
        first, then the per-actor locators) and only in Draw mode - a miss
        keeps the previous pivot.  Cleared by reset_camera (F) and on model
        swap.
        """
        if not self.brush.draw_enabled:
            return
        hit = self.brush.pick(x, y)
        if hit is not None:
            self._orbit_pivot = np.asarray(hit[0], dtype=float)

    def _dolly(self, factor: float):
        """Zoom by a multiplicative factor (>1 = closer), identical in both
        projection modes.

        Without a Draw-mode pivot this uses vtkCamera.Zoom, which adjusts
        ViewAngle in perspective and ParallelScale in orthographic - the
        camera position never moves.  That keeps the camera headlight
        shading constant while zooming (a positional dolly would sweep the
        light across the model) and makes zoom feel the same whichever
        projection is active.  With a pivot recorded, the camera dollies
        toward that point instead so the pivot stays put on screen (see
        the pivot branch below).
        """
        cam = self._ren.GetActiveCamera()
        pivot = self._orbit_pivot
        if pivot is not None:
            # Draw-mode pivot: dolly the camera toward the point instead of
            # zooming the view angle, so the pivot stays put on screen.  In
            # perspective the position move alone gives the magnification;
            # in orthographic position moves are invisible, so the focal
            # point translation pans the pivot inward and Zoom does the rest.
            p = np.asarray(cam.GetPosition(), dtype=float)
            f = np.asarray(cam.GetFocalPoint(), dtype=float)
            cam.SetPosition(*(pivot + (p - pivot) / factor))
            cam.SetFocalPoint(*(pivot + (f - pivot) / factor))
            if cam.GetParallelProjection():
                cam.Zoom(factor)
            self._ren.ResetCameraClippingRange()
            return
        cam.Zoom(factor)
        if not cam.GetParallelProjection():
            # guard against degenerate view angles at the zoom extremes
            cam.SetViewAngle(max(2.0, min(150.0, cam.GetViewAngle())))
        self._ren.ResetCameraClippingRange()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta:
            self._dolly(1.15 if delta > 0 else 1.0 / 1.15)
            self.render_now()
            event.accept()
            return
        super().wheelEvent(event)
