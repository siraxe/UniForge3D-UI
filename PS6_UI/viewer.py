"""Native VTK 3D viewer widget for PySide6.

Loads GLB / glTF / OBJ / STL / PLY with full PBR materials (glTF) and embeds
as a native Windows widget - no browser involved.

Maya-style camera: left-drag tumble   shift+left horizontal spin
middle-drag pan   right-drag / wheel dolly (zoom)

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

import vtk
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

_BG_BOTTOM = (0x2b / 255, 0x2e / 255, 0x37 / 255)   # #2b2e37
_BG_TOP = (0x17 / 255, 0x18 / 255, 0x1e / 255)      # #17181e

# glTF texture slots the importer assigns on vtkProperty
_TEXTURE_SLOTS = ("albedoTex", "normalTex", "rmTex", "aoTex", "emissiveTex")

# material presets applied to the model: (color, metallic, roughness)
_PRESETS = {
    "dark_specular": ((0.30, 0.32, 0.35), 0.35, 0.45),   # matte-ish dark
    "bronze": ((1.00, 0.66, 0.32), 0.70, 0.35),          # less metallic
}
_SWATCH_LOOKS = {                       # how each ball itself looks
    "original": ((0.80, 0.80, 0.80), 0.0, 0.45),
    "bronze": ((1.00, 0.66, 0.32), 0.70, 0.35),
    "dark_specular": ((0.30, 0.32, 0.35), 0.35, 0.45),
    "textured": ((0.55, 0.55, 0.55), 0.0, 0.60),   # + albedo texture
}
_SWATCH_ORDER = ("dark_specular", "bronze", "original", "textured")
_SWATCH_RADIUS = 0.19                  # world units, ball radius
_SWATCH_SPACING = 0.60                 # world units, center-to-center


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


class ModelViewer(QVTKRenderWindowInteractor):
    """Render-window widget; call load() with any supported model file."""

    SUPPORTED = {".glb", ".gltf", ".obj", ".stl", ".ply"}

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
        self._mode = None               # None|swatch|turntable|tumble|pan|dolly
        self._last = (0, 0)

        self._current_path = None
        self._model_actors = []          # vtkActor list of the loaded model
        self._originals = []             # per-actor saved material state
        self._texture_actors = []        # textured variant for the 4th swatch
        self._texture_originals = []     # material state of that variant
        self._has_textures = False
        self._preset = "original"
        self._backface_cull = True

        self._build_swatches()
        self._picker = vtk.vtkPropPicker()

        # F = frame/reset the view (works anywhere in the app)
        QShortcut(QKeySequence(Qt.Key_F), self, activated=self.reset_camera,
                  context=Qt.ApplicationShortcut)

        self.setFocusPolicy(Qt.StrongFocus)
        self.Initialize()
        self.Start()

    # ------------------------------------------------------------------
    @property
    def current_path(self) -> Path | None:
        return self._current_path

    @property
    def has_textures(self) -> bool:
        return self._has_textures
    def clear(self):
        self._ren.RemoveAllViewProps()
        self._current_path = None
        self._model_actors = []
        self._originals = []
        self._texture_actors = []
        self._texture_originals = []
        self._has_textures = False
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
        self._fit_swatches()             # fresh camera scale for this window size
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
        for actor in (*self._model_actors, *self._texture_actors):
            prop = actor.GetProperty()
            if self._backface_cull:
                prop.BackfaceCullingOn()
            else:
                prop.BackfaceCullingOff()

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
        self._ren.ResetCamera()
        self._ren.ResetCameraClippingRange()
        self.render_now()
    def render_now(self):
        self.GetRenderWindow().Render()

    # -- swatch layout helpers --------------------------------------------
    def _fit_swatches(self):
        """Keep the swatch row fully visible at any window aspect."""
        size = self._swatch_ren.GetSize()
        if size[0] <= 0 or size[1] <= 0:
            return                       # not laid out yet
        aspect = size[0] / max(1.0, float(size[1]))
        half_span = _SWATCH_SPACING * 1.5 + _SWATCH_RADIUS
        self._swatch_ren.GetActiveCamera().SetParallelScale(
            max(_SWATCH_RADIUS * 1.45, half_span / aspect))
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
        for actor in self._texture_actors:
            self._ren.RemoveActor(actor)
        self._texture_actors = []
        self._texture_originals = []
        self._has_textures = any(e["textures"] for e in self._originals)

    # -- swatch balls (top-right overlay) ------------------------------------
    def _build_swatches(self):
        self._swatch_ren = vtk.vtkRenderer()
        self._swatch_ren.SetViewport(0.72, 0.85, 0.99, 0.99)
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
        """Rotate the camera around the WORLD Y axis through the focal point.

        vtkCamera.Azimuth() orbits around the camera's own (orthogonalized)
        view-up axis, which is tilted whenever the camera looks up or down;
        every step then drifts the elevation and re-tilts the horizon. Doing
        the rotation manually around world Y keeps elevation frozen, so the
        horizon stays exactly level for the whole shift-drag orbit.
        """
        cam = self._ren.GetActiveCamera()
        f = cam.GetFocalPoint()
        p = cam.GetPosition()
        rad = math.radians(degrees)
        cos, sin = math.cos(rad), math.sin(rad)
        x, y, z = p[0] - f[0], p[1] - f[1], p[2] - f[2]
        # right-handed rotation about +Y (same sense vtkCamera.Azimuth uses
        # when view-up points along +Y)
        cam.SetPosition(f[0] + cos * x + sin * z,
                        f[1] + y,
                        f[2] - sin * x + cos * z)
        cam.SetViewUp(0.0, 1.0, 0.0)
        cam.OrthogonalizeViewUp()

    def _tumble(self, dx: float, dy: float):
        """Free orbit around the focal point, clamped away from the poles.
        Directions are inverted: dragging right orbits the model left, etc.
        """
        cam = self._ren.GetActiveCamera()
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
    def mousePressEvent(self, event):
        x, y = event.position().toTuple()
        btn = event.button()
        if btn == Qt.LeftButton:
            if self._swatch_at(x, y):
                name = getattr(self._picker.GetViewProp(), "_swatch_preset",
                               None)
                if name and not (name == "textured"
                                 and not self._has_textures):
                    self._set_preset(name)
                self._mode = "swatch"
            elif event.modifiers() & Qt.ShiftModifier:
                self._level_horizon()    # straighten up before spinning
                self._mode = "turntable"
                self._last = (x, y)
            else:
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
        event.accept()
    def mouseMoveEvent(self, event):
        if self._mode in (None, "swatch"):
            super().mouseMoveEvent(event)
            return
        x, y = event.position().toTuple()
        dx = x - self._last[0]
        dy = y - self._last[1]
        self._last = (x, y)
        cam = self._ren.GetActiveCamera()
        if self._mode == "turntable":
            if dx:
                self._orbit_world_y(-dx)  # horizontal only, tilt locked (inverted)
        elif self._mode == "tumble":
            self._tumble(dx, dy)
        elif self._mode == "pan":
            self._pan(dx, dy)
        elif self._mode == "dolly":
            if dx or dy:
                # drag right / down = zoom in; left / up = zoom out
                cam.Dolly(1.005 ** (dx + dy))
                self._ren.ResetCameraClippingRange()
        self.render_now()
        event.accept()
    def mouseReleaseEvent(self, event):
        if self._mode is None:
            super().mouseReleaseEvent(event)
            return
        self._mode = None
        self._last = (0, 0)
        event.accept()
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta:
            cam = self._ren.GetActiveCamera()
            cam.Dolly(1.15 if delta > 0 else 1.0 / 1.15)
            self._ren.ResetCameraClippingRange()
            self.render_now()
            event.accept()
            return
        super().wheelEvent(event)
