"""Edit-operation controllers: Reproject, Polish, Decimate, Remesh."""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal


class ReprojectController(QObject):
    log_line = Signal(str, str)
    state_changed = Signal(bool)          # True while projecting
    reproject_done = Signal(object)       # result dict (arrays + stats)
    reproject_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._cancel = threading.Event()

    @property
    def running(self) -> bool:
        return self._running

    def cancel(self):
        """Ask the running job to stop (cooperative; takes effect at the
        next cancellation point inside the tool)."""
        self._cancel.set()

    def start(self, target: Path, source: Path, *,
              mode: str, dist: float, strength: float = 1.0,
              pa_blur: int = 0, projection_shell: float = 0.0,
              detail_iterations: int = 5, subdiv_levels: int = 1,
              target_vertices=None, target_faces=None):
        if self._running:
            return
        self._cancel.clear()
        self._running = True
        self.state_changed.emit(True)
        threading.Thread(
            target=self._job,
            args=(Path(target), Path(source), str(mode), float(dist),
                  float(strength), int(pa_blur), float(projection_shell),
                  int(detail_iterations), int(subdiv_levels),
                  target_vertices, target_faces),
            daemon=True).start()

    def _job(self, target: Path, source: Path, mode: str, dist: float,
             strength: float, pa_blur: int, projection_shell: float,
             detail_iterations: int, subdiv_levels: int,
             target_vertices, target_faces):
        try:
            # loaded fresh from disk every job (tools/ is hot-editable)
            from tools._loader import load_tool
            reproject = load_tool("reproject")

            result = reproject.reproject_meshes(
                target, source, mode=mode, dist=dist, strength=strength,
                pa_blur=pa_blur, projection_shell=projection_shell,
                detail_iterations=detail_iterations,
                subdiv_levels=subdiv_levels,
                target_vertices=target_vertices,
                target_faces=target_faces,
                log=lambda message, level="info":
                    self.log_line.emit(message, level))
            self.reproject_done.emit(result)
        except Exception as ex:
            if self._cancel.is_set():
                self.reproject_failed.emit(f"Reproject cancelled.")
            else:
                self.reproject_failed.emit(f"Reproject failed: {ex}")
        finally:
            self._running = False
            self.state_changed.emit(False)


class PolishController(QObject):
    log_line = Signal(str, str)
    state_changed = Signal(bool)
    polish_done = Signal(object)
    polish_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._cancel = threading.Event()

    @property
    def running(self) -> bool:
        return self._running

    def cancel(self):
        """Ask the running job to stop (cooperative; takes effect at the
        next cancellation point inside the tool)."""
        self._cancel.set()

    def start(self, target: Path, *, amount: int,
              preserve_peaks: bool = False,
              vertices=None, faces=None):
        if self._running:
            return
        self._cancel.clear()
        self._running = True
        self.state_changed.emit(True)
        threading.Thread(
            target=self._job,
            args=(Path(target), int(amount), bool(preserve_peaks),
                  vertices, faces),
            daemon=True).start()

    def _job(self, target: Path, amount: int, preserve_peaks: bool,
             vertices, faces):
        try:
            # loaded fresh from disk every job (tools/ is hot-editable)
            from tools._loader import load_tool
            polish = load_tool("polish")
            polish_arrays = polish.polish_arrays

            if vertices is None or faces is None:
                import trimesh

                mesh = trimesh.load(str(target), force="mesh",
                                    process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError(
                        f"{target.name} did not load as one triangle mesh")
                vertices = mesh.vertices
                faces = mesh.faces
            result = polish_arrays(
                vertices, faces, amount, preserve_peaks=preserve_peaks,
                log=lambda message, level="info":
                    self.log_line.emit(message, level))
            result["target_path"] = str(target)
            self.polish_done.emit(result)
        except Exception as ex:
            if self._cancel.is_set():
                self.polish_failed.emit(f"Polish cancelled.")
            else:
                self.polish_failed.emit(f"Polish failed: {ex}")
        finally:
            self._running = False
            self.state_changed.emit(False)


class DecimateController(QObject):
    """Reduce the displayed mesh to a target triangle count."""

    log_line = Signal(str, str)
    state_changed = Signal(bool)
    decimate_done = Signal(object)
    decimate_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._cancel = threading.Event()

    @property
    def running(self) -> bool:
        return self._running

    def cancel(self):
        """Ask the running job to stop (cooperative; takes effect at the
        next cancellation point inside the tool)."""
        self._cancel.set()

    def start(self, target: Path, *, target_tris: int,
              vertices=None, faces=None):
        if self._running:
            return
        self._cancel.clear()
        self._running = True
        self.state_changed.emit(True)
        threading.Thread(
            target=self._job,
            args=(Path(target), int(target_tris), vertices, faces),
            daemon=True).start()

    def _job(self, target: Path, target_tris: int, vertices, faces):
        try:
            # loaded fresh from disk every job (tools/ is hot-editable)
            from tools._loader import load_tool
            decimate_arrays = load_tool("decimate").decimate_arrays

            if vertices is None or faces is None:
                import trimesh

                mesh = trimesh.load(str(target), force="mesh",
                                    process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError(
                        f"{target.name} did not load as one triangle mesh")
                vertices = mesh.vertices
                faces = mesh.faces
            result = decimate_arrays(
                vertices, faces, target_tris=target_tris,
                log=lambda message, level="info":
                    self.log_line.emit(message, level))
            result["target_path"] = str(target)
            self.decimate_done.emit(result)
        except Exception as ex:
            if self._cancel.is_set():
                self.decimate_failed.emit(f"Decimate cancelled.")
            else:
                self.decimate_failed.emit(f"Decimate failed: {ex}")
        finally:
            self._running = False
            self.state_changed.emit(False)


class SubdivideController(QObject):
    """Catmull-Clark subdivision of the displayed mesh, in memory."""

    log_line = Signal(str, str)
    state_changed = Signal(bool)
    subdivide_done = Signal(object)
    subdivide_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._cancel = threading.Event()

    @property
    def running(self) -> bool:
        return self._running

    def cancel(self):
        """Ask the running job to stop (cooperative; takes effect at the
        next cancellation point inside the tool)."""
        self._cancel.set()

    def start(self, target: Path, *, levels: int = 1,
              vertices=None, faces=None):
        if self._running:
            return
        self._cancel.clear()
        self._running = True
        self.state_changed.emit(True)
        threading.Thread(
            target=self._job,
            args=(Path(target), int(levels), vertices, faces),
            daemon=True).start()

    def _job(self, target: Path, levels: int, vertices, faces):
        try:
            # loaded fresh from disk every job (tools/ is hot-editable)
            from tools._loader import load_tool
            subdivide_arrays = load_tool("mesh_edit").subdivide_arrays

            if vertices is None or faces is None:
                import trimesh

                mesh = trimesh.load(str(target), force="mesh",
                                    process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError(
                        f"{target.name} did not load as one triangle mesh")
                vertices = mesh.vertices
                faces = mesh.faces
            result = subdivide_arrays(
                vertices, faces, levels=levels,
                log=lambda message, level="info":
                    self.log_line.emit(message, level))
            result["target_path"] = str(target)
            self.subdivide_done.emit(result)
        except Exception as ex:
            if self._cancel.is_set():
                self.subdivide_failed.emit("Subdivide cancelled.")
            else:
                self.subdivide_failed.emit(f"Subdivide failed: {ex}")
        finally:
            self._running = False
            self.state_changed.emit(False)


class CloseHolesController(QObject):
    """Detect and close the boundary holes of the displayed mesh."""

    log_line = Signal(str, str)
    state_changed = Signal(bool)
    close_holes_done = Signal(object)
    close_holes_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._cancel = threading.Event()

    @property
    def running(self) -> bool:
        return self._running

    def cancel(self):
        """Ask the running job to stop (cooperative; takes effect at the
        next cancellation point inside the tool)."""
        self._cancel.set()

    def start(self, target: Path, *, vertices=None, faces=None):
        if self._running:
            return
        self._cancel.clear()
        self._running = True
        self.state_changed.emit(True)
        threading.Thread(
            target=self._job,
            args=(Path(target), vertices, faces),
            daemon=True).start()

    def _job(self, target: Path, vertices, faces):
        try:
            # loaded fresh from disk every job (tools/ is hot-editable)
            from tools._loader import load_tool
            close_holes_arrays = load_tool("mesh_edit").close_holes_arrays

            if vertices is None or faces is None:
                import trimesh

                mesh = trimesh.load(str(target), force="mesh",
                                    process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError(
                        f"{target.name} did not load as one triangle mesh")
                vertices = mesh.vertices
                faces = mesh.faces
            # the filler consumes triangles; triangulate a quad edit first
            if getattr(faces, "ndim", 2) == 2 and faces.shape[1] == 4:
                import numpy as np
                faces = np.concatenate(
                    [faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)
            result = close_holes_arrays(
                vertices, faces,
                log=lambda message, level="info":
                    self.log_line.emit(message, level))
            result["target_path"] = str(target)
            self.close_holes_done.emit(result)
        except Exception as ex:  # noqa: BLE001 - report failures in the UI
            if self._cancel.is_set():
                self.close_holes_failed.emit("Close holes cancelled.")
            else:
                self.close_holes_failed.emit(f"Close holes failed: {ex}")
        finally:
            self._running = False
            self.state_changed.emit(False)


class DeleteFacesController(QObject):
    """Permanently delete the faces of the hidden polygroups."""

    log_line = Signal(str, str)
    state_changed = Signal(bool)
    delete_faces_done = Signal(object)
    delete_faces_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._cancel = threading.Event()

    @property
    def running(self) -> bool:
        return self._running

    def cancel(self):
        """Ask the running job to stop (cooperative)."""
        self._cancel.set()

    def start(self, target: Path, *, vertices=None, faces=None,
              keep_mask=None, face_groups=None):
        if self._running:
            return
        self._cancel.clear()
        self._running = True
        self.state_changed.emit(True)
        threading.Thread(
            target=self._job,
            args=(Path(target), vertices, faces, keep_mask, face_groups),
            daemon=True).start()

    def _job(self, target: Path, vertices, faces, keep_mask, face_groups):
        try:
            # loaded fresh from disk every job (tools/ is hot-editable)
            from tools._loader import load_tool
            delete_faces_arrays = load_tool("mesh_edit").delete_faces_arrays

            if vertices is None or faces is None or keep_mask is None:
                raise ValueError("no geometry/mask was provided")
            result = delete_faces_arrays(
                vertices, faces, keep_mask,
                log=lambda message, level="info":
                    self.log_line.emit(message, level))
            result["target_path"] = str(target)
            # one polygroup id per surviving face, same order as the
            # result faces (the tool never reorders kept faces)
            if face_groups is not None:
                result["face_groups"] = face_groups
            self.delete_faces_done.emit(result)
        except Exception as ex:  # noqa: BLE001 - report failures in the UI
            if self._cancel.is_set():
                self.delete_faces_failed.emit("Delete hidden cancelled.")
            else:
                self.delete_faces_failed.emit(f"Delete hidden failed: {ex}")
        finally:
            self._running = False
            self.state_changed.emit(False)


class MirrorMergeController(QObject):
    """Cut at the symmetry plane(s), keep one side, mirror and weld."""

    log_line = Signal(str, str)
    state_changed = Signal(bool)
    mirror_merge_done = Signal(object)
    mirror_merge_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._cancel = threading.Event()

    @property
    def running(self) -> bool:
        return self._running

    def cancel(self):
        """Ask the running job to stop (cooperative)."""
        self._cancel.set()

    def start(self, target: Path, *, planes, vertices=None, faces=None):
        if self._running:
            return
        self._cancel.clear()
        self._running = True
        self.state_changed.emit(True)
        threading.Thread(
            target=self._job,
            args=(Path(target), list(planes), vertices, faces),
            daemon=True).start()

    def _job(self, target: Path, planes, vertices, faces):
        try:
            # loaded fresh from disk every job (tools/ is hot-editable)
            from tools._loader import load_tool
            mirror_merge_arrays = load_tool("mesh_edit").mirror_merge_arrays

            if vertices is None or faces is None:
                import trimesh

                mesh = trimesh.load(str(target), force="mesh",
                                    process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError(
                        f"{target.name} did not load as one triangle mesh")
                vertices = mesh.vertices
                faces = mesh.faces
            result = mirror_merge_arrays(
                vertices, faces, planes,
                log=lambda message, level="info":
                    self.log_line.emit(message, level))
            result["target_path"] = str(target)
            self.mirror_merge_done.emit(result)
        except Exception as ex:  # noqa: BLE001 - report failures in the UI
            if self._cancel.is_set():
                self.mirror_merge_failed.emit("Mirror and merge cancelled.")
            else:
                self.mirror_merge_failed.emit(
                    f"Mirror and merge failed: {ex}")
        finally:
            self._running = False
            self.state_changed.emit(False)


class RemeshController(QObject):
    """Run the native Python quad remesher on in-memory geometry."""

    log_line = Signal(str, str)
    state_changed = Signal(bool)
    progress = Signal(int, str)
    remesh_done = Signal(object)
    remesh_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._cancel = threading.Event()

    @property
    def running(self) -> bool:
        return self._running

    def cancel(self):
        """Ask the running job to stop (cooperative; takes effect at the
        next cancellation point inside the tool)."""
        self._cancel.set()

    def start(self, target: Path, *, target_quads: int, symmetry: str,
              freeze_borders: bool, var_density_ratio: float,
              vertices=None, faces=None):
        if self._running:
            return
        self._cancel.clear()
        self._running = True
        self.state_changed.emit(True)
        threading.Thread(
            target=self._job,
            args=(Path(target), int(target_quads), str(symmetry),
                  bool(freeze_borders), float(var_density_ratio),
                  vertices, faces),
            daemon=True).start()

    def _load_rust_bridge(self):
        """Import tools/remesh_rust/bridge.py (compiled Rust remesher)."""
        import importlib.util

        root = Path(__file__).resolve().parents[2]
        path = root / "tools" / "remesh_rust" / "bridge.py"
        if not path.is_file():
            return None
        spec = importlib.util.spec_from_file_location(
            "remesh_rust_bridge", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not mod.remesh_available():
            return None
        return mod

    def _run_rust(self, target_quads: int, var_density_ratio: float,
                  seconds_started, vertices, faces):
        """Run the compiled Rust pipeline; None = unavailable/failed."""
        import time

        import numpy as np

        bridge = self._load_rust_bridge()
        if bridge is None:
            return None
        quads_target = (int(target_quads) if target_quads
                        and int(target_quads) > 0 else len(faces) // 2)
        quads_target = int(var_density_ratio ** 2 * quads_target)
        quads_target = int(min(150000, max(50, quads_target)))
        self.progress.emit(5, "rust pipeline")
        verts, quads, tris = bridge.remesh_quads(vertices, faces,
                                                 target_quads=quads_target)
        self.progress.emit(100, "done")
        seconds = time.perf_counter() - seconds_started
        out_v = np.asarray(verts, dtype=np.float64)
        q = np.asarray(quads, dtype=np.int64)
        t = np.asarray(tris, dtype=np.int64)
        parts = []
        if len(t):
            parts.append(t)
        if len(q):
            parts.append(q[:, [0, 1, 2]])
            parts.append(q[:, [0, 2, 3]])
        out_f = (np.concatenate(parts, axis=0) if parts
                 else np.zeros((0, 3), dtype=np.int64))
        return {
            "mode_label": "Remesh (rust)",
            "vertices": out_v,
            "faces": out_f,
            "vertex_count": int(len(out_v)),
            "face_count": int(len(out_f)),
            "input_vertex_count": int(len(vertices)),
            "input_face_count": int(len(faces)),
            "target_quads": quads_target,
            "symmetry": "none",
            "n_quads": int(len(q)),
            "n_tris": int(len(t)),
            "flips": 0,
            "seconds": round(seconds, 2),
            "backend": "rust",
        }

    def _job(self, target: Path, target_quads: int, symmetry: str,
             freeze_borders: bool, var_density_ratio: float,
             vertices, faces):
        started = __import__("time").perf_counter()
        try:
            if vertices is None or faces is None:
                import trimesh

                mesh = trimesh.load(str(target), force="mesh",
                                    process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError(
                        f"{target.name} did not load as one triangle mesh")
                vertices = mesh.vertices
                faces = mesh.faces

            # compiled Rust remesher only (tools/remesh_rust/remesh_rust.dll)
            if not self._load_rust_bridge():
                raise RuntimeError(
                    "the compiled remesher is unavailable "
                    "(tools/remesh_rust/remesh_rust.dll missing or not "
                    "loadable); rebuild with: cargo build --release "
                    "&& cp target/release/remesh_rust.dll .")
            result = self._run_rust(target_quads, var_density_ratio,
                                    started, vertices, faces)
            result["target_path"] = str(target)
            self.remesh_done.emit(result)
        except Exception as ex:
            if self._cancel.is_set():
                self.remesh_failed.emit(f"Remesh cancelled.")
            else:
                self.remesh_failed.emit(f"Remesh failed: {ex}")
        finally:
            self._running = False
            self.state_changed.emit(False)
