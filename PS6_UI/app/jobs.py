"""Background controllers plus pending-job bookkeeping."""

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from paths import OUTPUT_DIR
from workers.controllers import (
    CelloCutController, Hi3DController, Lato2Controller, ObjConverter,
    RmbgController, WorkerController,
)
from workers.edit_tools import (
    CloseHolesController, DecimateController, DeleteFacesController,
    MirrorMergeController, PolishController, RemeshController,
    ReprojectController, SubdivideController,
)


class JobCoordinator(QObject):
    """Owns the worker controllers and turns their results into signals."""

    log = Signal(str, str)
    changed = Signal()
    remesh_progress = Signal(int, str)
    cutout_ready = Signal(str, str)
    rmbg_finished = Signal(int)
    generation_done = Signal(object)
    generation_failed = Signal(str)
    hi3d_done = Signal(object)
    hi3d_failed = Signal(str)
    lato2_done = Signal(object)
    lato2_failed = Signal(str)
    cellocut_done = Signal(object)
    cellocut_failed = Signal(str)
    edit_done = Signal(str, object)
    edit_failed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.converter = ObjConverter(self)
        self.worker = WorkerController(self)
        self.hi3d = Hi3DController(self)
        self.lato2 = Lato2Controller(self)
        self.cellocut = CelloCutController(self)
        self.rmbg = RmbgController(OUTPUT_DIR / ".cutouts", self)
        self.reproject = ReprojectController(self)
        self.polish = PolishController(self)
        self.remesh = RemeshController(self)
        self.decimate = DecimateController(self)
        self.subdivide = SubdivideController(self)
        self.close_holes = CloseHolesController(self)
        self.delete_faces = DeleteFacesController(self)
        self.mirror_merge = MirrorMergeController(self)

        self.converter.log_line.connect(self.log)
        for ctrl in (self.worker, self.hi3d, self.lato2, self.cellocut,
                     self.reproject, self.polish, self.remesh, self.decimate,
                     self.subdivide, self.close_holes, self.delete_faces,
                     self.mirror_merge):
            ctrl.log_line.connect(self.log)
            ctrl.state_changed.connect(self._state_changed)
        self.rmbg.log_line.connect(lambda msg: self.log.emit(msg, "info"))
        self.rmbg.state_changed.connect(self._state_changed)
        self.rmbg.cutout_ready.connect(self.cutout_ready)
        self.rmbg.finished.connect(self._on_rmbg_finished)

        self.worker.generation_done.connect(self._on_generation_done)
        self.worker.generation_failed.connect(self._on_generation_failed)
        self.hi3d.refine_done.connect(self._on_hi3d_done)
        self.hi3d.refine_failed.connect(self._on_hi3d_failed)
        self.lato2.lato2_done.connect(self._on_lato2_done)
        self.lato2.lato2_failed.connect(self._on_lato2_failed)
        self.cellocut.cut_done.connect(self._on_cellocut_done)
        self.cellocut.cut_failed.connect(self._on_cellocut_failed)
        self.remesh.progress.connect(self.remesh_progress)

        self.reproject.reproject_done.connect(self._on_reproject_done)
        self.polish.polish_done.connect(self._on_polish_done)
        self.remesh.remesh_done.connect(self._on_remesh_done)
        self.decimate.decimate_done.connect(self._on_decimate_done)
        self.subdivide.subdivide_done.connect(self._on_subdivide_done)
        self.close_holes.close_holes_done.connect(self._on_close_holes_done)
        self.delete_faces.delete_faces_done.connect(
            self._on_delete_faces_done)
        self.mirror_merge.mirror_merge_done.connect(
            self._on_mirror_merge_done)
        for ctrl, failed, label in (
            (self.reproject, "reproject_failed", "Reproject"),
            (self.polish, "polish_failed", "Polish"),
            (self.remesh, "remesh_failed", "Remesh"),
            (self.decimate, "decimate_failed", "Decimate"),
            (self.subdivide, "subdivide_failed", "Subdivide"),
            (self.close_holes, "close_holes_failed", "Close holes"),
            (self.delete_faces, "delete_faces_failed", "Delete hidden"),
            (self.mirror_merge, "mirror_merge_failed",
             "Mirror and merge"),
        ):
            getattr(ctrl, failed).connect(
                lambda msg, label=label: self.edit_failed.emit(label, msg))

        self.pending_kind: str | None = None
        self.pending_state: Path | None = None
        self.pending_images: list[Path] | None = None
        self.scale_back: dict | None = None
        self._after_rmbg = None

    # -- pending job / helpers ---------------------------------------------
    @property
    def busy(self) -> bool:
        """True while any generation, repair or edit controller runs."""
        return (self.worker.generating or self.rmbg.running
                or self.hi3d.refining or self.lato2.running
                or self.cellocut.running
                or any(ctrl.running for ctrl in (
                    self.reproject, self.polish, self.remesh,
                    self.decimate, self.subdivide, self.close_holes,
                    self.delete_faces, self.mirror_merge)))

    def begin(self, kind: str, state_path: Path | None = None,
              images: list[Path] | None = None):
        self.pending_kind = kind
        self.pending_state = state_path
        self.pending_images = list(images) if images is not None else None

    def clear_pending(self):
        self.pending_kind = None
        self.pending_state = None
        self.pending_images = None

    def queue_after_rmbg(self, start):
        self._after_rmbg = start

    def cancel_after_rmbg(self) -> bool:
        had = self._after_rmbg is not None
        self._after_rmbg = None
        return had

    def convert(self, glb: Path):
        self.converter.convert(glb)

    def shutdown(self):
        for ctrl in (self.rmbg, self.hi3d, self.lato2, self.cellocut,
                     self.worker):
            ctrl.shutdown()

    def _state_changed(self, _busy):
        self.changed.emit()

    # -- generation / refine / repair --------------------------------------
    def _on_generation_done(self, output, seed=None, scale_back=None):
        out = Path(output)
        kind = self.pending_kind
        state_path = self.pending_state
        images = self.pending_images
        self.clear_pending()
        if kind == "stage2":
            self.scale_back = None
            if isinstance(scale_back, dict) and scale_back.get("scaled"):
                info = dict(scale_back)
                info["output"] = str(out)
                self.scale_back = info
        self.generation_done.emit({
            "output": out, "seed": seed, "kind": kind,
            "state_path": state_path, "images": images,
            "scale_back": scale_back,
        })

    def _on_generation_failed(self, msg: str):
        self.clear_pending()
        self.generation_failed.emit(msg)

    def _on_hi3d_done(self, output, seed=None):
        self.clear_pending()
        self.hi3d_done.emit({"output": Path(output), "seed": seed})

    def _on_hi3d_failed(self, msg: str):
        self.clear_pending()
        self.hi3d_failed.emit(msg)

    def _on_lato2_done(self, output, seed=None):
        self.clear_pending()
        self.lato2_done.emit({"output": Path(output), "seed": seed})

    def _on_lato2_failed(self, msg: str):
        self.clear_pending()
        self.lato2_failed.emit(msg)

    def _on_cellocut_done(self, output, seed=None):
        self.clear_pending()
        self.cellocut_done.emit({"output": Path(output)})

    def _on_cellocut_failed(self, msg: str):
        self.clear_pending()
        self.cellocut_failed.emit(msg)

    def _on_rmbg_finished(self, code: int):
        self.rmbg_finished.emit(code)
        start, self._after_rmbg = self._after_rmbg, None
        if start is not None:
            if code != 0:
                self.log.emit(
                    "Continuing with generation; the pipeline will remove "
                    "backgrounds for any missing cutouts.", "warn")
            start()

    # -- edit results --------------------------------------------------------
    def _on_reproject_done(self, result):
        result = result or {}
        try:
            vertices = result["vertices"]
        except (KeyError, TypeError):
            self.edit_failed.emit("Reproject",
                                  "Reproject returned no geometry.")
            return
        moved = int(result.get("moved", 0))
        total = int(result.get("vertex_count", len(vertices)))
        mode_label = result.get("mode_label", "Reproject")
        if moved:
            message = (
                f"Reproject ({mode_label}) finished: {moved:,}/{total:,} "
                f"vertices moved (strength "
                f"{result.get('strength', 1.0) * 100:.0f}%, tolerance "
                f"{result.get('tolerance', 0.0):.6g}, max "
                f"{result.get('max_displacement', 0.0):.6g}).")
            if result.get("subdiv_levels"):
                message += (f" Topology subdivided x{result['subdiv_levels']}: "
                            f"{result.get('input_vertex_count', 0):,} -> "
                            f"{total:,} vertices.")
            self.log.emit(message, "ok")
        else:
            self.log.emit(f"Reproject ({mode_label}) finished without moving "
                          "any vertex; the displayed mesh is unchanged.",
                          "warn")
        self.edit_done.emit(mode_label, result)

    def _on_polish_done(self, result):
        result = result or {}
        try:
            vertices = result["vertices"]
        except (KeyError, TypeError):
            self.edit_failed.emit("Polish", "Polish returned no geometry.")
            return
        total = int(result.get("vertex_count", len(vertices)))
        moved = int(result.get("moved", 0))
        label = result.get("mode_label", "Polish")
        self.log.emit(f"{label} finished: moved {moved:,}/{total:,} vertices "
                      f"(n={result.get('iterations', 0)}, passes="
                      f"{result.get('passes', 0)}, max "
                      f"{result.get('max_displacement', 0.0):.6g}, mean "
                      f"{result.get('mean_displacement', 0.0):.6g}).", "ok")
        self.edit_done.emit(label, result)

    def _on_remesh_done(self, result):
        result = result or {}
        try:
            vertices = result["vertices"]
            faces = result["faces"]
        except (KeyError, TypeError):
            self.edit_failed.emit("Remesh", "Remesh returned no geometry.")
            return
        self.log.emit(
            f"Remesh finished: {result.get('input_vertex_count', 0):,} "
            f"vertices / {result.get('input_face_count', 0):,} triangles "
            f"-> {len(vertices):,} vertices / {len(faces):,} triangles "
            f"({result.get('n_quads', 0):,} quads, target "
            f"{result.get('target_quads', 0):,}, "
            f"{result.get('seconds', 0.0):.2f}s).", "ok")
        self.edit_done.emit(result.get("mode_label", "Remesh"), result)

    def _on_decimate_done(self, result):
        result = result or {}
        try:
            vertices = result["vertices"]
            faces = result["faces"]
        except (KeyError, TypeError):
            self.edit_failed.emit("Decimate",
                                  "Decimate returned no geometry.")
            return
        self.log.emit(
            f"Decimate finished: {result.get('input_face_count', 0):,} "
            f"-> {len(faces):,} triangles ({len(vertices):,} vertices, "
            f"{result.get('backend', '?')}) in "
            f"{result.get('seconds', 0):.1f}s.", "ok")
        self.edit_done.emit(result.get("mode_label", "Decimate"), result)

    def _on_subdivide_done(self, result):
        result = result or {}
        try:
            vertices = result["vertices"]
            faces = result["faces"]
        except (KeyError, TypeError):
            self.edit_failed.emit("Subdivide",
                                  "Subdivide returned no geometry.")
            return
        self.log.emit(
            f"Subdivide finished: {result.get('input_face_count', 0):,} "
            f"-> {len(faces):,} quads ({len(vertices):,} vertices) in "
            f"{result.get('seconds', 0):.2f}s.", "ok")
        self.edit_done.emit(result.get("mode_label", "Subdivide"), result)

    def _on_close_holes_done(self, result):
        result = result or {}
        try:
            vertices = result["vertices"]
            faces = result["faces"]
        except (KeyError, TypeError):
            self.edit_failed.emit("Close holes",
                                  "Close holes returned no geometry.")
            return
        if result.get("watertight"):
            self.log.emit(
                "Close holes finished: mesh is now watertight "
                f"({result.get('triangles_added', 0):,} cap triangles added "
                f"via {result.get('engine', '?')}) in "
                f"{result.get('seconds', 0):.2f}s.", "ok")
        else:
            self.log.emit(
                "Close holes finished with "
                f"{result.get('open_edges_remaining', 0)} boundary edge(s) "
                "still open (non-manifold boundary); "
                f"{result.get('triangles_added', 0):,} cap triangles were "
                "added.", "warn")
        self.edit_done.emit(result.get("mode_label", "Close holes"), result)

    def _on_delete_faces_done(self, result):
        result = result or {}
        try:
            vertices = result["vertices"]
            faces = result["faces"]
        except (KeyError, TypeError):
            self.edit_failed.emit("Delete hidden",
                                  "Delete hidden returned no geometry.")
            return
        self.log.emit(
            "Delete hidden finished: "
            f"{result.get('input_face_count', 0):,} -> "
            f"{result.get('face_count', 0):,} faces "
            f"({result.get('input_vertex_count', 0):,} -> "
            f"{result.get('vertex_count', 0):,} vertices) in "
            f"{result.get('seconds', 0):.2f}s. Press Bake changes to write "
            "the deletion into the model file.", "ok")
        self.edit_done.emit(result.get("mode_label", "Delete hidden"), result)

    def _on_mirror_merge_done(self, result):
        result = result or {}
        try:
            vertices = result["vertices"]
            faces = result["faces"]
        except (KeyError, TypeError):
            self.edit_failed.emit("Mirror and merge",
                                  "Mirror and merge returned no geometry.")
            return
        self.log.emit(
            "Mirror and merge finished: "
            f"{result.get('input_vertex_count', 0):,} -> "
            f"{len(vertices):,} vertices, "
            f"{result.get('input_face_count', 0):,} -> "
            f"{len(faces):,} faces in "
            f"{result.get('seconds', 0):.2f}s. Press Bake changes to write "
            "the merged mesh into the model file.", "ok")
        self.edit_done.emit(result.get("mode_label", "Mirror and merge"),
                            result)
