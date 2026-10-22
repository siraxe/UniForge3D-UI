"""Generation, upscale, repair and refinement workflows."""

import shutil
import time
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

from dialogs import stage2_rescale_enabled as _stage2_rescale_enabled
from model_paths import missing_hi3d_assets, missing_lato2_assets
from paths import (
    MODEL_DIR,
    OUTPUT_DIR,
    ROOT,
    _cellocut_output_path,
    _hi3d_output_path,
    _lato2_output_path,
    _paths_equal,
    _shape512_state_path,
    _shape_state_path,
    _stage2_output_path,
    _textured_output_path,
)


class Pipeline(QObject):
    """Runs the model pipeline and reacts to its finished jobs."""

    log = Signal(str, str)
    changed = Signal()

    def __init__(self, window, viewer, jobs, models, edits, app, left,
                 parent=None):
        super().__init__(parent)
        self.window = window
        self.viewer = viewer
        self.jobs = jobs
        self.models = models
        self.edits = edits
        self.app = app
        self.left = left

    # -- generation ----------------------------------------------------------
    def generate(self):
        if self.jobs.busy:
            return
        if not self.app.image_paths:
            QMessageBox.information(self.window, "UniForge3D",
                                    "Add an input image first.")
            return
        sources = list(self.app.image_paths)
        self.queue_after_cutouts(
            sources, lambda: self._start_generate(sources))

    def _start_generate(self, sources: list[Path]):
        sources = [p for p in sources if p in self.app.image_paths]
        if not sources:
            self.log.emit("No input images left; generation cancelled.", "warn")
            return
        images = [self.app.cutout_map.get(p, p) for p in sources]
        ready = sum(1 for p in sources if p in self.app.cutout_map)
        if ready:
            self.log.emit(f"Using pre-removed backgrounds for "
                          f"{ready}/{len(sources)} image(s) - RMBG-2.0 will "
                          "be skipped for them.", "info")
        self.models.clear_stage1()
        geometry_only = self.left.mode_combo.currentIndex() == 1
        stop_at_stage1 = self.left.stop_stage1_check.isChecked()
        geometry_only = geometry_only or stop_at_stage1
        base_name = sources[0].stem + "_" + time.strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_DIR / base_name
        output_name = ("prediction_512.glb" if stop_at_stage1
                       else "prediction.glb")
        suffix = 2
        while any((out_dir / name).exists()
                  for name in ("prediction.glb", "prediction_512.glb")):
            out_dir = OUTPUT_DIR / f"{base_name}_{suffix}"
            suffix += 1
        output_path = out_dir / output_name
        self._copy_input_images(out_dir, sources)
        state_path = (_shape_state_path(output_path)
                      if geometry_only and not stop_at_stage1 else None)
        self.jobs.begin("stage1" if stop_at_stage1 else
                        "geometry" if geometry_only else "textured",
                        state_path, images)
        self.changed.emit()
        self.jobs.worker.start_run(
            images, output_path,
            seed=(None if self.left.seed_spin.value() < 0
                  else self.left.seed_spin.value()),
            grid_size=int(self.left.grid_combo.currentText()),
            texture_size=int(self.left.tex_combo.currentText()),
            steps=self.left.steps_spin.value(),
            texture=not geometry_only,
            shape_state=state_path,
            stop_at_stage1=stop_at_stage1,
        )
        self.log.emit("Generation queued.", "info")

    def _copy_input_images(self, out_dir: Path, sources: list[Path]):
        """Copy the images used by this run into out_dir/inputs."""
        inputs_dir = out_dir / "inputs"
        used = 0
        try:
            for src in sources:
                if not src.is_file():
                    continue
                inputs_dir.mkdir(parents=True, exist_ok=True)
                stem, n = src.stem, 2
                target = inputs_dir / f"{stem}_src{src.suffix}"
                while target.exists():
                    target = inputs_dir / f"{stem}_{n}_src{src.suffix}"
                    n += 1
                shutil.copy2(src, target)
                used += 1
                cutout = self.app.cutout_map.get(src)
                if cutout and cutout.is_file():
                    base = (target.stem[:-4] if target.stem.endswith("_src")
                            else target.stem)
                    shutil.copy2(cutout, inputs_dir / f"{base}_cutout.png")
            if used:
                self.log.emit(f"Copied {used} input image(s) to {inputs_dir}.",
                              "info")
        except OSError as ex:
            self.log.emit("Could not copy input images into the run folder: "
                          f"{ex}", "warn")

    def open_folder(self, folder: Path):
        """Load a project folder's images, then its models."""
        folder = Path(folder)
        self.jobs.clear_pending()
        loaded = self.app.load_folder(folder)
        if loaded is not None:
            count, restored, inputs = loaded
            where = "inputs subfolder" if inputs else "folder"
            note = f"; {restored} cutout(s) restored." if restored else "."
            self.log.emit(f"Loaded {count} input image(s) from the project "
                          f"{where}{note}", "info")
        self.models.open_folder(folder)
        self.changed.emit()

    def remove_background(self):
        """Cut out all imported images with RMBG-2.0 in a light process."""
        if self.jobs.busy:
            return
        todo = [p for p in self.app.image_paths
                if p not in self.app.cutout_map]
        if not todo:
            self.log.emit("All imported images already have a removed "
                          "background.", "ok")
            return
        self.log.emit(f"Removing backgrounds for {len(todo)} image(s) "
                      "(loads RMBG-2.0 only, not the full model) ...", "info")
        self.changed.emit()
        self.jobs.rmbg.start(todo)

    def queue_after_cutouts(self, sources: list[Path],
                            start: Callable[[], None],
                            what: str = "generation") -> None:
        """Run RMBG for missing cutouts, then invoke *start*."""
        todo = [p for p in sources if p not in self.app.cutout_map]
        if not todo:
            start()
            return
        self.log.emit(f"Removing backgrounds for {len(todo)} image(s) before "
                      f"{what} ...", "info")
        self.jobs.queue_after_rmbg(start)
        self.changed.emit()
        try:
            self.jobs.rmbg.start(todo)
        except Exception as ex:  # keep generation possible if RMBG cannot start
            self.jobs.cancel_after_rmbg()
            self.log.emit(f"Background removal could not start: {ex}", "error")
            self.log.emit("Starting generation; the pipeline will remove "
                          "backgrounds internally.", "warn")
            start()

    def generate_texture(self):
        if self.jobs.busy:
            return
        state = self.models.texture_state
        if state is None or not state.is_file():
            QMessageBox.information(
                self.window, "UniForge3D",
                "Run a Geometry only generation first; Generate Texture "
                "reuses its saved Shape-1024 state.")
            return
        if self.models.texture_source is None:
            QMessageBox.warning(self.window, "UniForge3D",
                                "The geometry output path is missing.")
            return
        if not self.app.image_paths:
            QMessageBox.information(
                self.window, "UniForge3D",
                "Add the input images used for the geometry pass.")
            return
        sources = list(self.app.image_paths)
        self.queue_after_cutouts(
            sources, lambda: self._start_generate_texture(sources),
            what="texture generation")

    def _start_generate_texture(self, sources: list[Path]):
        sources = [p for p in sources if p in self.app.image_paths]
        if not sources:
            self.log.emit("No input images left; texture generation "
                          "cancelled.", "warn")
            return
        images = [self.app.cutout_map.get(p, p) for p in sources]
        output_path = _textured_output_path(self.models.texture_source)
        self._copy_input_images(output_path.parent, sources)
        self.jobs.begin("texture")
        self.changed.emit()
        self.jobs.worker.start_texture(
            images, output_path, self.models.texture_state,
            seed=(None if self.left.seed_spin.value() < 0
                  else self.left.seed_spin.value()),
            texture_size=int(self.left.tex_combo.currentText()),
            steps=self.left.steps_spin.value(),
        )
        self.log.emit(f"Texture generation queued (size "
                      f"{self.left.tex_combo.currentText()}); reusing "
                      f"{self.models.texture_source.name}.", "info")

    # -- 2nd stage -----------------------------------------------------------
    def run_stage2(self):
        if self.jobs.busy:
            self.changed.emit()
            return
        picked = self._stage2_source()
        if picked is None:
            self.changed.emit()
            return
        loaded, shape_source = picked
        images = self._stage2_images()
        if not images:
            return
        output_path = _stage2_output_path(loaded)
        state_path = _shape_state_path(output_path)
        original = {v.resolve(): k for k, v in self.app.cutout_map.items()}
        sources = [original.get(p.resolve(), p) for p in images]

        def start():
            ready = [self.app.cutout_map.get(p, p) for p in sources]
            self.jobs.scale_back = None
            self.jobs.begin("stage2", state_path, ready)
            self.changed.emit()
            self.jobs.worker.start_stage2(
                ready, output_path, shape_source,
                seed=(None if self.left.seed_spin.value() < 0
                      else self.left.seed_spin.value()),
                grid_size=int(self.left.grid_combo.currentText()),
                texture_size=int(self.left.tex_combo.currentText()),
                steps=self.left.steps_spin.value(),
                texture=False,
                shape_state=state_path,
            )
            self.log.emit(f"Running 2nd stage: Shape-1024 upscale from "
                          f"{loaded.name} ({shape_source.name}; stage 1 is "
                          "not re-run) ...", "info")

        self.queue_after_cutouts(sources, start, what="the 2nd stage")

    def _stage2_source(self):
        """Loaded model + the Shape-512 checkpoint to upscale, or None."""
        loaded = self.viewer.current_path
        if loaded is None or not loaded.is_file():
            QMessageBox.information(
                self.window, "UniForge3D",
                "Load or generate a model first; Run 2nd stage upscales the "
                "displayed mesh.")
            return None
        if self.left.upscale_scene.isChecked():
            sidecar = self._write_shape512_from_scene(
                loaded, edit=self.edits.pending)
            if sidecar is None:
                QMessageBox.information(
                    self.window, "UniForge3D",
                    "The mesh in the scene could not be exported to a "
                    "Shape-512 checkpoint; save it first, then run the "
                    "2nd stage.")
                return None
            return loaded, sidecar
        sidecar = _shape512_state_path(loaded)
        if not sidecar.is_file():
            QMessageBox.information(
                self.window, "UniForge3D",
                f"No {sidecar.name} found next to {loaded.name}.\n\n"
                "Tick 'Apply to mesh in scene' to upscale the displayed "
                "mesh, or run an Early stop / CelloCut result first.")
            return None
        if loaded.stat().st_mtime - sidecar.stat().st_mtime >= 10:
            self.log.emit(f"{sidecar.name} is older than {loaded.name}; tick "
                          "'Apply to mesh in scene' to upscale the displayed "
                          "mesh instead.", "warn")
        if self.edits.pending is not None:
            self.log.emit("2nd stage: an unsaved edit is pending but 'Apply "
                          "to mesh in scene' is off - the checkpoint on disk "
                          "(not the edit) will be upscaled.", "warn")
        return loaded, sidecar

    def _stage2_images(self):
        """Conditioning images: the Early stop run's, imported ones, or a pick."""
        stage1_images = self.models.stage1_images
        if stage1_images:
            images = [p for p in stage1_images if p.exists()]
            if images:
                return images
        sources = [p for p in self.app.image_paths if p.exists()]
        if sources:
            images = [self.app.cutout_map.get(p, p) for p in sources]
            ready = sum(1 for p in sources if p in self.app.cutout_map)
            if ready:
                self.log.emit(f"2nd stage: using pre-removed backgrounds for "
                              f"{ready}/{len(sources)} image(s).", "info")
            return images
        files, _ = QFileDialog.getOpenFileNames(
            self.window, "Select input image(s) for the 2nd stage",
            str(OUTPUT_DIR if OUTPUT_DIR.is_dir() else ROOT),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not files:
            return None
        return [Path(f) for f in files]

    def _write_shape512_from_scene(self, loaded: Path,
                                   edit: dict | None) -> Path | None:
        """Write a Shape-512 checkpoint from the mesh currently displayed."""
        import os

        import numpy as np
        try:
            import torch
        except ImportError:
            return None
        if edit is not None:
            vertices = np.asarray(edit.get("vertices"), dtype=np.float32)
            faces = np.asarray(edit.get("faces"), dtype=np.int64)
            target = loaded.with_name(loaded.stem + "_scene.shape512.pt")
        else:
            try:
                import trimesh
                mesh = trimesh.load(str(loaded), force="mesh",
                                    process=False, merge_primitives=True)
            except Exception:
                return None
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                return None
            vertices = mesh.vertices.astype(np.float32, copy=False)
            faces = mesh.faces.astype(np.int64, copy=False)
            target = _shape512_state_path(loaded)
        if (vertices.ndim != 2 or vertices.shape[1] != 3
                or faces.ndim != 2 or faces.shape[1] != 3
                or len(faces) == 0):
            return None
        payload = {
            "format": 1,
            "vertices": torch.from_numpy(np.ascontiguousarray(vertices)),
            "faces": torch.from_numpy(np.ascontiguousarray(faces)),
        }
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, target)
        except OSError as ex:
            self.log.emit(f"Could not write {target.name}: {ex}", "warn")
            return None
        source_label = ("the unsaved edit" if edit is not None
                        else loaded.name)
        self.log.emit(f"Wrote {target.name} from {source_label}; the 2nd "
                      "stage will upscale exactly this mesh.", "info")
        return target

    def _auto_stage2_rescale(self, info: dict):
        """Apply the inverse normalization and save the 2nd-stage result."""
        loaded = self.viewer.current_path
        target = None
        try:
            target = Path(info.get("output"))
        except (TypeError, ValueError):
            return
        if (not info.get("scaled") or loaded is None or not loaded.is_file()
                or target is None or not _paths_equal(loaded, target)):
            return
        vertices, faces = self.edits.arrays_for(loaded)
        if vertices is None or faces is None:
            try:
                import trimesh
                mesh = trimesh.load(str(loaded), force="mesh",
                                    process=False, merge_primitives=True)
            except Exception as ex:  # noqa: BLE001 - report in the UI
                self.log.emit(f"Auto rescale: could not read {loaded.name}: "
                              f"{ex}", "error")
                return
            if not isinstance(mesh, trimesh.Trimesh):
                self.log.emit(f"Auto rescale: {loaded.name} did not load as "
                              "one mesh.", "error")
                return
            vertices, faces = mesh.vertices, mesh.faces
        try:
            from tools._loader import load_tool
            result = load_tool("mesh_edit").scale_back_arrays(
                vertices, faces, info=info,
                log=lambda message, level="info": self.log.emit(message, level))
        except Exception as ex:  # noqa: BLE001 - report in the UI
            self.log.emit(f"Auto rescale failed: {ex}", "error")
            return
        result["target_path"] = str(loaded)
        factor = float(info.get("scale_back_factor", 1.0))
        offset = info.get("translation_display") or [0.0, 0.0, 0.0]
        self.log.emit(
            f"Auto rescale: restoring the source size/location "
            f"(x{factor:.6g}, offset "
            f"[{offset[0]:g}, {offset[1]:g}, {offset[2]:g}]) and saving ...",
            "info")
        if self.edits.accept_result("Rescale", result):
            self.jobs.scale_back = None
            self.edits.save(replace=True)

    # -- Hi3D / CelloCut -----------------------------------------------------
    def _hi3d_condition_image(self) -> Path | None:
        stage1_images = self.models.stage1_images
        if not stage1_images:
            return None
        run_images = {p.resolve() for p in stage1_images}
        if self.app.ref_reference is not None:
            source = self.app.ref_reference
            candidate = self.app.cutout_map.get(source, source)
            for path in (candidate, source):
                if path.resolve() in run_images:
                    return path
            self.log.emit("The selected REF image was not part of the Early "
                          "stop run; using that run's first image for Hi3D.",
                          "warn")
        return stage1_images[0]

    def run_hi3d(self):
        """Run Hi3D v1.0 with the passes chosen in the dropdown."""
        selection = self.left.hi3d_pass_combo.currentData() or (None, 1536)
        self._run_hi3d(start_structure=selection[0],
                       resolution=selection[1])

    def _run_hi3d(self, *, start_structure, resolution=1536):
        if self.jobs.busy:
            return
        loaded = self.viewer.current_path
        if loaded is None or not loaded.is_file():
            QMessageBox.information(
                self.window, "UniForge3D",
                "Load or generate a model first; Hi3D 1536 refines the "
                "displayed mesh.")
            self.changed.emit()
            return
        image = self._hi3d_condition_image()
        if image is None:
            file, _ = QFileDialog.getOpenFileName(
                self.window, "Select the reference image for Hi3D",
                str(OUTPUT_DIR if OUTPUT_DIR.is_dir() else ROOT),
                "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
            if not file:
                return
            image = Path(file)
        if not image.is_file():
            QMessageBox.warning(
                self.window, "UniForge3D",
                f"Reference image not found:\n{image}")
            return
        missing = missing_hi3d_assets(MODEL_DIR)
        if missing:
            self.log.emit("Hi3D is not ready; missing: " + "; ".join(missing),
                          "error")
            shown = "\n".join(missing[:8])
            if len(missing) > 8:
                shown += f"\n... and {len(missing) - 8} more"
            QMessageBox.warning(
                self.window, "Hi3D is not ready",
                "Hi3D 1536 needs the Hi3D v1.0 source and its models.\n"
                "Run get_models.bat first.\n\nMissing:\n" + shown)
            return
        output_path = _hi3d_output_path(loaded, resolution)
        if self.jobs.worker.alive:
            self.jobs.worker.stop()
            self.log.emit("UniForge3D worker unloaded to free GPU memory for "
                          "Hi3D.", "info")
        self.jobs.begin("hi3d")
        self.changed.emit()
        self.jobs.hi3d.start(
            image, loaded, output_path, resolution=resolution,
            seed=(None if self.left.seed_spin.value() < 0
                  else self.left.seed_spin.value()),
            start_structure=start_structure,
            steps=self.left.hi3d_steps.value(),
            budget_percent=self.left.hi3d_budget.value(),
            low_vram=(False if self.left.hi3d_fullgpu.isChecked() else None))
        structures = (64, 128, 192)
        if start_structure is None:
            passes = "passes 1-3 (structures 64 > 128 > 192)"
        else:
            first = structures.index(start_structure) + 1
            last = (512, 1024, 1536).index(resolution) + 1
            skipped = "; earlier passes skipped" if first > 1 else ""
            if first == last:
                passes = (f"pass {first} only (structure {start_structure}"
                          f"{skipped})")
            else:
                passes = (f"passes {first}-{last} (structures "
                          f"{' > '.join(str(s) for s in structures[first - 1:last])}"
                          f"{skipped})")
        self.log.emit(
            f"Running Hi3D ({passes}, output {resolution}): refining "
            f"{loaded.name} with {image.name} "
            f"({self.left.hi3d_steps.value()} steps/pass, voxel budget "
            f"{self.left.hi3d_budget.value()}%, "
            f"{'full GPU residency' if self.left.hi3d_fullgpu.isChecked() else 'global memory mode'}, "
            "geometry only) ...", "info")

    def run_lato2(self):
        """Regenerate the displayed mesh with LATO.2 (retopology)."""
        if self.jobs.busy:
            return
        loaded = self.viewer.current_path
        if loaded is None or not loaded.is_file():
            QMessageBox.information(
                self.window, "UniForge3D",
                "Load or generate a model first; LATO.2 retopologizes the "
                "displayed mesh.")
            self.changed.emit()
            return
        missing = missing_lato2_assets(MODEL_DIR)
        if missing:
            self.log.emit("LATO.2 is not ready; missing: "
                          + "; ".join(missing), "error")
            shown = "\n".join(missing[:8])
            if len(missing) > 8:
                shown += f"\n... and {len(missing) - 8} more"
            QMessageBox.warning(
                self.window, "LATO.2 is not ready",
                "LATO.2 needs the LATO.2 source and its checkpoints.\n"
                "Run get_models.bat (LATO.2) first.\n\nMissing:\n" + shown)
            return
        output_path = _lato2_output_path(loaded)
        state_path = _shape512_state_path(output_path)
        if self.jobs.worker.alive:
            self.jobs.worker.stop()
            self.log.emit("UniForge3D worker unloaded to free GPU memory for "
                          "LATO.2.", "info")
        panel = self.window.right_panel
        vert_num = int(panel.lato2_verts.value())
        fill_rings = panel.lato2_fill_rings.isChecked()
        drop_tiny = panel.lato2_drop_tiny.isChecked()
        self.jobs.begin("lato2")
        self.changed.emit()
        self.jobs.lato2.start(
            loaded, output_path, vert_num=vert_num,
            seed=(None if self.left.seed_spin.value() < 0
                  else self.left.seed_spin.value()),
            fill_quad_rings=fill_rings, drop_tiny_patches=drop_tiny,
            shape512_state=state_path)
        self.log.emit(
            f"Running LATO.2: retopologizing {loaded.name} with a target of "
            f"{vert_num:,} vertices (V-Flow + T-Flow, geometry only; "
            f"quad rings {'on' if fill_rings else 'off'}, "
            f"drop small patches {'on' if drop_tiny else 'off'}) ...",
            "info")

    def run_cellocut(self):
        """Watertight-repair the displayed mesh with CelloCut."""
        if self.jobs.busy:
            return
        loaded = self.viewer.current_path
        if loaded is None or not loaded.is_file():
            QMessageBox.information(
                self.window, "UniForge3D",
                "Load or generate a model first; CelloCut repairs the "
                "displayed mesh.")
            self.changed.emit()
            return
        output_path = _cellocut_output_path(loaded)
        state_path = _shape512_state_path(output_path)
        if self.jobs.worker.alive:
            self.jobs.worker.stop()
            self.log.emit("UniForge3D worker unloaded to free GPU memory for "
                          "CelloCut.", "info")
        self.jobs.begin("cellocut")
        self.changed.emit()
        self.jobs.cellocut.start(loaded, output_path, shape512_state=state_path)
        self.log.emit(f"Running CelloCut watertight repair on "
                      f"{loaded.name} (512^3 UDF, graph cut, "
                      "geometry only) ...", "info")

    def set_low_vram(self, checked: bool):
        self.jobs.worker.set_low_vram(checked)
        self.jobs.hi3d.set_low_vram(checked)
        message = (
            "Low VRAM enabled: only the active stage stays on the GPU."
            if checked else
            "Low VRAM disabled: full-precision weights; needs substantially "
            "more GPU memory.")
        if self.jobs.worker.alive and not self.jobs.worker.generating:
            self.jobs.worker.stop()
            message += " Worker will reload on the next run."
        self.log.emit(message, "info")

    # -- stop / job events ---------------------------------------------------
    def stop(self):
        cellocut, hi3d, lato2, worker = (
            self.jobs.cellocut, self.jobs.hi3d, self.jobs.lato2,
            self.jobs.worker)
        if cellocut.running:
            cellocut.stop()
            self.jobs.clear_pending()
            self.log.emit("CelloCut repair stopped (worker killed); the 512 "
                          "mesh is unchanged.", "warn")
            self.changed.emit()
            return
        if lato2.running:
            lato2.stop()
            self.jobs.clear_pending()
            self.log.emit("LATO.2 generation stopped (worker killed); the "
                          "source mesh is unchanged.", "warn")
            self.changed.emit()
            return
        if hi3d.refining:
            hi3d.stop()
            self.jobs.clear_pending()
            self.log.emit("Hi3D refinement stopped (worker killed); Hi3D "
                          "reloads on the next refinement.", "warn")
            self.changed.emit()
            return
        if not worker.generating:
            return
        worker.stop()
        self.jobs.clear_pending()
        self.log.emit("Generation stopped (worker killed); the model reloads "
                      "on the next run.", "warn")
        self.changed.emit()

    def on_cutout_ready(self, source: str, processed: str):
        self.app.set_cutout(source, processed)
        self.log.emit(f"Cutout ready: {Path(processed).name}", "ok")

    def on_rmbg_finished(self, code: int):
        ready = sum(1 for p in self.app.image_paths
                    if p in self.app.cutout_map)
        if code != 0:
            self.log.emit(f"Background removal process failed (exit code "
                          f"{code}) - see terminal for details.", "error")
        else:
            self.log.emit(f"Background removal finished "
                          f"({ready}/{len(self.app.image_paths)} ready); "
                          "RMBG-2.0 unloaded.", "ok")
        self.changed.emit()

    def on_generation_failed(self, msg: str):
        self._failed("Generation failed", msg)

    def on_hi3d_failed(self, msg: str):
        self._failed("Hi3D refinement failed", msg)

    def on_lato2_failed(self, msg: str):
        self._failed("LATO.2 mesh generation failed", msg)

    def on_cellocut_failed(self, msg: str):
        self._failed("CelloCut repair failed", msg)

    def _failed(self, title: str, msg: str):
        self.log.emit(msg, "error")
        QMessageBox.warning(self.window, title, msg)
        self.changed.emit()

    def on_generation_done(self, data: dict):
        out = data["output"]
        kind = data["kind"]
        state_path = data["state_path"]
        pending_images = data["images"]
        scale_back = data["scale_back"]
        self.log.emit(f"Done -> {out}", "ok")
        self.app.set_seed(data["seed"])
        self.models.track_output(out)
        if kind == "stage1":
            self.models.set_texture_state(None, None)
            self.log.emit(f"Shape-512 preview finished ({out.name}); 1024 "
                          "upscale and texture were skipped.", "info")
            stage1_state = _shape512_state_path(out)
            if stage1_state.is_file() and pending_images:
                self.models.set_stage1_state(stage1_state, out, pending_images)
                self.log.emit("Run 2nd stage is available; it will upscale "
                              "this Shape-512 mesh with Shape-1024.", "ok")
            else:
                self.models.clear_stage1()
                self.log.emit("The Shape-512 state was not saved; Run 2nd "
                              "stage is unavailable.", "warn")
        elif kind == "geometry":
            if state_path is not None and state_path.is_file():
                self.models.set_texture_state(state_path, out)
                self.log.emit("Geometry saved; Generate Texture is available.",
                              "ok")
            else:
                self.models.set_texture_state(None, None)
                self.log.emit("Geometry finished, but its texture state was "
                              "not saved; Generate Texture is unavailable.",
                              "warn")
        elif kind == "textured":
            self.models.set_texture_state(None, None)
        elif kind == "stage2":
            if state_path is not None and state_path.is_file():
                self.models.set_texture_state(state_path, out)
                self.log.emit("2nd stage finished; Generate Texture is "
                              "available.", "ok")
            else:
                self.models.set_texture_state(None, None)
                self.log.emit("2nd stage finished, but its texture state was "
                              "not saved; Generate Texture is unavailable.",
                              "warn")
            if isinstance(scale_back, dict) and scale_back.get("scaled"):
                info = self.jobs.scale_back or {}
                if _stage2_rescale_enabled():
                    self.log.emit(
                        "The 2nd-stage source mesh was normalized by UniForge "
                        f"(x{float(info.get('uniform_scale', 1.0)):.6g} into "
                        "the canonical cube); Uniforge 2nd stage Rescale will "
                        "restore the source's original size/location and "
                        "overwrite the result.", "info")
                else:
                    factor = float(info.get("scale_back_factor", 1.0))
                    offset = (info.get("translation_display")
                              or [0.0, 0.0, 0.0])
                    self.log.emit(
                        "The 2nd-stage source mesh was normalized by UniForge "
                        f"(x{float(info.get('uniform_scale', 1.0)):.6g} into "
                        "the canonical cube); enable Uniforge 2nd stage "
                        "Rescale in Settings to restore its original "
                        f"size/location (x{factor:.6g}, offset "
                        f"[{offset[0]:g}, {offset[1]:g}, {offset[2]:g}]).",
                        "info")
            elif isinstance(scale_back, dict):
                self.log.emit("The 2nd-stage source was already in UniForge's "
                              "canonical bounds; no rescale is needed.",
                              "info")
        same_model = (
            kind == "texture"
            and self.models.texture_source is not None
            and self.viewer.current_path is not None
            and self.viewer.current_path.resolve()
            == self.models.texture_source.resolve()
        )
        if same_model and self.viewer.add_textured(out):
            self.log.emit("Texture added; the first three materials are "
                          "unchanged. Click the 4th swatch to view the "
                          "textured model.", "ok")
        elif self.viewer.load(out):
            if kind == "texture":
                self.edits.discard("another model was loaded")
            else:
                self.models.adopt(out)
            self.log.emit(f"Loaded result into viewer: {out.name}", "ok")
            if (kind == "stage2" and self.jobs.scale_back is not None
                    and _stage2_rescale_enabled()):
                self._auto_stage2_rescale(self.jobs.scale_back)
        else:
            self.log.emit("Could not display result in viewer.", "error")
        self.jobs.convert(out)
        self.changed.emit()

    def on_hi3d_done(self, data: dict):
        out = data["output"]
        self.log.emit(f"Hi3D finished -> {out}", "ok")
        self.app.set_seed(data["seed"])
        self.log.emit("Hi3D recenters the result and normalizes it to unit "
                      "radius; absolute scale is not preserved.", "info")
        if self.viewer.load(out):
            self.edits.discard("another model was loaded")
            self.log.emit(f"Loaded Hi3D result into viewer: {out.name}", "ok")
        else:
            self.log.emit("Could not display the Hi3D result in viewer.",
                          "error")
        self.models.track_output(out)
        self.jobs.convert(out)
        self.changed.emit()

    def on_lato2_done(self, data: dict):
        out = data["output"]
        self.log.emit(f"LATO.2 finished -> {out}", "ok")
        self.app.set_seed(data["seed"])
        self.log.emit("LATO.2 generates new vertices and connectivity and "
                      "normalizes the result to the canonical cube; absolute "
                      "scale is not preserved.", "info")
        if self.viewer.load(out):
            self.edits.discard("another model was loaded")
            self.log.emit(f"Loaded LATO.2 result into viewer: {out.name}",
                          "ok")
        else:
            self.log.emit("Could not display the LATO.2 result in viewer.",
                          "error")
        self.models.track_output(out)
        self.models.set_texture_state(None, None)
        if _shape512_state_path(out).is_file():
            self.log.emit("Run 2nd stage can upscale this LATO.2 mesh.", "ok")
        self.jobs.convert(out)
        self.changed.emit()

    def on_cellocut_done(self, data: dict):
        repaired = data["output"]
        self.log.emit(f"CelloCut finished; saved {repaired.name} "
                      "(the source mesh is untouched).", "ok")
        self.edits.discard("CelloCut finished")
        if self.viewer.load(repaired, preserve_camera=True,
                            preserve_material=True):
            self.log.emit("Showing the watertight mesh (camera kept).", "ok")
        else:
            self.log.emit("Could not display the repaired mesh in viewer.",
                          "error")
        self.models.track_output(repaired)
        self.jobs.convert(repaired)
        self.changed.emit()
