"""Model-folder contents, loaded-model bookkeeping, and model file ops.

A folder lists its ``*.glb`` files; an ``*.obj`` appears too when it has no
same-stem GLB beside it (see ``paths._folder_models``).
"""

from pathlib import Path
import os
import shutil
import subprocess

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QInputDialog, QLineEdit, QMessageBox

from paths import (
    OUTPUT_DIR,
    _folder_models,
    _model_sidecars,
    _paths_equal,
    _shape512_state_path,
    _shape_state_path,
    _validate_model_stem,
)


class ModelLibrary(QObject):
    """Owns the Outliner folder and the path state tied to its models.

    Also tracks the outputs and checkpoints later runs reuse, and keeps all
    of it in sync when a model is renamed or deleted.
    """

    changed = Signal()
    log = Signal(str, str)

    def __init__(self, viewer, jobs, edits, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.jobs = jobs
        self.edits = edits
        self.folder: Path | None = None
        self.last_output: Path | None = None
        self.texture_state: Path | None = None
        self.texture_source: Path | None = None
        self.stage1_state: Path | None = None
        self.stage1_source: Path | None = None
        self.stage1_images: list[Path] | None = None

    # -- tracked paths -------------------------------------------------------
    def set_folder(self, folder):
        self.folder = Path(folder) if folder is not None else None
        self.changed.emit()

    def track_output(self, path: Path):
        """Record a finished output so the Outliner lists its folder."""
        path = Path(path)
        self.last_output = path
        self.folder = path.parent
        self.changed.emit()

    def set_texture_state(self, state, source):
        self.texture_state = state
        self.texture_source = source
        self.changed.emit()

    def set_stage1_state(self, state, source, images):
        self.stage1_state = state
        self.stage1_source = source
        self.stage1_images = list(images) if images else None
        self.changed.emit()

    def clear_stage1(self):
        self.stage1_state = None
        self.stage1_source = None
        self.stage1_images = None
        self.changed.emit()

    def adopt(self, path: Path):
        """Bookkeeping for a model the viewer just displayed."""
        path = Path(path)
        self.edits.discard(f"loaded {path.name}")
        state = _shape_state_path(path)
        self.last_output = path
        self.folder = path.parent
        if state.is_file():
            self.texture_state = state
            self.texture_source = path
        else:
            self.texture_state = None
            self.texture_source = None
        self.changed.emit()

    # -- display -------------------------------------------------------------
    def open_folder(self, folder: Path):
        """List *folder* in the Outliner and load its most useful model."""
        folder = Path(folder)
        models = _folder_models(folder)
        self.folder = folder
        if not models:
            self.log.emit(f"No *.glb files (or unpaired *.obj files) in "
                          f"{folder}.", "warn")
            self.changed.emit()
            return
        names = {model.name for model in models}
        first = next((n for n in (
            "prediction.glb", "prediction_512.glb",
            "prediction.obj", "prediction_512.obj") if n in names),
            models[0].name)
        target = folder / first
        self.log.emit(f"Opened {folder.name} ({len(models)} model file(s)); "
                      f"loading {first} ...", "info")
        if self.viewer.load(target):
            self.adopt(target)
            self.log.emit(f"Loaded {first}. Use the Outliner column to switch "
                          "between the models in this folder.", "ok")
        else:
            self.log.emit(f"Could not load {first}.", "error")
            self.changed.emit()

    def show(self, path: Path):
        """Load one of the folder's model files, keeping camera and material."""
        path = Path(path)
        if not path.is_file():
            self.log.emit(f"Model missing: {path}", "warn")
            self.changed.emit()
            return
        if _paths_equal(self.viewer.current_path, path):
            return                      # already displayed; keep pending edit
        if self.viewer.load(path, preserve_camera=True,
                            preserve_material=True):
            self.log.emit(f"Showing {path.name}.", "info")
            self.adopt(path)
        else:
            self.log.emit(f"Could not show {path.name}.", "error")
            self.changed.emit()

    def ghost(self, path: Path):
        """Middle-click: toggle a gray transparent ghost overlay.

        Ghosting another mesh replaces the current ghost; ghosting the
        mesh that is already the ghost removes it.  The ghost also selects
        the Reproject donor: with a ghost picked, Reproject copies that
        mesh's detail onto the displayed mesh."""
        path = Path(path)
        if not path.is_file():
            self.log.emit(f"Ghost: {path.name} no longer exists.", "warn")
            self.changed.emit()
            return
        if _paths_equal(self.viewer.current_path, path):
            self.log.emit(f"Ghost: {path.name} is the displayed mesh; "
                          "nothing to overlay.", "info")
            return
        try:
            shown = self.viewer.toggle_ghost(path)
        except Exception as ex:  # noqa: BLE001 - keep the UI alive
            self.log.emit(f"Ghost overlay failed: {ex}", "error")
            return
        if shown:
            self.log.emit(
                f"Ghost overlay: {path.name} shown as a gray transparent "
                "reference next to the loaded mesh (Reproject uses it as the "
                "donor; Ctrl+1 hides/shows it).", "info")
        else:
            self.log.emit("Ghost overlay removed; Reproject has no donor "
                          "mesh and is disabled until another mesh is "
                          "ghosted.", "info")
        self.changed.emit()

    # -- file operations -----------------------------------------------------
    @staticmethod
    def obj_path(path: Path) -> Path | None:
        """Return the .obj companion next to *path*, if one exists."""
        path = Path(path)
        candidate = path.with_suffix(".obj")
        if candidate.is_file():
            return candidate
        try:
            siblings = path.parent.iterdir()
        except OSError:
            return None
        for sibling in siblings:
            if (sibling.suffix.lower() == ".obj"
                    and sibling.stem.lower() == path.stem.lower()):
                return sibling
        return None

    def show_obj(self, path: Path):
        """Reveal the model's .obj companion in Windows Explorer."""
        obj = self.obj_path(path)
        if obj is None:
            self.log.emit(
                f"Show obj: no matching .obj next to {Path(path).name}.",
                "warn")
            return
        if os.name != "nt":
            self.log.emit("Show obj is only available on Windows.", "warn")
            return
        try:
            subprocess.Popen(
                f'explorer /select,"{os.path.normpath(str(obj))}"')
        except OSError as ex:
            self.log.emit(f"Could not open Explorer: {ex}", "error")

    def duplicate(self, path: Path):
        """Duplicate one model file as <stem>_copy (without its sidecars)."""
        path = Path(path)
        if self.jobs.busy:
            QMessageBox.information(
                self.parent(), "Duplicate object",
                "Wait for the current background job to finish first.")
            return
        if not path.is_file():
            self.log.emit(f"Duplicate: {path.name} no longer exists.", "warn")
            self.changed.emit()
            return
        copy_path = path.with_name(path.stem + "_copy" + path.suffix)
        n = 2
        while copy_path.exists():
            copy_path = path.with_name(
                f"{path.stem}_copy_{n}{path.suffix}")
            n += 1
        try:
            shutil.copy2(path, copy_path)
        except OSError as ex:
            self.log.emit(f"Duplicate failed: {ex}", "error")
            QMessageBox.warning(self.parent(), "Duplicate failed", str(ex))
            return
        self.log.emit(f"Duplicated {path.name} -> {copy_path.name}.", "ok")
        self.changed.emit()

    def rename(self, path: Path):
        """Rename one model file and its state sidecars from the Outliner."""
        path = Path(path)
        if self.jobs.busy:
            QMessageBox.information(
                self.parent(), "Rename object",
                "Wait for the current background job to finish first.")
            return
        if not path.is_file():
            self.log.emit(f"Rename: {path.name} no longer exists.", "warn")
            self.changed.emit()
            return
        parent = self.parent()
        new_stem, accepted = QInputDialog.getText(
            parent, "Rename object", f"New name for {path.name}:",
            QLineEdit.Normal, path.stem)
        if not accepted:
            return
        new_stem = new_stem.strip()
        # Keep the entry's own extension: an unpaired OBJ stays an OBJ.
        for extension in (".glb", ".obj"):
            if new_stem.lower().endswith(extension):
                new_stem = new_stem[:-len(extension)].rstrip()
                break
        if new_stem == path.stem:
            return
        problem = _validate_model_stem(new_stem)
        if problem:
            QMessageBox.warning(parent, "Rename object", problem)
            return
        new_path = path.with_name(new_stem + path.suffix)
        case_only = path.name.lower() == new_path.name.lower()
        if new_path.exists() and not case_only:
            QMessageBox.warning(
                parent, "Rename object",
                f"{new_path.name} already exists in this folder.")
            return

        old_sidecars = _model_sidecars(path)
        new_sidecars_by_kind = {
            "state": _shape_state_path(new_path),
            "shape512": _shape512_state_path(new_path),
        }
        moved: list[tuple[Path, Path]] = []
        for old_sidecar in old_sidecars:
            kind = ("state" if old_sidecar.name.endswith(".shape_state.pt")
                    else "shape512")
            moved.append((old_sidecar, new_sidecars_by_kind[kind]))
        for _, new in moved:
            if new.exists() and not case_only:
                QMessageBox.warning(
                    parent, "Rename object",
                    f"Cannot rename: {new.name} already exists.")
                return

        renamed: list[tuple[Path, Path]] = []
        try:
            for old, new in moved:
                old.rename(new)
                renamed.append((old, new))
            path.rename(new_path)
        except OSError as ex:
            for old, new in reversed(renamed):
                try:
                    new.rename(old)
                except OSError:
                    pass
            self.log.emit(f"Rename failed: {ex}", "error")
            QMessageBox.warning(parent, "Rename failed", str(ex))
            return

        self._remap(path, new_path, moved)
        sidecar_note = (f" (and {len(moved)} state file"
                        f"{'s' if len(moved) != 1 else ''})"
                        if moved else "")
        self.log.emit(f"Renamed {path.name} -> {new_path.name}{sidecar_note}.",
                      "ok")
        self.changed.emit()

    def _remap(self, old: Path, new: Path, moved_sidecars):
        """Retarget viewer/edit/pending/checkpoint state after a rename."""
        self.viewer.retarget_path(old, new)
        self.edits.retarget(old, new)
        for attr in ("last_output", "texture_source", "stage1_source"):
            if _paths_equal(getattr(self, attr), old):
                setattr(self, attr, new)
        sidecar_map = dict(moved_sidecars)
        for attr in ("texture_state", "stage1_state"):
            value = getattr(self, attr)
            for old_state, new_state in sidecar_map.items():
                if _paths_equal(value, old_state):
                    setattr(self, attr, new_state)
                    break
        if self.jobs.scale_back is not None:
            try:
                target = Path(self.jobs.scale_back.get("output"))
            except (TypeError, ValueError):
                target = None
            if target is not None and _paths_equal(target, old):
                self.jobs.scale_back["output"] = str(new)

    def delete(self, path: Path):
        """Delete one model file and its state sidecars from the Outliner."""
        path = Path(path)
        if self.jobs.busy:
            QMessageBox.information(
                self.parent(), "Delete object",
                "Wait for the current background job to finish first.")
            return
        if not path.is_file():
            self.log.emit(f"Delete: {path.name} no longer exists.", "warn")
            self.changed.emit()
            return
        parent = self.parent()
        current = _paths_equal(self.viewer.current_path, path)
        pending = False
        if self.edits.pending is not None:
            try:
                pending_target = Path(self.edits.pending["target_path"])
            except (KeyError, TypeError):
                pending_target = None
            pending = (pending_target is not None
                       and _paths_equal(pending_target, path))
        sidecars = _model_sidecars(path)

        message = f"Delete {path.name}?\n\nThis cannot be undone."
        if sidecars:
            names = ", ".join(sidecar.name for sidecar in sidecars)
            message += f"\n\nAlso deletes its state file(s):\n{names}"
        if current:
            message += "\n\nThe model is currently displayed and will close."
        if pending:
            message += "\n\nIts unsaved edit will be discarded."
        answer = QMessageBox.question(
            parent, "Delete object", message,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return

        try:
            path.unlink()
        except OSError as ex:
            self.log.emit(f"Delete failed: {ex}", "error")
            QMessageBox.warning(parent, "Delete failed", str(ex))
            return
        failed_sidecars = []
        for sidecar in sidecars:
            try:
                sidecar.unlink()
            except OSError as ex:
                failed_sidecars.append(f"{sidecar.name}: {ex}")

        if self.jobs.scale_back is not None:
            try:
                scale_target = Path(self.jobs.scale_back.get("output"))
            except (TypeError, ValueError):
                scale_target = None
            if scale_target is not None and _paths_equal(scale_target, path):
                self.jobs.scale_back = None
        if current:
            self.viewer.clear()
            self.last_output = None
            self.texture_state = None
            self.texture_source = None
            self.stage1_state = None
            self.stage1_source = None
        if pending:
            self.edits.discard(f"deleted {path.name}")
        deleted_states = len(sidecars) - len(failed_sidecars)
        state_note = (f" and {deleted_states} state file"
                      f"{'s' if deleted_states != 1 else ''}"
                      if deleted_states else "")
        self.log.emit(f"Deleted {path.name}{state_note}.", "ok")
        for problem in failed_sidecars:
            self.log.emit(f"Could not delete state file {problem}", "warn")
        self.changed.emit()

    def open_output_folder(self):
        """Open the folder holding the last generated model."""
        target = (self.last_output.parent if self.last_output
                  else OUTPUT_DIR)
        if target.is_dir():
            os.startfile(str(target))
