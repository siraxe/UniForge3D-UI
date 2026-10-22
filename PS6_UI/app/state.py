"""Shared application and action state."""

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from paths import IMAGE_SUFFIXES


class AppState(QObject):
    """Imported images, their cutouts, the REF selection, and last seed."""

    changed = Signal()          # images / cutouts / reference changed
    seed_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_paths: list[Path] = []
        self.cutout_map: dict[Path, Path] = {}
        self.ref_reference: Path | None = None
        self.last_seed: int | None = None

    def add_images(self, paths) -> bool:
        added = False
        for raw in paths:
            path = Path(raw)
            if path not in self.image_paths:
                self.image_paths.append(path)
                added = True
        if added:
            self.changed.emit()
        return added

    def clear(self):
        if (not self.image_paths and not self.cutout_map
                and self.ref_reference is None):
            return
        self.image_paths.clear()
        self.cutout_map.clear()
        self.ref_reference = None
        self.changed.emit()

    def remove_image(self, path) -> bool:
        path = Path(path)
        removed = path in self.image_paths
        if removed:
            self.image_paths.remove(path)
        self.cutout_map.pop(path, None)
        if path == self.ref_reference:
            self.ref_reference = None
            removed = True
        if removed:
            self.changed.emit()
        return removed

    def set_reference(self, path) -> bool:
        path = Path(path)
        if path not in self.image_paths or path == self.ref_reference:
            return False
        self.ref_reference = path
        self.changed.emit()
        return True

    def set_cutout(self, source, processed):
        self.cutout_map[Path(source)] = Path(processed)
        self.changed.emit()

    def set_seed(self, seed):
        try:
            value = int(seed)
        except (TypeError, ValueError):
            return
        self.last_seed = value
        self.seed_changed.emit()

    def load_folder(self, folder: Path):
        """Replace the image list with the images stored in *folder*.

        Returns ``(count, restored_cutouts, used_inputs_dir)`` or None."""
        folder = Path(folder)
        inputs_dir = folder / "inputs"
        search = inputs_dir if inputs_dir.is_dir() else folder
        try:
            files = sorted(
                p for p in search.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                and not p.stem.endswith("_cutout"))
        except OSError:
            return None
        if not files:
            return None
        self.image_paths = list(files)
        self.cutout_map.clear()
        self.ref_reference = None
        restored = 0
        for src in files:
            cutout = src.with_name(src.stem + "_cutout.png")
            if cutout.is_file():
                self.cutout_map[src] = cutout
                restored += 1
        self.changed.emit()
        return len(files), restored, inputs_dir.is_dir()


@dataclass(frozen=True)
class ActionState:
    """Snapshot of what the UI is allowed to do right now."""

    busy: bool = False
    cutting: bool = False
    refining: bool = False
    lato2: bool = False
    repairing: bool = False
    reprojecting: bool = False
    polishing: bool = False
    remeshing: bool = False
    decimating: bool = False
    subdividing: bool = False
    closing_holes: bool = False
    deleting_hidden: bool = False
    mirroring: bool = False
    has_images: bool = False
    has_model: bool = False
    has_texture_state: bool = False
    pending_edit: bool = False
    donor_ok: bool = False
    has_ghost: bool = False
    ghost_visible: bool = False
    pending_kind: str | None = None

    @property
    def idle(self) -> bool:
        return not (self.busy or self.cutting or self.refining
                    or self.lato2 or self.repairing or self.edit_running)

    @property
    def edit_running(self) -> bool:
        return any((self.reprojecting, self.polishing, self.remeshing,
                    self.decimating, self.subdividing, self.closing_holes,
                    self.deleting_hidden, self.mirroring))

    @property
    def activity(self) -> str:
        if self.cutting:
            return "Running: background removal"
        if self.refining:
            return "Running: Hi3D 1536 refinement"
        if self.lato2:
            return "Running: LATO.2 mesh generation"
        if self.repairing:
            return "Running: CelloCut watertight repair"
        if self.reprojecting:
            return "Running: detail reprojection"
        if self.mirroring:
            return "Running: mirror and merge"
        if self.polishing:
            return "Running: Polish"
        if self.remeshing:
            return "Running: Remesh"
        if self.busy:
            return {
                "texture": "Running: texture generation",
                "stage2": "Running: 2nd stage (1024 upscale)",
            }.get(self.pending_kind, "Running: UniForge3D generation")
        return "Idle"

