"""Shared application and action state."""

import json
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from paths import IMAGE_SUFFIXES


VIEW_NAMES = ("front", "left", "right", "top", "bottom", "back")
VIEWS_FILENAME = "reference_views.json"


class AppState(QObject):
    """Imported images, their cutouts, the REF selection, and last seed."""

    changed = Signal()          # images / cutouts / reference changed
    seed_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_paths: list[Path] = []
        self.cutout_map: dict[Path, Path] = {}
        self.ref_reference: Path | None = None
        #: path -> one of VIEW_NAMES ("front", "left", ...) or "clear" absent
        self.view_map: dict[Path, str] = {}
        #: folder the images were loaded from (persist view tags there)
        self.project_dir: Path | None = None
        self.last_seed: int | None = None

    @property
    def reference_views(self) -> list[tuple[Path, str]]:
        """[(path, view), ...] for images with an assigned view."""
        return [(p, v) for p, v in self.view_map.items()
                if p in self.image_paths]

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
                and self.ref_reference is None and not self.view_map):
            return
        self.image_paths.clear()
        self.cutout_map.clear()
        self.view_map.clear()
        self.project_dir = None
        self.ref_reference = None
        self.changed.emit()

    def remove_image(self, path) -> bool:
        path = Path(path)
        removed = path in self.image_paths
        if removed:
            self.image_paths.remove(path)
        self.cutout_map.pop(path, None)
        self.view_map.pop(path, None)
        if path == self.ref_reference:
            self.ref_reference = None
            removed = True
        if removed:
            self._save_current_views()
            self.changed.emit()
        return removed

    def set_view(self, path, view: str | None) -> bool:
        """Assign a reference view to an image; None clears it."""
        path = Path(path)
        if path not in self.image_paths:
            return False
        if view:
            view = view.lower()
            if view not in VIEW_NAMES or self.view_map.get(path) == view:
                return False
            self.view_map[path] = view
        elif path not in self.view_map:
            return False
        else:
            del self.view_map[path]
        self._save_current_views()
        self.changed.emit()
        return True

    def _save_current_views(self):
        """Persist view tags to the project folder immediately (if any)."""
        if self.project_dir is None:
            return
        mapping = {p.name: v for p, v in self.view_map.items()
                   if p.parent == self.project_dir}
        self.save_views(self.project_dir, mapping)

    def save_views(self, inputs_dir: Path, mapping: dict[str, str]) -> bool:
        """Write ``{filename: view}`` into *inputs_dir* as JSON.

        *mapping* keys are image file names relative to the folder. Returns
        True when the file was written. An empty mapping removes a stale
        file instead."""
        inputs_dir = Path(inputs_dir)
        if not mapping:
            try:
                (inputs_dir / VIEWS_FILENAME).unlink(missing_ok=True)
            except OSError:
                pass
            return False
        try:
            inputs_dir = Path(inputs_dir)
            inputs_dir.mkdir(parents=True, exist_ok=True)
            (inputs_dir / VIEWS_FILENAME).write_text(
                json.dumps(mapping, indent=2), encoding="utf-8")
            return True
        except (OSError, TypeError):
            return False

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
        self.view_map.clear()
        self.project_dir = search
        self.ref_reference = None
        restored_views = self._load_views(search, files)
        restored = 0
        for src in files:
            # _copy_input_images stores sources as <stem>_src.png next to
            # <stem>_cutout.png, so strip the _src suffix before probing for
            # the cutout (a drag-dropped <stem>.png maps to <stem>_cutout.png).
            stem = src.stem[:-4] if src.stem.endswith("_src") else src.stem
            cutout = src.with_name(stem + "_cutout.png")
            if cutout.is_file():
                self.cutout_map[src] = cutout
                restored += 1
        self.changed.emit()
        return len(files), restored, inputs_dir.is_dir()

    def _load_views(self, search: Path, files: list[Path]) -> int:
        """Restore view tags from ``reference_views.json`` next to the images.

        Returns the number of restored assignments."""
        views_file = search / VIEWS_FILENAME
        if not views_file.is_file():
            return 0
        try:
            saved = json.loads(views_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        if not isinstance(saved, dict):
            return 0
        present = set(files)
        for name, view in saved.items():
            if not isinstance(view, str) or view not in VIEW_NAMES:
                continue
            path = search / name
            if path in present:
                self.view_map[path] = view
        return len(self.view_map)


@dataclass(frozen=True)
class ActionState:
    """Snapshot of what the UI is allowed to do right now."""

    busy: bool = False
    cutting: bool = False
    refining: bool = False
    lato2: bool = False
    repairing: bool = False
    utex: bool = False
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
                    or self.lato2 or self.repairing or self.utex
                    or self.edit_running)

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
        if self.utex:
            return {"utexws": "Running: WS normal bake"}.get(
                self.pending_kind, "Running: UltraTex texturing")
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

