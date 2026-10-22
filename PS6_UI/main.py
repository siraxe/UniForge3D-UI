"""UniForge3D - PySide6 + VTK desktop UI for image-to-3D generation.

Native Windows app (no browser): PySide6 controls on the left, an embedded
VTK 3D viewer (GLB/OBJ/STL/PLY with PBR materials) on the right.

Feature parity with the Flet UI:
  - imported-image thumbnails with per-image remove buttons
  - pre-generation background removal (RMBG-2.0 in a light worker process),
    with the resulting RGBA cutouts shown as thumbnails of their own
  - textured / geometry-only mode, grid, texture size, steps, seed

Generation runs in a separate worker process (worker.py in this directory) so
Stop can hard-kill a run and this process stays lightweight (no torch import).
Background removal uses rmbg_worker.py, which only loads RMBG-2.0 (~never the
26 GB pipeline).

Run with:  .venv/Scripts/python.exe PS6_UI/main.py   (or start_ps6.bat)
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import (
    QEvent, QObject, QPoint, QSize, Qt, QThread, QTimer, Signal,
)
from PySide6.QtGui import QAction, QColor, QPalette, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QSpinBox, QSplitter, QToolButton, QVBoxLayout, QWidget,
)

import viewer as viewer_mod
from model_paths import find_model_dir

ROOT = Path(__file__).resolve().parent.parent
WORKER = Path(__file__).resolve().parent / "worker.py"
RMBG_WORKER = Path(__file__).resolve().parent / "rmbg_worker.py"
MODEL_DIR = find_model_dir()
OUTPUT_DIR = ROOT / "output_dir"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

THUMB_SIZE = 48


def _shape_state_path(glb: Path) -> Path:
    """Geometry-only GLB -> the Shape-1024 state saved next to it."""
    return glb.with_name(glb.stem + ".shape_state.pt")


def _textured_output_path(glb: Path) -> Path:
    """Geometry-only GLB -> the textured GLB generated from it."""
    return glb.with_name(glb.stem + "_textured.glb")


# ---------------------------------------------------------------------------
# Thumbnail widget with an overlaid remove button (mirrors the Flet thumbs)
# ---------------------------------------------------------------------------
class _ImagePreview(QWidget):
    """Large frameless popup shown while hovering the selected REF thumb.

    It is a transparent-for-mouse tooltip, so it never steals the hover or
    activates the window; it simply sits next to the thumbnail.
    """

    MAX_SIZE = 600

    def __init__(self, path: Path, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_StyledBackground, True)   # paint stylesheet bg
        self.setObjectName("imagePreview")
        self.setStyleSheet(
            "QWidget#imagePreview { background: #16181a;"
            " border: 1px solid #6FA8FF; border-radius: 6px; }")

        label = QLabel(self)
        label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            label.setText("?")
        else:
            label.setPixmap(pixmap.scaled(
                self.MAX_SIZE, self.MAX_SIZE,
                Qt.KeepAspectRatio, Qt.SmoothTransformation))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(label)
        self.adjustSize()

    def show_near(self, anchor: QWidget) -> None:
        """Place the preview beside the anchor, clamped to the screen."""
        self.adjustSize()
        screen = anchor.screen()
        if screen is None:
            self.show()
            return
        geo = screen.availableGeometry()
        top_left = anchor.mapToGlobal(QPoint(0, 0))
        x = top_left.x() + anchor.width() + 10
        if x + self.width() > geo.right() + 1:
            x = top_left.x() - self.width() - 10
        y = top_left.y() + (anchor.height() - self.height()) // 2
        x = max(geo.left() + 4, min(x, geo.right() - self.width() - 3))
        y = max(geo.top() + 4, min(y, geo.bottom() - self.height() - 3))
        self.move(x, y)
        self.show()


class _Thumb(QWidget):
    """One image thumbnail; x removes it, clicking selects it as reference."""

    remove = Signal(object)                      # carries the Path
    selected = Signal(object)                    # carries the Path

    def __init__(self, path: Path, size: int = THUMB_SIZE,
                 border: str | None = None, removable: bool = True,
                 selectable: bool = False, parent=None):
        super().__init__(parent)
        self.path = path
        self.setFixedSize(size, size)
        self._selectable = selectable
        self._border = border
        self._selected = False
        self._preview: _ImagePreview | None = None
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(250)   # brief dwell avoids popup flashes
        self._hover_timer.timeout.connect(self._show_preview)

        pixmap = QPixmap(str(path))
        self._label = QLabel(self)
        self._label.setFixedSize(size, size)
        self._label.setAlignment(Qt.AlignCenter)
        # Let clicks fall through to this widget so the whole tile selects.
        self._label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        if not pixmap.isNull():
            self._label.setPixmap(pixmap.scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:                                     # unreadable image
            self._label.setText("?")

        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._label, 0, 0)

        self._badge = QLabel("REF", self)
        self._badge.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._badge.setStyleSheet(
            "QLabel { background: #4A6FA5; color: white; font-size: 8px;"
            "  font-weight: bold; padding: 1px 3px; border-radius: 3px; }")
        self._badge.adjustSize()
        lay.addWidget(self._badge, 0, 0, Qt.AlignBottom | Qt.AlignHCenter)
        self._badge.hide()
        self._label.installEventFilter(self)
        self._badge.installEventFilter(self)

        if removable:
            btn = QToolButton(self)
            btn.setText("x")
            btn.setFixedSize(16, 16)
            btn.setStyleSheet(
                "QToolButton { background: rgba(0,0,0,60%); color: #e0e0e0;"
                "  border: none; font-size: 9px; }"
                "QToolButton:hover { background: rgba(180,50,50,90%); }")
            btn.clicked.connect(lambda: self.remove.emit(self.path))
            lay.addWidget(btn, 0, 0, Qt.AlignTop | Qt.AlignRight)
        self._base_tip = ("Click to use this image as the REF selection. "
                          "Hover the selected REF image for a larger "
                          "preview.")
        if selectable:
            self.setCursor(Qt.PointingHandCursor)
            self.setToolTip(self._base_tip)
        self._apply_style()

    def _apply_style(self):
        border = "#6FA8FF" if self._selected else self._border
        style = ""
        if border:
            width = 3 if self._selected else 2
            style = f"border: {width}px solid {border}; border-radius: 4px;"
        self._label.setStyleSheet(f"background: #2a2d30; {style}")
        self._badge.setVisible(self._selected)

    def set_selected(self, selected: bool):
        self._selected = bool(selected)
        self._apply_style()
        # While selected, suppress the native tooltip so it cannot pop up on
        # top of the large preview; the hover preview takes over.
        self.setToolTip("" if self._selected else self._base_tip)
        if not self._selected:
            self._hide_preview()
        elif self.underMouse():
            # Selected via click without moving the cursor: behave as if the
            # cursor had just entered the tile.
            self._hover_timer.start()

    # -- hover preview for the selected REF image -------------------------
    def _show_preview(self):
        if not (self._selectable and self._selected and self.isVisible()):
            return
        if self._preview is None:
            self._preview = _ImagePreview(self.path, self)
        self._preview.show_near(self)

    def _hide_preview(self):
        if self._preview is not None and self._preview.isVisible():
            self._preview.hide()

    def enterEvent(self, event):
        super().enterEvent(event)
        if self._selectable and self._selected:
            self._hover_timer.start()

    def leaveEvent(self, event):
        self._hover_timer.stop()
        self._hide_preview()
        super().leaveEvent(event)

    def hideEvent(self, event):
        self._hover_timer.stop()
        self._hide_preview()
        super().hideEvent(event)

    def wheelEvent(self, event):
        # Scrolling moves the tile out from under the preview; drop it and
        # let the list scroll normally.
        self._hover_timer.stop()
        self._hide_preview()
        event.ignore()

    def eventFilter(self, obj, event):
        # The image label covers this widget, so handle its clicks too.
        if (self._selectable and obj in (self._label, self._badge)
                and event.type() == QEvent.MouseButtonPress
                and event.button() == Qt.LeftButton):
            self.selected.emit(self.path)
            event.accept()
            return True
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        if self._selectable and event.button() == Qt.LeftButton:
            self.selected.emit(self.path)
            event.accept()
            return
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Generation worker plumbing (JSON on stdio, one object per line)
# ---------------------------------------------------------------------------
class _WorkerReader(QThread):
    log_line = Signal(str, str)          # msg, level
    job_done = Signal(str, object)       # output path, seed used
    job_failed = Signal(str)             # error message
    worker_dead = Signal(int)            # exit code

    def __init__(self, proc: subprocess.Popen, parent=None):
        super().__init__(parent)
        self._proc = proc

    def run(self):
        proc = self._proc
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    self.log_line.emit(line, "info")
                    continue
                kind = msg.get("type")
                if kind == "log":
                    self.log_line.emit(str(msg.get("msg", "")),
                                       str(msg.get("level", "info")))
                elif kind == "done":
                    self.job_done.emit(str(msg.get("output", "")),
                                       msg.get("seed"))
                elif kind == "error":
                    self.job_failed.emit(str(msg.get("msg", "")))
                elif kind == "ready":
                    self.log_line.emit("Worker ready.", "ok")
        finally:
            code = proc.wait()
            if code not in (0, None):
                self.worker_dead.emit(code)


class WorkerController(QObject):
    """Owns the generation worker subprocess."""

    log_line = Signal(str, str)
    state_changed = Signal(bool)          # True while generating
    generation_done = Signal(str, object)  # output path, seed used
    generation_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: subprocess.Popen | None = None
        self._reader: _WorkerReader | None = None
        self._generating = False

    @property
    def generating(self) -> bool:
        return self._generating

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _spawn(self):
        if self._proc and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [sys.executable, str(WORKER),
             "--model-dir", str(MODEL_DIR), "--low-vram"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, cwd=str(ROOT),
        )
        self._reader = _WorkerReader(self._proc, self)
        self._reader.log_line.connect(self.log_line)
        self._reader.job_done.connect(self._on_done)
        self._reader.job_failed.connect(self._on_failed)
        self._reader.worker_dead.connect(self._on_dead)
        self._reader.start()

    def _send(self, cmd: dict):
        self._spawn()
        self._proc.stdin.write(json.dumps(cmd) + "\n")
        self._proc.stdin.flush()
        self._generating = True
        self.state_changed.emit(True)

    def start_run(self, images, output: Path, *, seed, grid_size: int,
                  texture_size: int, steps: int, texture: bool,
                  shape_state: Path | None = None):
        cmd = {
            "cmd": "generate",
            "images": [str(p) for p in images],
            "output": str(output),
            "seed": seed,
            "overwrite": True,
            "grid_size": grid_size,
            "texture_size": texture_size,
            "steps": steps,
            "texture": texture,
        }
        if shape_state is not None:
            cmd["shape_state"] = str(shape_state)
        self._send(cmd)

    def start_texture(self, images, output: Path, state: Path, *,
                      seed, texture_size: int, steps: int):
        self._send({
            "cmd": "texture",
            "images": [str(p) for p in images],
            "output": str(output),
            "state": str(state),
            "seed": seed,
            "overwrite": True,
            "texture_size": texture_size,
            "steps": steps,
        })

    def stop(self):
        """Hard-kill the worker; a stopped run leaves no partial output."""
        if not self._proc:
            return
        proc, self._proc = self._proc, None
        if self._reader:
            self._reader.job_done.disconnect(self._on_done)
            self._reader.job_failed.disconnect(self._on_failed)
            self._reader.worker_dead.disconnect(self._on_dead)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._reap_reader()
        self._generating = False
        self.state_changed.emit(False)

    def _reap_reader(self):
        """Let the stdout reader finish so no QThread outlives its parent."""
        if self._reader is None:
            return
        if not self._reader.wait(3000):
            self._reader.terminate()
            self._reader.wait(1000)
        self._reader = None

    def _on_done(self, output: str, seed=None):
        self._generating = False
        self.state_changed.emit(False)
        self.generation_done.emit(output, seed)

    def _on_failed(self, msg: str):
        self._generating = False
        self.state_changed.emit(False)
        self.generation_failed.emit(msg)

    def _on_dead(self, code: int):
        self._proc = None
        if self._generating:
            self._generating = False
            self.state_changed.emit(False)
            self.generation_failed.emit(
                f"Worker exited unexpectedly (code {code}).")
        self.log_line.emit(f"Worker process exited (code {code}).", "warn")

    def shutdown(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._reap_reader()


# ---------------------------------------------------------------------------
# Background-removal (RMBG-2.0) worker plumbing
# ---------------------------------------------------------------------------
class _RmbgReader(QThread):
    log_line = Signal(str)
    cutout_ready = Signal(str, str)      # source path, processed path
    finished = Signal(int)               # process return code

    def __init__(self, proc: subprocess.Popen, parent=None):
        super().__init__(parent)
        self._proc = proc

    def run(self):
        proc = self._proc
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = msg.get("type")
            if kind == "image":
                self.cutout_ready.emit(str(msg.get("source", "")),
                                       str(msg.get("processed", "")))
            elif kind in ("log", "error"):
                self.log_line.emit(str(msg.get("msg", "")))
        code = proc.wait()
        self.finished.emit(code if code is not None else 0)


class RmbgController(QObject):
    """Runs rmbg_worker.py: loads RMBG-2.0 only, exits when done."""

    log_line = Signal(str)
    cutout_ready = Signal(str, str)
    state_changed = Signal(bool)         # True while running
    finished = Signal(int)

    def __init__(self, out_dir: Path, parent=None):
        super().__init__(parent)
        self._out_dir = Path(out_dir)
        self._proc: subprocess.Popen | None = None
        self._reader: _RmbgReader | None = None

    @property
    def running(self) -> bool:
        return self._reader is not None and self._reader.isRunning()

    def start(self, images):
        if self.running:
            return
        self._out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(RMBG_WORKER),
               "--out-dir", str(self._out_dir)]
        cmd += ["--images"] + [str(p) for p in images]
        self._proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        self._reader = _RmbgReader(self._proc, self)
        self._reader.log_line.connect(self.log_line)
        self._reader.cutout_ready.connect(self.cutout_ready)
        self._reader.finished.connect(self._on_finished)
        self._reader.start()
        self.state_changed.emit(True)

    def _on_finished(self, code: int):
        self._reader = None
        self._proc = None
        self.state_changed.emit(False)
        self.finished.emit(code)

    def shutdown(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        if self._reader is not None:
            if not self._reader.wait(3000):
                self._reader.terminate()
                self._reader.wait(1000)
            self._reader = None


# ---------------------------------------------------------------------------
# GLB -> OBJ conversion (runs in a plain thread; trimesh imported lazily)
# ---------------------------------------------------------------------------
class ObjConverter(QObject):
    """Convert a finished GLB to OBJ beside it, like the Flet UI's button."""

    log_line = Signal(str, str)          # msg, level

    def __init__(self, parent=None):
        super().__init__(parent)
        self._busy = False

    def convert(self, glb: Path):
        if self._busy:
            return
        self._busy = True
        threading.Thread(target=self._job, args=(Path(glb),),
                         daemon=True).start()

    def _job(self, glb: Path):
        """Write <stem>.obj; trimesh also emits <stem>.mtl + texture PNGs
        for textured models. trimesh is imported here so app startup stays
        light.
        """
        try:
            import trimesh

            stem = glb.stem
            self.log_line.emit(f"Converting {glb.name} -> {stem}.obj ...",
                               "info")
            before = {p.name for p in glb.parent.iterdir()}
            loaded = trimesh.load(str(glb))
            geometries = (loaded.geometry.values()
                          if hasattr(loaded, "geometry") else [loaded])
            for geom in geometries:
                material = getattr(getattr(geom, "visual", None),
                                   "material", None)
                if material is not None and hasattr(material, "name"):
                    # trimesh derives the exported texture file name from the
                    # material name; match it to the OBJ stem.
                    material.name = stem
            loaded.export(str(glb.with_suffix(".obj")),
                          include_texture=True, mtl_name=f"{stem}.mtl")
            sidecars = sorted(p.name for p in glb.parent.iterdir()
                              if p.name not in before
                              and p.suffix.lower() != ".obj")
            self.log_line.emit(
                f"Saved {stem}.obj"
                + (f" (+ {', '.join(sidecars)})" if sidecars
                   else " (mesh only, no texture)"), "ok")
        except Exception as ex:
            self.log_line.emit(f"OBJ conversion failed: {ex}", "error")
        finally:
            self._busy = False


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
_DARK = QPalette()
_DARK.setColor(QPalette.Window, QColor(0x1E, 0x20, 0x22))
_DARK.setColor(QPalette.WindowText, QColor(0xE0, 0xE0, 0xE0))
_DARK.setColor(QPalette.Base, QColor(0x16, 0x18, 0x1A))
_DARK.setColor(QPalette.Text, QColor(0xE0, 0xE0, 0xE0))
_DARK.setColor(QPalette.Button, QColor(0x2A, 0x2D, 0x30))
_DARK.setColor(QPalette.ButtonText, QColor(0xE0, 0xE0, 0xE0))
_DARK.setColor(QPalette.Highlight, QColor(0x4A, 0x6F, 0xA5))
_DARK.setColor(QPalette.HighlightedText, QColor(0xFF, 0xFF, 0xFF))
_DARK.setColor(QPalette.PlaceholderText, QColor(0x80, 0x84, 0x88))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UniForge3D - image to 3D")
        self.resize(1500, 900)
        self.setPalette(_DARK)

        # original path -> RGBA cutout path (skip RMBG in the pipeline)
        self.image_paths: list[Path] = []
        self.cutout_map: dict[Path, Path] = {}
        self.last_output: Path | None = None   # GLB of the last successful run
        self.last_seed: int | None = None      # seed of the last completed run
        # Geometry-only runs save a Shape-1024 state next to the GLB. Generate
        # Texture reuses it, so the mesh is not regenerated.
        self.texture_state: Path | None = None
        self.texture_source: Path | None = None
        self._pending_kind: str | None = None
        self._pending_state: Path | None = None
        # Input image the user clicked as the REF selection. There is
        # deliberately no default: the user must pick one.
        self.ref_reference: Path | None = None

        self.converter = ObjConverter(self)
        self.converter.log_line.connect(self._on_worker_log)

        self.worker = WorkerController(self)
        self.worker.log_line.connect(self._on_worker_log)
        self.worker.state_changed.connect(self._on_state_changed)
        self.worker.generation_done.connect(self._on_done)
        self.worker.generation_failed.connect(self._on_failed)

        self.rmbg = RmbgController(OUTPUT_DIR / ".cutouts", self)
        self.rmbg.log_line.connect(lambda m: self.log(m, "info"))
        self.rmbg.cutout_ready.connect(self._on_cutout_ready)
        self.rmbg.state_changed.connect(lambda b: self._refresh_buttons())
        self.rmbg.finished.connect(self._on_rmbg_finished)

        self._build_menu()
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_viewer())
        splitter.setSizes([380, 1120])
        self._refresh_buttons()

    # -- UI construction ------------------------------------------------
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        act_open = QAction("Open &model...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.on_open_model)
        m.addAction(act_open)
        m.addSeparator()
        act_quit = QAction("E&xit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        m.addAction(act_quit)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        # Input images with thumbnails ------------------------------------
        grp_input = QGroupBox("Input image(s)")
        v = QVBoxLayout(grp_input)
        self.thumb_list = QListWidget()
        self.thumb_list.setViewMode(QListWidget.IconMode)
        self.thumb_list.setResizeMode(QListWidget.Adjust)
        self.thumb_list.setMovement(QListWidget.Static)
        self.thumb_list.setSpacing(6)
        self.thumb_list.setUniformItemSizes(True)
        self.thumb_list.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.thumb_list.setMaximumHeight(THUMB_SIZE * 2 + 24)
        self.thumb_list.setStyleSheet(
            "QListWidget::item { border: none; background: transparent; }"
            "QListWidget::item:selected { border: none; background: transparent; }")
        self.thumb_list.itemClicked.connect(self._on_thumbnail_item_clicked)
        v.addWidget(self.thumb_list)
        row = QHBoxLayout()
        b_add = QPushButton("Add...")
        b_add.clicked.connect(self.on_add_images)
        b_clear = QPushButton("Clear")
        b_clear.clicked.connect(self.on_clear_images)
        row.addWidget(b_add)
        row.addWidget(b_clear)
        self.btn_rmbg = QPushButton("Remove background")
        self.btn_rmbg.clicked.connect(self.on_remove_background)
        row.addWidget(self.btn_rmbg)
        row.addStretch(1)
        v.addLayout(row)
        self.cutout_label = QLabel("")
        self.cutout_label.setStyleSheet("color: #7bd88f; font-size: 10px;")
        self.cutout_label.setVisible(False)
        v.addWidget(self.cutout_label)
        self.cutout_list = QListWidget()
        self.cutout_list.setViewMode(QListWidget.IconMode)
        self.cutout_list.setResizeMode(QListWidget.Adjust)
        self.cutout_list.setMovement(QListWidget.Static)
        self.cutout_list.setSpacing(6)
        self.cutout_list.setUniformItemSizes(True)
        self.cutout_list.setMaximumHeight(THUMB_SIZE + 12)
        self.cutout_list.setStyleSheet(
            "QListWidget { border: none; background: transparent; }")
        self.cutout_list.setVisible(False)
        v.addWidget(self.cutout_list)
        layout.addWidget(grp_input)

        # UniForge3D settings ---------------------------------------------
        grp_set = QGroupBox("UniForge3D settings")
        form = QFormLayout(grp_set)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Textured (PBR)", "Geometry only"])
        form.addRow("Output mode", self.mode_combo)

        self.grid_combo = QComboBox()
        self.grid_combo.addItems(["256", "384", "512", "768", "1024"])
        self.grid_combo.setCurrentText("512")
        form.addRow("Geometry grid", self.grid_combo)

        self.tex_combo = QComboBox()
        self.tex_combo.addItems(["1024", "2048", "4096"])
        self.tex_combo.setCurrentText("2048")
        self.tex_combo.setToolTip(
            "PBR texture size; also used by Generate Texture.")
        form.addRow("Texture size", self.tex_combo)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(0, 64)
        self.steps_spin.setValue(20)   # pipeline default (all denoise stages)
        self.steps_spin.setSpecialValueText("default")
        form.addRow("Sampler steps", self.steps_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2_147_483_647)
        self.seed_spin.setValue(-1)
        self.seed_spin.setToolTip("-1 uses a fresh random seed for each run.")
        # Toggles the field between "random" and the seed used by the last
        # completed run; the label follows the field's current mode.
        self.btn_seed = QPushButton("Use last")
        self.btn_seed.clicked.connect(self.on_seed_button)
        self.seed_spin.valueChanged.connect(self._update_seed_button)
        seed_row = QHBoxLayout()
        seed_row.setContentsMargins(0, 0, 0, 0)
        seed_row.addWidget(self.seed_spin, 1)
        seed_row.addWidget(self.btn_seed)
        seed_box = QWidget()
        seed_box.setLayout(seed_row)
        form.addRow("Seed", seed_box)
        self._update_seed_button()

        # Geometry only is the default. Texture size stays editable because
        # Generate Texture uses it to re-texture an existing geometry result.
        self.mode_combo.setCurrentIndex(1)

        run_row = QHBoxLayout()
        run_row.setContentsMargins(0, 4, 0, 0)
        self.btn_generate = QPushButton("Generate 3D model")
        self.btn_generate.setFixedHeight(32)
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_generate_texture = QPushButton("Generate Texture")
        self.btn_generate_texture.setFixedHeight(32)
        self.btn_generate_texture.setToolTip(
            "Reuse the last Geometry only result and bake a PBR texture with "
            "the selected Texture size.")
        self.btn_generate_texture.setVisible(False)
        self.btn_generate_texture.clicked.connect(self.on_generate_texture)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setFixedHeight(32)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop)
        run_row.addWidget(self.btn_generate)
        run_row.addWidget(self.btn_generate_texture)
        run_row.addStretch(1)
        run_row.addWidget(self.btn_stop)
        run_box = QWidget()
        run_box.setLayout(run_row)
        form.addRow(run_box)
        layout.addWidget(grp_set)

        # Log ---------------------------------------------------------------
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(5000)
        layout.addWidget(self.log_box, 1)

        self.statusBar().showMessage("Ready.")
        return panel

    def _build_viewer(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        b_open = QPushButton("Open model...")
        b_open.clicked.connect(self.on_open_model)
        b_reset = QPushButton("Reset view")
        b_reset.clicked.connect(self.viewer_reset)
        self.btn_folder = QPushButton("Open output folder")
        self.btn_folder.clicked.connect(self.on_open_folder)
        self.btn_folder.setVisible(False)
        bar.addWidget(b_open)
        bar.addWidget(b_reset)
        bar.addWidget(self.btn_folder)
        bar.addStretch(1)
        self.row_status = QLabel("● Idle")
        self.row_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.row_status.setStyleSheet(
            "color: #7bd88f; font-size: 11px;")
        self.row_status.setToolTip("Current background/GPU activity")
        bar.addWidget(self.row_status)
        bar.addSpacing(12)               # breathing room from the window edge
        v.addLayout(bar)

        self.viewer = viewer_mod.ModelViewer(panel)
        v.addWidget(self.viewer, 1)
        return panel

    # -- thumbnails -------------------------------------------------------
    def _rebuild_thumbs(self):
        self.thumb_list.clear()
        for path in self.image_paths:
            item = QListWidgetItem()
            item.setSizeHint(QSize(THUMB_SIZE + 4, THUMB_SIZE + 4))
            self.thumb_list.addItem(item)
            thumb = _Thumb(path, removable=True, selectable=True)
            thumb.remove.connect(self._remove_image)
            thumb.selected.connect(self._on_reference_selected)
            thumb.set_selected(path == self.ref_reference)
            self.thumb_list.setItemWidget(item, thumb)
        self._rebuild_cutouts()
        self._refresh_buttons()

    def _rebuild_cutouts(self):
        self.cutout_list.clear()
        items = [self.cutout_map[p] for p in self.image_paths
                 if p in self.cutout_map]
        self.cutout_label.setText(
            f"Background removed ({len(items)}/{len(self.image_paths)}):"
            if items else "")
        self.cutout_label.setVisible(bool(items))
        self.cutout_list.setVisible(bool(items))
        for path in items:
            item = QListWidgetItem()
            item.setSizeHint(QSize(THUMB_SIZE, THUMB_SIZE))
            self.cutout_list.addItem(item)
            thumb = _Thumb(path, removable=False, border="#7bd88f")
            self.cutout_list.setItemWidget(item, thumb)

    def _remove_image(self, path: Path):
        if path in self.image_paths:
            self.image_paths.remove(path)
        if path == self.ref_reference:
            self.ref_reference = None
            self.log("Reference cleared (image removed).", "warn")
        self.cutout_map.pop(path, None)
        self._rebuild_thumbs()

    def _on_thumbnail_item_clicked(self, item):
        row = self.thumb_list.row(item)
        if 0 <= row < len(self.image_paths):
            self._on_reference_selected(self.image_paths[row])

    def _on_reference_selected(self, path):
        path = Path(path)
        if path not in self.image_paths:
            return
        if path == self.ref_reference:
            return
        self.ref_reference = path
        self._update_reference_markers()
        cutout = self.cutout_map.get(path)
        note = (" (using existing background cutout)"
                if cutout is not None and cutout.is_file() else "")
        self.log(f"Reference image: {path.name}{note}.", "ok")
        self._refresh_buttons()

    def _update_reference_markers(self):
        for i in range(self.thumb_list.count()):
            widget = self.thumb_list.itemWidget(self.thumb_list.item(i))
            if isinstance(widget, _Thumb):
                widget.set_selected(widget.path == self.ref_reference)

    def _on_cutout_ready(self, source: str, processed: str):
        self.cutout_map[Path(source)] = Path(processed)
        self._rebuild_cutouts()
        self.log(f"Cutout ready: {Path(processed).name}", "ok")
        self._refresh_buttons()

    def _on_rmbg_finished(self, code: int):
        ready = sum(1 for p in self.image_paths if p in self.cutout_map)
        if code != 0:
            self.log(f"Background removal process failed (exit code {code}) "
                     "- see terminal for details.", "error")
        else:
            self.log(f"Background removal finished ({ready}/"
                     f"{len(self.image_paths)} ready); RMBG-2.0 unloaded.", "ok")
        self._refresh_buttons()

    def _refresh_buttons(self):
        busy = self.worker.generating
        cutting = self.rmbg.running
        idle = not busy and not cutting
        has_images = bool(self.image_paths)
        has_texture_state = bool(
            self.texture_state and self.texture_state.is_file())
        self.btn_generate.setEnabled(idle and has_images)
        self.btn_generate_texture.setVisible(has_texture_state)
        self.btn_generate_texture.setEnabled(idle and has_images)
        self.btn_stop.setEnabled(busy)
        self.btn_rmbg.setEnabled(idle and has_images)
        if cutting:
            activity = "Running: background removal"
        elif busy:
            activity = ("Running: texture generation"
                        if self._pending_kind == "texture"
                        else "Running: UniForge3D generation")
        else:
            activity = "Idle"
        if hasattr(self, "row_status"):
            color = "#f0c674" if activity != "Idle" else "#7bd88f"
            self.row_status.setText(f"● {activity}")
            self.row_status.setStyleSheet(
                f"color: {color}; font-size: 11px;")
        self.statusBar().showMessage(
            "Ready." if activity == "Idle" else activity)

    # -- Handlers ---------------------------------------------------------
    def viewer_reset(self):
        if hasattr(self, "viewer"):
            self.viewer.reset_camera()

    def on_add_images(self):
        exts = " ".join("*" + s for s in sorted(IMAGE_SUFFIXES))
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add input image(s)", str(ROOT / "assets"),
            f"Images ({exts})")
        added = False
        for f in files:
            path = Path(f)
            if path not in self.image_paths:
                self.image_paths.append(path)
                added = True
        if added:
            self._rebuild_thumbs()

    def on_clear_images(self):
        self.image_paths.clear()
        self.cutout_map.clear()
        self.ref_reference = None
        self._rebuild_thumbs()

    def on_remove_background(self):
        """Cut out all imported images with RMBG-2.0 in a light process."""
        if self.worker.generating or self.rmbg.running:
            return
        todo = [p for p in self.image_paths if p not in self.cutout_map]
        if not todo:
            self.log("All imported images already have a removed background.",
                     "ok")
            return
        self.log(f"Removing backgrounds for {len(todo)} image(s) "
                 "(loads RMBG-2.0 only, not the full model) ...")
        self._refresh_buttons()
        self.rmbg.start(todo)

    def on_generate(self):
        if self.worker.generating or self.rmbg.running:
            return
        if not self.image_paths:
            QMessageBox.information(self, "UniForge3D",
                                    "Add an input image first.")
            return
        # Swap in pre-made RGBA cutouts so the pipeline skips RMBG-2.0.
        sources = list(self.image_paths)
        images = [self.cutout_map.get(p, p) for p in sources]
        ready = sum(1 for p in sources if p in self.cutout_map)
        if ready:
            self.log(f"Using pre-removed backgrounds for {ready}/{len(sources)}"
                     " image(s) - RMBG-2.0 will be skipped for them.")
        # One fresh folder per run: <stem>_<timestamp>, _2, _3, ... on clash.
        base_name = sources[0].stem + "_" + time.strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_DIR / base_name
        suffix = 2
        while (out_dir / "prediction.glb").exists():
            out_dir = OUTPUT_DIR / f"{base_name}_{suffix}"
            suffix += 1
        geometry_only = self.mode_combo.currentIndex() == 1
        output_path = out_dir / "prediction.glb"
        state_path = _shape_state_path(output_path) if geometry_only else None
        self._pending_kind = "geometry" if geometry_only else "textured"
        self._pending_state = state_path
        self.btn_folder.setVisible(False)
        self._refresh_buttons()
        self.worker.start_run(
            images, output_path,
            seed=(None if self.seed_spin.value() < 0
                  else self.seed_spin.value()),
            grid_size=int(self.grid_combo.currentText()),
            texture_size=int(self.tex_combo.currentText()),
            steps=self.steps_spin.value(),
            texture=not geometry_only,
            shape_state=state_path,
        )
        self.log("Generation queued.", "info")

    def on_generate_texture(self):
        if self.worker.generating or self.rmbg.running:
            return
        if self.texture_state is None or not self.texture_state.is_file():
            QMessageBox.information(
                self, "UniForge3D",
                "Run a Geometry only generation first; Generate Texture "
                "reuses its saved Shape-1024 state.")
            return
        if self.texture_source is None:
            QMessageBox.warning(self, "UniForge3D",
                                "The geometry output path is missing.")
            return
        if not self.image_paths:
            QMessageBox.information(
                self, "UniForge3D",
                "Add the input images used for the geometry pass.")
            return
        # Swap in pre-made RGBA cutouts, exactly like on_generate.
        sources = list(self.image_paths)
        images = [self.cutout_map.get(p, p) for p in sources]
        output_path = _textured_output_path(self.texture_source)
        self._pending_kind = "texture"
        self._pending_state = None
        self.btn_folder.setVisible(False)
        self._refresh_buttons()
        self.worker.start_texture(
            images, output_path, self.texture_state,
            seed=(None if self.seed_spin.value() < 0
                  else self.seed_spin.value()),
            texture_size=int(self.tex_combo.currentText()),
            steps=self.steps_spin.value(),
        )
        self.log(f"Texture generation queued (size "
                 f"{self.tex_combo.currentText()}); reusing "
                 f"{self.texture_source.name}.", "info")

    def on_seed_button(self):
        """Toggle between the last used seed and a random seed."""
        if self.seed_spin.value() < 0:
            if self.last_seed is None:
                return
            self.seed_spin.setValue(self.last_seed)
            self.log(f"Seed set to last used: {self.last_seed}.", "info")
        else:
            self.seed_spin.setValue(-1)
            self.log("Seed reset to random.", "info")

    def _update_seed_button(self, *_):
        if not hasattr(self, "btn_seed"):
            return
        if self.seed_spin.value() < 0:
            self.btn_seed.setText("Use last")
            self.btn_seed.setEnabled(self.last_seed is not None)
            self.btn_seed.setToolTip(
                "Use the seed from the last completed run."
                if self.last_seed is None
                else f"Use the last seed ({self.last_seed}).")
        else:
            self.btn_seed.setText("random")
            self.btn_seed.setEnabled(True)
            self.btn_seed.setToolTip("Reset Seed to a random value.")

    def _clear_pending_job(self):
        self._pending_kind = None
        self._pending_state = None

    def on_stop(self):
        if not self.worker.generating:
            return
        self.worker.stop()
        self._clear_pending_job()
        self.log("Generation stopped (worker killed); the model reloads on "
                 "the next run.", "warn")
        self._refresh_buttons()

    def on_open_model(self):
        start = str(OUTPUT_DIR) if OUTPUT_DIR.is_dir() else str(ROOT)
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open 3D model(s)", start,
            "Models (*.glb *.gltf *.obj *.stl *.ply)")
        loaded_path: Path | None = None
        for f in files:
            self.log(f"Loading {f} ...", "info")
            if self.viewer.load(f):
                self.log(f"Loaded {Path(f).name}", "ok")
                loaded_path = Path(f)
            else:
                self.log(f"Failed to load {f}", "error")
        if loaded_path is not None:
            self.last_output = loaded_path
            self.btn_folder.setVisible(True)
            # Reopening a geometry-only result restores Generate Texture if
            # its shape-state sidecar still exists.
            state = _shape_state_path(loaded_path)
            if state.is_file():
                self.texture_state = state
                self.texture_source = loaded_path
            else:
                self.texture_state = None
                self.texture_source = None
            self._clear_pending_job()
        self._refresh_buttons()

    def on_open_folder(self):
        """Open the folder holding the last generated model (glb/obj/...)."""
        import os
        target = (self.last_output.parent if self.last_output
                  else OUTPUT_DIR)
        if target.is_dir():
            os.startfile(str(target))

    # -- Worker events ------------------------------------------------------
    def _on_worker_log(self, msg: str, level: str):
        self.log(msg, level)

    def _on_state_changed(self, busy: bool):
        self._refresh_buttons()

    def _on_done(self, output: str, seed=None):
        self.log(f"Done -> {output}", "ok")
        out = Path(output)
        kind = self._pending_kind
        pending_state = self._pending_state
        self._clear_pending_job()
        if seed is not None:
            try:
                self.last_seed = int(seed)
            except (TypeError, ValueError):
                pass
            else:
                self._update_seed_button()
        self.last_output = out
        self.btn_folder.setVisible(True)
        if kind == "geometry":
            if pending_state is not None and pending_state.is_file():
                self.texture_state = pending_state
                self.texture_source = out
                self.log("Geometry saved; Generate Texture is available.", "ok")
            else:
                self.texture_state = None
                self.texture_source = None
                self.log("Geometry finished, but its texture state was not "
                         "saved; Generate Texture is unavailable.", "warn")
        elif kind == "textured":
            # A full textured run replaces the geometry-only reference.
            self.texture_state = None
            self.texture_source = None
        # kind == "texture": keep the geometry state for further re-textures.
        same_model = (
            kind == "texture"
            and self.texture_source is not None
            and self.viewer.current_path is not None
            and self.viewer.current_path.resolve()
            == self.texture_source.resolve()
        )
        if same_model and self.viewer.add_textured(out):
            self.log("Texture added; the first three materials are unchanged. "
                     "Click the 4th swatch to view the textured model.", "ok")
        elif self.viewer.load(output):
            self.log(f"Loaded result into viewer: {out.name}", "ok")
        else:
            self.log("Could not display result in viewer.", "error")
        # automatically publish an OBJ next to the GLB
        self.converter.convert(out)
        self._refresh_buttons()

    def _on_failed(self, msg: str):
        self._clear_pending_job()
        self.log(msg, "error")
        QMessageBox.warning(self, "Generation failed", msg)
        self._refresh_buttons()

    # -- misc -----------------------------------------------------------------
    def log(self, msg: str, level: str = "info"):
        self.log_box.appendPlainText(msg)

    def closeEvent(self, event):
        self.rmbg.shutdown()
        self.worker.shutdown()
        super().closeEvent(event)


def main() -> int:
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("UniForge3D")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
