"""UniForge3D - PySide6 + VTK desktop UI for image-to-3D generation.

Native Windows app (no browser): PySide6 controls on the left, an embedded
VTK 3D viewer (GLB/OBJ/STL/PLY with PBR materials) on the right.

Feature parity with the Flet UI:
  - imported-image thumbnails with per-image remove buttons
  - pre-generation background removal (RMBG-2.0 in a light worker process),
    with the resulting RGBA cutouts shown as thumbnails of their own; Generate
    runs this automatically for any image that has no cutout yet
  - textured / geometry-only mode, grid, texture size, steps, seed, the
    Low VRAM switch (on by default; off runs fp32 weights + bf16 compute),
    and an optional early stop after Shape-512 (skips the 1024 upscale);
    Run 2nd stage can later resume the 1024 upscale from that saved mesh,
    then Show low / Show high swaps prediction_512.glb and prediction.glb
    in place (camera and material preserved)
  - Run Hi3D 1536 refines the same saved Shape-512 mesh with Hi3D v1.0
    (image-conditioned, geometry only); a dropdown picks the passes:
    1-3 (out 1536), 2-3 (out 1536), 1 (out 512), 2 (out 1024) or
    3 (out 1536). The result is saved as prediction_hi3d_<out>.glb next to
    the preview. It runs in one-shot hi3d_worker.py so its
    transformers/VAE unload immediately afterwards; Show low / Show high
    then compares it with the saved Shape-512 mesh. The decode phase
    reports per-stage progress in the log.
  - Run CelloCut (watertight) repairs the same saved Shape-512 mesh with
    CelloCut (constructive 512^3 UDF + graph-cut remeshing; no model
    download). The repaired mesh REPLACES prediction_512.glb in place AND
    regenerates its prediction_512.shape512.pt state, so Run 2nd stage and
    Hi3D start from the fixed geometry; the untouched originals are kept
    (prediction_512_original.glb / .shape512.pt) and the 3D view reloads
    the repaired mesh. Show low / Show high then compares original vs
    watertight.

Generation runs in a separate worker process (worker.py in this directory) so
Stop can hard-kill a run and this process stays lightweight (no torch import).
Background removal uses rmbg_worker.py, which only loads RMBG-2.0 (~never the
26 GB pipeline).

Run with:  .venv/Scripts/python.exe PS6_UI/main.py   (or start_ps6.bat)
"""

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from PySide6.QtCore import (
    QEvent, QObject, QPoint, QSize, Qt, QThread, QTimer, Signal,
)
from PySide6.QtGui import QAction, QColor, QPalette, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QSpinBox, QSplitter, QToolButton, QVBoxLayout, QWidget,
)

import viewer as viewer_mod
from model_paths import find_model_dir, missing_hi3d_assets

ROOT = Path(__file__).resolve().parent.parent
WORKER = Path(__file__).resolve().parent / "worker.py"
RMBG_WORKER = Path(__file__).resolve().parent / "rmbg_worker.py"
HI3D_WORKER = Path(__file__).resolve().parent / "hi3d_worker.py"
CELLO_WORKER = Path(__file__).resolve().parent / "cello_worker.py"
MODEL_DIR = find_model_dir()
OUTPUT_DIR = ROOT / "output_dir"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

THUMB_SIZE = 48


def _shape_state_path(glb: Path) -> Path:
    """Geometry-only GLB -> the Shape-1024 state saved next to it."""
    return glb.with_name(glb.stem + ".shape_state.pt")


def _shape512_state_path(glb: Path) -> Path:
    """Early-stop GLB -> the Shape-512 mesh saved next to it."""
    return glb.with_name(glb.stem + ".shape512.pt")


def _textured_output_path(glb: Path) -> Path:
    """Geometry-only GLB -> the textured GLB generated from it."""
    return glb.with_name(glb.stem + "_textured.glb")


def _high_output_path(low_glb: Path) -> Path:
    """Shape-512 preview GLB -> the Shape-1024 GLB in the same folder."""
    return low_glb.with_name("prediction.glb")


def _hi3d_output_path(low_glb: Path, resolution: int) -> Path:
    """Shape-512 preview GLB -> the Hi3D result GLB in the same folder."""
    return low_glb.with_name(f"prediction_hi3d_{resolution}.glb")


def _cellocut_output_path(low_glb: Path) -> Path:
    """Shape-512 preview GLB -> CelloCut's scratch output in the same folder.
    On success this file replaces the Shape-512 GLB itself."""
    return low_glb.with_name("prediction_512_cellocut.glb")


def _cellocut_backup_path(low_glb: Path) -> Path:
    """Shape-512 preview GLB -> backup of the unfixed mesh, written the
    first time CelloCut replaces it."""
    return low_glb.with_name("prediction_512_original.glb")


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
    worker_dead = Signal(int)            # nonzero exit code
    reader_finished = Signal(int)        # exit code, always emitted

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
            self.reader_finished.emit(code if code is not None else 0)


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
        self.low_vram = True
        # Memory mode the live worker was started with. The worker loads the
        # pipeline once, so changing the checkbox has to restart it.
        self._spawned_low_vram: bool | None = None

    @property
    def generating(self) -> bool:
        return self._generating

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def set_low_vram(self, enabled: bool) -> None:
        self.low_vram = bool(enabled)

    def _spawn(self):
        if self._proc and self._proc.poll() is None:
            if self._spawned_low_vram == self.low_vram:
                return
            # Idle worker from the other memory mode: reload before this job.
            self.stop()
        args = [sys.executable, str(WORKER), "--model-dir", str(MODEL_DIR),
                "--low-vram" if self.low_vram else "--no-low-vram"]
        self._spawned_low_vram = self.low_vram
        self._proc = subprocess.Popen(
            args,
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
                  shape_state: Path | None = None,
                  stop_at_stage1: bool = False):
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
            "stop_at_stage1": stop_at_stage1,
        }
        if shape_state is not None:
            cmd["shape_state"] = str(shape_state)
        self._send(cmd)

    def start_stage2(self, images, output: Path, shape512_state: Path, *,
                     seed, grid_size: int, texture_size: int, steps: int,
                     texture: bool = False, shape_state: Path | None = None):
        cmd = {
            "cmd": "stage2",
            "images": [str(p) for p in images],
            "output": str(output),
            "shape512_state": str(shape512_state),
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
        self._spawned_low_vram = None
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
        self._spawned_low_vram = None
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
# Hi3D v1.0 refinement worker plumbing (one-shot process per refine)
# ---------------------------------------------------------------------------
class Hi3DController(QObject):
    """Runs hi3d_worker.py: load Hi3D, refine one mesh, exit.

    A fresh process per refinement keeps the ~11 GB of Hi3D weights out of
    the UniForge3D worker and releases them the moment the job ends.
    """

    log_line = Signal(str, str)
    state_changed = Signal(bool)         # True while refining
    refine_done = Signal(str, object)    # output path, seed used
    refine_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: subprocess.Popen | None = None
        self._reader: _WorkerReader | None = None
        self._refining = False
        self.low_vram = True

    @property
    def refining(self) -> bool:
        return self._refining

    def set_low_vram(self, enabled: bool) -> None:
        self.low_vram = bool(enabled)

    def start(self, image: Path, coarse_mesh: Path, output: Path, *,
              resolution: int = 1536, seed=None, start_structure=None,
              steps: int | None = None, budget_percent: int = 100,
              low_vram: bool | None = None):
        if self._refining:
            return
        # A finished one-shot worker may still be shutting down; reap it
        # before spawning the next refinement.
        if self._proc is not None and self._proc.poll() is None:
            self.stop()
        use_low_vram = self.low_vram if low_vram is None else low_vram
        args = [sys.executable, str(HI3D_WORKER),
                "--model-dir", str(MODEL_DIR),
                "--low-vram" if use_low_vram else "--no-low-vram"]
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, cwd=str(ROOT),
        )
        self._reader = _WorkerReader(self._proc, self)
        self._reader.log_line.connect(self.log_line)
        self._reader.job_done.connect(self._on_done)
        self._reader.job_failed.connect(self._on_failed)
        self._reader.worker_dead.connect(self._on_dead)
        self._reader.reader_finished.connect(self._on_reader_finished)
        self._reader.start()
        # The worker only reads stdin after its ~11 GB model load, so this
        # small command simply waits in the pipe buffer.
        cmd = {
            "cmd": "refine",
            "image": str(image),
            "coarse_mesh": str(coarse_mesh),
            "output": str(output),
            "resolution": int(resolution),
            "seed": seed,
        }
        if start_structure is not None:
            cmd["start_structure"] = int(start_structure)
        if steps is not None and int(steps) > 0:
            cmd["steps"] = int(steps)
        if int(budget_percent) != 100:
            cmd["budget_percent"] = int(budget_percent)
        self._proc.stdin.write(json.dumps(cmd) + "\n")
        self._proc.stdin.flush()
        self._refining = True
        self.state_changed.emit(True)

    def stop(self):
        """Hard-kill the worker; a stopped refine leaves no partial output."""
        if self._proc is None and self._reader is None:
            return
        proc, self._proc = self._proc, None
        if self._reader is not None:
            for signal, slot in (
                (self._reader.job_done, self._on_done),
                (self._reader.job_failed, self._on_failed),
                (self._reader.worker_dead, self._on_dead),
                (self._reader.reader_finished, self._on_reader_finished),
            ):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._reap_reader()
        self._refining = False
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
        self._refining = False
        self.state_changed.emit(False)
        self.refine_done.emit(output, seed)

    def _on_failed(self, msg: str):
        self._refining = False
        self.state_changed.emit(False)
        self.refine_failed.emit(msg)

    def _on_dead(self, code: int):
        if self._refining:
            self._refining = False
            self.state_changed.emit(False)
            self.refine_failed.emit(
                f"Hi3D worker exited unexpectedly (code {code}).")
        self.log_line.emit(f"Hi3D worker process exited (code {code}).",
                           "warn")

    def _on_reader_finished(self, code: int):
        # Ignore a late signal from a worker that a newer run replaced.
        if self.sender() is not self._reader:
            return
        self._reap_reader()
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def shutdown(self):
        if self._reader is not None:
            for signal, slot in (
                (self._reader.job_done, self._on_done),
                (self._reader.job_failed, self._on_failed),
                (self._reader.worker_dead, self._on_dead),
                (self._reader.reader_finished, self._on_reader_finished),
            ):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._reap_reader()
        self._refining = False


# ---------------------------------------------------------------------------
# CelloCut watertight-remeshing worker plumbing (one-shot process per run)
# ---------------------------------------------------------------------------
class CelloCutController(QObject):
    """Runs cello_worker.py: load the CelloCut modules, repair one mesh,
    exit. CelloCut uses no network weights, but the 512^3 UDF rasterization
    needs the GPU, so a fresh process per run keeps it out of the way of the
    UniForge3D worker (which is unloaded while this runs, like for Hi3D).
    """

    log_line = Signal(str, str)
    state_changed = Signal(bool)         # True while repairing
    cut_done = Signal(str, object)       # output path, seed (unused)
    cut_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: subprocess.Popen | None = None
        self._reader: _WorkerReader | None = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self, mesh: Path, output: Path, *, shape512_state: Path | None = None,
              voxel_res: int = 512, decimate_ratio: float = 0.95,
              lambda_fill: int = 20):
        if self._running:
            return
        # A finished one-shot worker may still be shutting down; reap it
        # before spawning the next run.
        if self._proc is not None and self._proc.poll() is None:
            self.stop()
        self._proc = subprocess.Popen(
            [sys.executable, str(CELLO_WORKER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, cwd=str(ROOT),
        )
        self._reader = _WorkerReader(self._proc, self)
        self._reader.log_line.connect(self.log_line)
        self._reader.job_done.connect(self._on_done)
        self._reader.job_failed.connect(self._on_failed)
        self._reader.worker_dead.connect(self._on_dead)
        self._reader.reader_finished.connect(self._on_reader_finished)
        self._reader.start()
        # The worker only reads stdin after its module imports, so this
        # small command simply waits in the pipe buffer.
        cmd = {
            "cmd": "watertight",
            "mesh": str(mesh),
            "output": str(output),
            "voxel_res": int(voxel_res),
            "decimate_ratio": float(decimate_ratio),
            "lambda_fill": int(lambda_fill),
        }
        if shape512_state is not None:
            cmd["shape512_state"] = str(shape512_state)
        self._proc.stdin.write(json.dumps(cmd) + "\n")
        self._proc.stdin.flush()
        self._running = True
        self.state_changed.emit(True)

    def stop(self):
        """Hard-kill the worker; a stopped run leaves no partial output."""
        if self._proc is None and self._reader is None:
            return
        proc, self._proc = self._proc, None
        if self._reader is not None:
            for signal, slot in (
                (self._reader.job_done, self._on_done),
                (self._reader.job_failed, self._on_failed),
                (self._reader.worker_dead, self._on_dead),
                (self._reader.reader_finished, self._on_reader_finished),
            ):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._reap_reader()
        self._running = False
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
        self._running = False
        self.state_changed.emit(False)
        self.cut_done.emit(output, seed)

    def _on_failed(self, msg: str):
        self._running = False
        self.state_changed.emit(False)
        self.cut_failed.emit(msg)

    def _on_dead(self, code: int):
        if self._running:
            self._running = False
            self.state_changed.emit(False)
            self.cut_failed.emit(
                f"CelloCut worker exited unexpectedly (code {code}).")
        self.log_line.emit(
            f"CelloCut worker process exited (code {code}).", "warn")

    def _on_reader_finished(self, code: int):
        # Ignore a late signal from a worker that a newer run replaced.
        if self.sender() is not self._reader:
            return
        self._reap_reader()
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def shutdown(self):
        if self._reader is not None:
            for signal, slot in (
                (self._reader.job_done, self._on_done),
                (self._reader.job_failed, self._on_failed),
                (self._reader.worker_dead, self._on_dead),
                (self._reader.reader_finished, self._on_reader_finished),
            ):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._reap_reader()
        self._running = False


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
        # Early-stop runs save the Shape-512 mesh next to the GLB; Run 2nd
        # stage reuses it to do the 1024 upscale without redoing stage 1.
        self.stage1_state: Path | None = None
        self.stage1_source: Path | None = None
        self.stage1_images: list[Path] | None = None
        # Low/high comparison pair offered once both prediction_512.glb and
        # prediction.glb exist; compare_align caches the low mesh bounds.
        self.compare_low: Path | None = None
        self.compare_high: Path | None = None
        self.compare_align = None
        self._pending_kind: str | None = None
        self._pending_state: Path | None = None
        self._pending_images: list[Path] | None = None
        # Set while Generate waits for automatic background cutouts.
        self._after_rmbg: Callable[[], None] | None = None
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

        self.hi3d = Hi3DController(self)
        self.hi3d.log_line.connect(self._on_worker_log)
        self.hi3d.state_changed.connect(self._on_state_changed)
        self.hi3d.refine_done.connect(self._on_hi3d_done)
        self.hi3d.refine_failed.connect(self._on_hi3d_failed)

        self.cellocut = CelloCutController(self)
        self.cellocut.log_line.connect(self._on_worker_log)
        self.cellocut.state_changed.connect(self._on_state_changed)
        self.cellocut.cut_done.connect(self._on_cellocut_done)
        self.cellocut.cut_failed.connect(self._on_cellocut_failed)

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
        splitter.setSizes([310, 1190])
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
        self.grid_combo.setCurrentText("1024")
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

        self.low_vram_check = QCheckBox("Low VRAM")
        self.low_vram_check.setChecked(True)
        self.low_vram_check.setToolTip(
            "ON: keeps only the active stage's weights on the GPU and uses "
            "fp16 weights/compute. Fits smaller cards.\n"
            "OFF: fp32 weights with bf16 compute, closer to the upstream "
            "pipeline; needs substantially more GPU memory. The worker "
            "reloads when this changes.")
        self.low_vram_check.toggled.connect(self._on_low_vram_toggled)
        form.addRow("Memory mode", self.low_vram_check)

        self.stop_stage1_check = QCheckBox("Stop at 1st stage")
        self.stop_stage1_check.setChecked(True)
        self.stop_stage1_check.setToolTip(
            "OFF: full pipeline.\n"
            "ON: decode the Shape-512 mesh and export it after the "
            "sparse-structure stage, skipping the Shape-1024 upscale and "
            "texture stages. Faster; useful for checking the first geometry "
            "pass. Texture state is not saved in this mode.")
        self._stop_stage1_previous_mode = self.mode_combo.currentIndex()
        self.stop_stage1_check.toggled.connect(self._on_stop_stage1_toggled)
        form.addRow("Early stop", self.stop_stage1_check)

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

        # Upscale settings --------------------------------------------------
        # Shown only after an Early stop run saved a Shape-512 sidecar.
        self.grp_upscale = QGroupBox("Upscale settings")
        upscale_col = QVBoxLayout(self.grp_upscale)
        upscale_col.setContentsMargins(8, 6, 8, 6)

        # Row above "Run 2nd stage": CelloCut watertight repair.
        cello_row = QHBoxLayout()
        cello_row.setContentsMargins(0, 0, 0, 0)
        self.btn_cellocut = QPushButton("Run CelloCut (watertight)")
        self.btn_cellocut.setFixedHeight(28)
        self.btn_cellocut.setToolTip(
            "Repair the saved Shape-512 mesh with CelloCut: it rebuilds the "
            "mesh from a 512^3 unsigned-distance volume, solves a graph-cut "
            "cell labeling, and returns a strictly watertight solid (fixes "
            "holes, self-intersections, non-manifold edges).\n"
            "The repaired mesh REPLACES prediction_512.glb AND its "
            "prediction_512.shape512.pt state, so both Run 2nd stage and "
            "Hi3D continue from the fixed geometry. The originals are kept "
            "once (prediction_512_original.glb / .shape512.pt) and the 3D "
            "view reloads the repaired mesh.\n"
            "Needs the GPU (CelloCut wheels are installed by install.bat; "
            "no model download).")
        self.btn_cellocut.clicked.connect(self.on_run_cellocut)
        cello_row.addWidget(self.btn_cellocut)

        # "Run 2nd stage" shares the CelloCut row, offset a bit to the right.
        self.btn_stage2 = QPushButton("Run 2nd stage")
        self.btn_stage2.setFixedHeight(28)
        self.btn_stage2.setToolTip(
            "Continue from the last Early stop result: run the Shape-1024 "
            "upscale from the saved Shape-512 mesh and replace the preview "
            "GLB with the refined geometry. Generate Texture becomes "
            "available afterwards.")
        self.btn_stage2.clicked.connect(self.on_run_stage2)
        cello_row.addSpacing(16)
        cello_row.addWidget(self.btn_stage2)
        cello_row.addStretch(1)
        upscale_col.addLayout(cello_row)

        # Hi3D section header (underlined), above the experiment knobs.
        hi3d_header = QLabel("Hi3D (optional) :")
        header_font = hi3d_header.font()
        header_font.setUnderline(True)
        hi3d_header.setFont(header_font)
        upscale_col.addWidget(hi3d_header)

        # Hi3D experiment knobs: sampling steps per pass and the
        # occupied-voxel budget scale. Both trade quality for speed/VRAM.
        # Row 1: steps + pass dropdown.
        hi3d_row = QHBoxLayout()
        hi3d_row.setContentsMargins(0, 0, 0, 0)
        hi3d_label = QLabel("steps")
        hi3d_row.addWidget(hi3d_label)
        self.hi3d_steps = QSpinBox()
        self.hi3d_steps.setRange(4, 50)
        self.hi3d_steps.setValue(25)
        self.hi3d_steps.setToolTip(
            "Sampling steps per Hi3D pass (default 25, the validated "
            "value). Fewer steps finish sampling proportionally faster "
            "with somewhat rougher geometry; try 12-15 for quick tests.")
        hi3d_row.addWidget(self.hi3d_steps)
        hi3d_row.addSpacing(12)

        # Hi3D pass selection: which ladder rungs to run. Each rung refines at
        # its structure resolution and outputs its own mesh resolution
        # (64 -> 512, 128 -> 1024, 192 -> 1536); a "N pass" entry runs just
        # that rung, a "X-Y pass" entry runs rungs X through Y.
        self.hi3d_pass_combo = QComboBox()
        self.hi3d_pass_combo.setFixedHeight(28)
        for label, start, resolution in (
            ("1-3 pass", None, 1536),
            ("2-3 pass", 128, 1536),
            ("1 pass", 64, 512),
            ("2 pass", 128, 1024),
            ("3 pass", 192, 1536),
        ):
            self.hi3d_pass_combo.addItem(label, (start, resolution))
        self.hi3d_pass_combo.setCurrentIndex(4)   # default: 3 pass
        self.hi3d_pass_combo.setToolTip(
            "Which Hi3D v1.0 passes to run on the saved Shape-512 mesh.\n"
            "Pass 1 runs at structure 64 and outputs 512, pass 2 at 128 -> "
            "1024, pass 3 at 192 -> 1536. A 'N pass' entry runs just that "
            "rung; a 'X-Y pass' entry runs rungs X through Y. Starting at "
            "pass 2 or 3 voxelizes the saved mesh at that structure and "
            "skips the earlier passes.")
        hi3d_row.addWidget(self.hi3d_pass_combo)
        hi3d_row.addStretch(1)
        upscale_col.addLayout(hi3d_row)

        # Row 2: voxel budget + full-GPU switch + the run button.
        hi3d_row2 = QHBoxLayout()
        hi3d_row2.setContentsMargins(0, 0, 0, 0)
        hi3d_label2 = QLabel("budget")
        hi3d_row2.addWidget(hi3d_label2)
        self.hi3d_budget = QSpinBox()
        self.hi3d_budget.setRange(25, 100)
        self.hi3d_budget.setValue(100)
        self.hi3d_budget.setSuffix("%")
        self.hi3d_budget.setToolTip(
            "Scale for the occupied-voxel budgets of each pass (default "
            "100% = 80k/90k/110k voxels). Fewer voxels means fewer tokens: "
            "sampling and the memory-heavy decode both get faster and "
            "smaller, at the cost of fine detail. The OOM retry already "
            "drops to 65% automatically.")
        hi3d_row2.addWidget(self.hi3d_budget)
        hi3d_row2.addSpacing(12)
        self.hi3d_fullgpu = QCheckBox("Full GPU")
        self.hi3d_fullgpu.setToolTip(
            "OFF (default): Hi3D uses sequential CPU offload when the global "
            "Low VRAM switch is on - layers stream on/off the GPU per "
            "forward pass, so GPU utilization looks low (10-30%) and it "
            "runs slowly.\n"
            "ON: keep all Hi3D weights resident on the GPU for this run - "
            "much faster and the GPU is actually used, but the 1536 decode "
            "peaks several GB higher and can OOM on 24-32 GB cards (it "
            "retries once with 65% voxel budgets on OOM).")
        hi3d_row2.addWidget(self.hi3d_fullgpu)
        hi3d_row2.addStretch(1)
        upscale_col.addLayout(hi3d_row2)

        # Run button on its own row, aligned to the right edge.
        hi3d_row3 = QHBoxLayout()
        hi3d_row3.setContentsMargins(0, 0, 0, 0)
        self.btn_hi3d = QPushButton("Run Hi3D 1536")
        self.btn_hi3d.setFixedHeight(28)
        self.btn_hi3d.setToolTip(
            "Run Hi3D v1.0 with the selected passes, conditioned on the "
            "selected REF image (or that run's first image when no REF is "
            "selected).\nGeometry only: writes prediction_hi3d_<output "
            "resolution>.glb next to the Shape-512 preview. Hi3D runs in "
            "its own process and unloads when done; the 1536 decode is "
            "memory-intensive.")
        self.btn_hi3d.clicked.connect(self.on_run_hi3d)
        hi3d_row3.addStretch(1)
        hi3d_row3.addWidget(self.btn_hi3d)
        upscale_col.addLayout(hi3d_row3)

        layout.addWidget(self.grp_upscale)

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
        bar.addSpacing(12)
        self.btn_compare = QPushButton("Show low")
        self.btn_compare.setToolTip(
            "Swap the viewer between the saved low-resolution mesh and the "
            "refined high-resolution mesh from this result (Shape-512 vs "
            "Shape-1024, or Shape-512 vs Hi3D 1536), keeping the current "
            "camera and material.")
        self.btn_compare.setVisible(False)
        self.btn_compare.clicked.connect(self.on_toggle_compare)
        bar.addWidget(self.btn_compare)
        bar.addSpacing(12)
        self.backface_check = QCheckBox("Backface culling")
        self.backface_check.setChecked(True)   # default ON (no listener yet)
        self.backface_check.setToolTip(
            "Hide polygons that face away from the camera. A closed, "
            "consistently oriented mesh looks unchanged, while double-sided "
            "sheets, flipped patches and holes become see-through - use it "
            "to spot surfaces that are not single-sided watertight.")
        self.backface_check.toggled.connect(
            lambda checked: self.viewer.set_backface_culling(checked))
        bar.addWidget(self.backface_check)
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
        # A Generate click that found images without cutouts waits here; the
        # pipeline can still remove any that failed, so continue regardless.
        start, self._after_rmbg = self._after_rmbg, None
        if start is not None:
            if code != 0:
                self.log("Continuing with generation; the pipeline will "
                         "remove backgrounds for any missing cutouts.", "warn")
            start()

    def _refresh_buttons(self):
        busy = self.worker.generating
        refining = self.hi3d.refining
        cutting = self.rmbg.running
        repairing = self.cellocut.running
        idle = not busy and not cutting and not refining and not repairing
        has_images = bool(self.image_paths)
        has_texture_state = bool(
            self.texture_state and self.texture_state.is_file())
        has_stage1_state = bool(
            self.stage1_state and self.stage1_state.is_file())
        self.btn_generate.setEnabled(idle and has_images)
        self.btn_generate_texture.setVisible(has_texture_state)
        self.btn_generate_texture.setEnabled(idle and has_images)
        if hasattr(self, "grp_upscale"):
            # The panel is visible from the start but stays disabled until an
            # Early stop run has saved a Shape-512 state to continue from.
            self.grp_upscale.setEnabled(has_stage1_state)
        if hasattr(self, "btn_stage2"):
            self.btn_stage2.setEnabled(idle)
        if hasattr(self, "btn_hi3d"):
            self.btn_hi3d.setEnabled(idle)
        if hasattr(self, "hi3d_pass_combo"):
            self.hi3d_pass_combo.setEnabled(idle)
        if hasattr(self, "hi3d_fullgpu"):
            self.hi3d_fullgpu.setEnabled(idle)
        if hasattr(self, "btn_cellocut"):
            self.btn_cellocut.setEnabled(idle)
        self.btn_stop.setEnabled(busy or refining or repairing)
        self.btn_rmbg.setEnabled(idle and has_images)
        if hasattr(self, "low_vram_check"):
            self.low_vram_check.setEnabled(idle)
        if hasattr(self, "stop_stage1_check"):
            self.stop_stage1_check.setEnabled(idle)
            # Early stop implies geometry-only output; keep the mode pinned
            # while it is active (and while a run is in flight).
            self.mode_combo.setEnabled(
                idle and not self.stop_stage1_check.isChecked())
        if hasattr(self, "btn_compare"):
            self._update_compare_button()
        if cutting:
            activity = "Running: background removal"
        elif refining:
            activity = "Running: Hi3D 1536 refinement"
        elif repairing:
            activity = "Running: CelloCut watertight repair"
        elif busy:
            if self._pending_kind == "texture":
                activity = "Running: texture generation"
            elif self._pending_kind == "stage2":
                activity = "Running: 2nd stage (1024 upscale)"
            else:
                activity = "Running: UniForge3D generation"
        else:
            activity = "Idle"
        if hasattr(self, "row_status"):
            color = "#f0c674" if activity != "Idle" else "#7bd88f"
            self.row_status.setText(f"● {activity}")
            self.row_status.setStyleSheet(
                f"color: {color}; font-size: 11px;")
        self.statusBar().showMessage(
            "Ready." if activity == "Idle" else activity)

    def _update_compare_button(self):
        """Show Show low/Show high only while both comparison GLBs exist."""
        has_pair = bool(
            self.compare_low and self.compare_high
            and self.compare_low.is_file() and self.compare_high.is_file())
        self.btn_compare.setVisible(has_pair)
        if not has_pair:
            return
        current = self.viewer.current_path
        showing_low = (current is not None
                       and current.resolve() == self.compare_low.resolve())
        self.btn_compare.setText("Show high" if showing_low else "Show low")
        self.btn_compare.setEnabled(
            not self.worker.generating and not self.rmbg.running
            and not self.hi3d.refining)

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
        if self._after_rmbg is not None:
            self._after_rmbg = None
            self.log("Pending generation cancelled (images cleared).", "warn")
        self.image_paths.clear()
        self.cutout_map.clear()
        self.ref_reference = None
        self._rebuild_thumbs()

    def on_remove_background(self):
        """Cut out all imported images with RMBG-2.0 in a light process."""
        if self.worker.generating or self.rmbg.running or self.hi3d.refining:
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

    def _queue_after_cutouts(self, sources: list[Path],
                             start: Callable[[], None]) -> None:
        """Run RMBG for missing cutouts, then invoke *start*.

        The pipeline removes backgrounds internally for inputs without alpha,
        but those results never reach the UI. Doing the pass here first makes
        the automatic cutouts visible, exactly like pressing Remove background
        manually, and lets the pipeline skip RMBG-2.0 afterwards.
        """
        todo = [p for p in sources if p not in self.cutout_map]
        if not todo:
            start()
            return
        self.log(f"Removing backgrounds for {len(todo)} image(s) before "
                 "generation ...")
        self._after_rmbg = start
        self._refresh_buttons()
        try:
            self.rmbg.start(todo)
        except Exception as ex:  # keep generation possible if RMBG cannot start
            self._after_rmbg = None
            self.log(f"Background removal could not start: {ex}", "error")
            self.log("Starting generation; the pipeline will remove "
                     "backgrounds internally.", "warn")
            start()

    def on_generate(self):
        if self.worker.generating or self.rmbg.running or self.hi3d.refining:
            return
        if not self.image_paths:
            QMessageBox.information(self, "UniForge3D",
                                    "Add an input image first.")
            return
        sources = list(self.image_paths)
        self._queue_after_cutouts(
            sources, lambda: self._start_generate(sources))

    def _start_generate(self, sources: list[Path]):
        sources = [p for p in sources if p in self.image_paths]
        if not sources:
            self.log("No input images left; generation cancelled.", "warn")
            return
        # Swap in pre-made RGBA cutouts so the pipeline skips RMBG-2.0.
        images = [self.cutout_map.get(p, p) for p in sources]
        ready = sum(1 for p in sources if p in self.cutout_map)
        if ready:
            self.log(f"Using pre-removed backgrounds for {ready}/{len(sources)}"
                     " image(s) - RMBG-2.0 will be skipped for them.")
        # A new generation supersedes any earlier Shape-512 preview.
        self.stage1_state = None
        self.stage1_source = None
        self.stage1_images = None
        self.compare_low = None
        self.compare_high = None
        self.compare_align = None
        geometry_only = self.mode_combo.currentIndex() == 1
        stop_at_stage1 = self.stop_stage1_check.isChecked()
        geometry_only = geometry_only or stop_at_stage1
        # One fresh folder per run: <stem>_<timestamp>, _2, _3, ... on clash.
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
        # Early stop has no Shape-1024 latent, so there is no texture state.
        state_path = (_shape_state_path(output_path)
                      if geometry_only and not stop_at_stage1 else None)
        self._pending_kind = ("stage1" if stop_at_stage1 else
                              "geometry" if geometry_only else "textured")
        self._pending_state = state_path
        self._pending_images = list(images)
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
            stop_at_stage1=stop_at_stage1,
        )
        self.log("Generation queued.", "info")

    def on_run_stage2(self):
        if self.worker.generating or self.rmbg.running or self.hi3d.refining:
            return
        if self.stage1_state is None or not self.stage1_state.is_file():
            QMessageBox.information(
                self, "UniForge3D",
                "Run a generation with Early stop enabled first; Run 2nd "
                "stage reuses its saved Shape-512 mesh.")
            self._refresh_buttons()
            return
        if self.stage1_source is None or self.stage1_images is None:
            QMessageBox.warning(
                self, "UniForge3D",
                "The Early stop result is incomplete; run that generation "
                "again.")
            return
        missing = [p for p in self.stage1_images if not p.exists()]
        if missing:
            QMessageBox.warning(
                self, "UniForge3D",
                "Input images from the Early stop run are missing:\n"
                + "\n".join(str(p) for p in missing))
            return
        # Keep the Shape-512 preview as prediction_512.glb and write the
        # refined result next to it as prediction.glb for comparison.
        output_path = _high_output_path(self.stage1_source)
        state_path = _shape_state_path(output_path)
        self._pending_kind = "stage2"
        self._pending_state = state_path
        self._pending_images = list(self.stage1_images)
        self.btn_folder.setVisible(False)
        self._refresh_buttons()
        self.worker.start_stage2(
            list(self.stage1_images), output_path, self.stage1_state,
            seed=(None if self.seed_spin.value() < 0
                  else self.seed_spin.value()),
            grid_size=int(self.grid_combo.currentText()),
            texture_size=int(self.tex_combo.currentText()),
            steps=self.steps_spin.value(),
            texture=False,
            shape_state=state_path,
        )
        self.log("Running 2nd stage: Shape-1024 upscale from the saved "
                 "Shape-512 mesh (stage 1 is not re-run) ...", "info")

    def _hi3d_condition_image(self) -> Path | None:
        """Single Hi3D condition image: the REF selection when it was used.

        REF points at an original import while the Early stop run stored
        cutout-substituted paths, so map through cutout_map first. If no REF
        is selected (or it was not part of that run), use the run's first
        image.
        """
        if not self.stage1_images:
            return None
        run_images = {p.resolve() for p in self.stage1_images}
        if self.ref_reference is not None:
            source = self.ref_reference
            candidate = self.cutout_map.get(source, source)
            for path in (candidate, source):
                if path.resolve() in run_images:
                    return path
            self.log("The selected REF image was not part of the Early stop "
                     "run; using that run's first image for Hi3D.", "warn")
        return self.stage1_images[0]

    def on_run_hi3d(self):
        """Run Hi3D v1.0 with the passes chosen in the dropdown."""
        selection = self.hi3d_pass_combo.currentData() or (None, 1536)
        self._run_hi3d(start_structure=selection[0], resolution=selection[1])

    def _run_hi3d(self, *, start_structure, resolution=1536):
        if self.worker.generating or self.rmbg.running or self.hi3d.refining:
            return
        if self.stage1_state is None or not self.stage1_state.is_file():
            QMessageBox.information(
                self, "UniForge3D",
                "Run a generation with Early stop enabled first; Hi3D 1536 "
                "refines its saved Shape-512 mesh.")
            self._refresh_buttons()
            return
        if self.stage1_source is None or not self.stage1_source.is_file():
            QMessageBox.warning(
                self, "UniForge3D",
                "The Early stop GLB is missing; run that generation again.")
            return
        if not self.stage1_images:
            QMessageBox.warning(
                self, "UniForge3D",
                "The reference image from the Early stop run is missing; "
                "run that generation again.")
            return
        image = self._hi3d_condition_image()
        if image is None or not image.is_file():
            QMessageBox.warning(
                self, "UniForge3D",
                f"Reference image not found:\n{image}")
            return
        missing = missing_hi3d_assets(MODEL_DIR)
        if missing:
            self.log("Hi3D is not ready; missing: " + "; ".join(missing),
                     "error")
            shown = "\n".join(missing[:8])
            if len(missing) > 8:
                shown += f"\n... and {len(missing) - 8} more"
            QMessageBox.warning(
                self, "Hi3D is not ready",
                "Hi3D 1536 needs the Hi3D v1.0 source and its models.\n"
                "Run get_models.bat first.\n\nMissing:\n" + shown)
            return
        output_path = _hi3d_output_path(self.stage1_source, resolution)
        # The idle UniForge3D worker keeps its pipeline resident; release it
        # so Hi3D 1536 gets the full GPU. It reloads on the next generation.
        if self.worker.alive:
            self.worker.stop()
            self.log("UniForge3D worker unloaded to free GPU memory for "
                     "Hi3D.", "info")
        self._pending_kind = "hi3d"
        self._pending_state = None
        self._pending_images = None
        self.btn_folder.setVisible(False)
        self._refresh_buttons()
        self.hi3d.start(
            image, self.stage1_source, output_path, resolution=resolution,
            seed=(None if self.seed_spin.value() < 0
                  else self.seed_spin.value()),
            start_structure=start_structure,
            steps=self.hi3d_steps.value(),
            budget_percent=self.hi3d_budget.value(),
            # Full-GPU checkbox forces residency ON; unchecked follows the
            # global Low VRAM switch instead of overriding it.
            low_vram=(False if self.hi3d_fullgpu.isChecked() else None))
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
                passes = (f"passes {first}-{last} "
                          f"(structures "
                          f"{' > '.join(str(s) for s in structures[first - 1:last])}"
                          f"{skipped})")
        self.log(
            f"Running Hi3D ({passes}, output {resolution}): refining "
            f"{self.stage1_source.name} with {image.name} "
            f"({self.hi3d_steps.value()} steps/pass, voxel budget "
            f"{self.hi3d_budget.value()}%, "
            f"{'full GPU residency' if self.hi3d_fullgpu.isChecked() else 'global memory mode'}, "
            f"geometry only) ...", "info")

    def on_run_cellocut(self):
        """Watertight-repair the saved Shape-512 mesh with CelloCut."""
        if (self.worker.generating or self.rmbg.running
                or self.hi3d.refining or self.cellocut.running):
            return
        if self.stage1_state is None or not self.stage1_state.is_file():
            QMessageBox.information(
                self, "UniForge3D",
                "Run a generation with Early stop enabled first; CelloCut "
                "repairs its saved Shape-512 mesh.")
            self._refresh_buttons()
            return
        if self.stage1_source is None or not self.stage1_source.is_file():
            QMessageBox.warning(
                self, "UniForge3D",
                "The Early stop GLB is missing; run that generation again.")
            return
        output_path = _cellocut_output_path(self.stage1_source)
        state_path = _shape512_state_path(self.stage1_source)
        # Back up the original Shape-512 state once, before the worker
        # regenerates it from the repaired mesh.
        if state_path.is_file():
            state_backup = state_path.with_name(
                state_path.name.replace(".shape512.pt",
                                        "_original.shape512.pt"))
            if not state_backup.is_file():
                shutil.copy2(state_path, state_backup)
                self.log(f"Original Shape-512 state kept as "
                         f"{state_backup.name}.", "info")
        # The idle UniForge3D worker keeps its pipeline resident; release it
        # so the 512^3 UDF rasterization gets the full GPU. It reloads on the
        # next generation.
        if self.worker.alive:
            self.worker.stop()
            self.log("UniForge3D worker unloaded to free GPU memory for "
                     "CelloCut.", "info")
        self._pending_kind = "cellocut"
        self._pending_state = None
        self._pending_images = None
        self.btn_folder.setVisible(False)
        self._refresh_buttons()
        self.cellocut.start(self.stage1_source, output_path,
                            shape512_state=state_path)
        self.log(f"Running CelloCut watertight repair on "
                 f"{self.stage1_source.name} (512^3 UDF, graph cut, "
                 f"geometry only) ...", "info")

    def _on_cellocut_done(self, output: str, seed=None):
        self._clear_pending_job()
        repaired = Path(output)
        target = self.stage1_source
        if target is None or not target.is_file():
            self.log("The Shape-512 GLB vanished; keeping the repaired "
                     f"mesh as {repaired.name}.", "warn")
            target = repaired.with_name("prediction_512.glb")
            shutil.copy2(repaired, target)
            repaired.unlink(missing_ok=True)
        else:
            # Replace the stage-1 mesh in place; keep the original once.
            backup = _cellocut_backup_path(target)
            if not backup.is_file():
                shutil.copy2(target, backup)
                self.log(f"Original mesh kept as {backup.name}.", "info")
            shutil.copy2(repaired, target)
            repaired.unlink(missing_ok=True)
        self.log(f"CelloCut finished; {target.name} now holds the watertight "
                 "mesh.", "ok")
        self.last_output = target
        self.btn_folder.setVisible(True)
        # Show low / Show high compares the untouched original with the
        # repaired mesh now stored under the Shape-512 name.
        backup = _cellocut_backup_path(target)
        if backup.is_file():
            self.compare_low = backup
            self.compare_high = target
            self.compare_align = None
            self.log("Show low / Show high is available "
                     "(original vs watertight).", "ok")
        else:
            self.compare_low = None
            self.compare_high = None
            self.compare_align = None
        if self.viewer.load(target, preserve_camera=True,
                            preserve_material=True):
            self.log("Reloaded the watertight mesh in the 3D view "
                     "(camera kept).", "ok")
        else:
            self.log("Could not display the repaired mesh in viewer.",
                     "error")
        # Publish the repaired OBJ next to the GLB, like every other run.
        self.converter.convert(target)
        self._refresh_buttons()

    def _on_cellocut_failed(self, msg: str):
        self._clear_pending_job()
        self.log(msg, "error")
        QMessageBox.warning(self, "CelloCut repair failed", msg)
        self._refresh_buttons()

    def on_generate_texture(self):
        if self.worker.generating or self.rmbg.running or self.hi3d.refining:
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
        sources = list(self.image_paths)
        self._queue_after_cutouts(
            sources, lambda: self._start_generate_texture(sources))

    def _start_generate_texture(self, sources: list[Path]):
        sources = [p for p in sources if p in self.image_paths]
        if not sources:
            self.log("No input images left; texture generation cancelled.",
                     "warn")
            return
        # Swap in pre-made RGBA cutouts, exactly like _start_generate.
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

    def _on_low_vram_toggled(self, checked: bool):
        self.worker.set_low_vram(checked)
        self.hi3d.set_low_vram(checked)
        message = (
            "Low VRAM enabled: only the active stage stays on the GPU."
            if checked else
            "Low VRAM disabled: full-precision weights; needs substantially "
            "more GPU memory.")
        if self.worker.alive and not self.worker.generating:
            self.worker.stop()
            message += " Worker will reload on the next run."
        self.log(message, "info")

    def _on_stop_stage1_toggled(self, checked: bool):
        if checked:
            # A Shape-512-only export cannot be textured, so force the
            # geometry-only output mode while early stop is active.
            self._stop_stage1_previous_mode = self.mode_combo.currentIndex()
            self.mode_combo.setCurrentIndex(1)
            self.log("Stop at 1st stage enabled: Shape-512 mesh will be "
                     "decoded directly; Shape-1024 and texture are skipped.",
                     "info")
        else:
            self.mode_combo.setCurrentIndex(self._stop_stage1_previous_mode)
            self.log("Stop at 1st stage disabled: the full pipeline will "
                     "run.", "info")
        self._refresh_buttons()

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
        self._pending_images = None

    def on_stop(self):
        if self.cellocut.running:
            self.cellocut.stop()
            self._clear_pending_job()
            self.log("CelloCut repair stopped (worker killed); the 512 "
                     "mesh is unchanged.", "warn")
            self._refresh_buttons()
            return
        if self.hi3d.refining:
            self.hi3d.stop()
            self._clear_pending_job()
            self.log("Hi3D refinement stopped (worker killed); Hi3D reloads "
                     "on the next refinement.", "warn")
            self._refresh_buttons()
            return
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
            # Offer the low/high toggle when this folder holds a pair.
            low_path = loaded_path.with_name("prediction_512.glb")
            high_path = loaded_path.with_name("prediction.glb")
            hi3d_paths = sorted(loaded_path.parent.glob(
                "prediction_hi3d_*.glb"))
            original_path = loaded_path.with_name(
                "prediction_512_original.glb")
            pair = None
            if loaded_path.name == "prediction.glb":
                if low_path.is_file():
                    pair = (low_path, high_path)
            elif loaded_path.name.startswith("prediction_hi3d_"):
                if low_path.is_file():
                    pair = (low_path, loaded_path)
            elif loaded_path.name == "prediction_512.glb":
                if original_path.is_file():
                    # CelloCut replaced the Shape-512 mesh; compare against
                    # the untouched original.
                    pair = (original_path, loaded_path)
                elif high_path.is_file():
                    pair = (low_path, high_path)
                elif hi3d_paths:
                    pair = (low_path, hi3d_paths[-1])
            elif loaded_path.name == "prediction_512_original.glb":
                if low_path.is_file():
                    pair = (loaded_path, low_path)
            if pair is not None:
                self.compare_low, self.compare_high = pair
                self.compare_align = None
                self.log("Show low / Show high is available for this folder.",
                         "ok")
            else:
                self.compare_low = None
                self.compare_high = None
                self.compare_align = None
            self._clear_pending_job()
        self._refresh_buttons()

    def on_open_folder(self):
        """Open the folder holding the last generated model (glb/obj/...)."""
        import os
        target = (self.last_output.parent if self.last_output
                  else OUTPUT_DIR)
        if target.is_dir():
            os.startfile(str(target))

    def on_toggle_compare(self):
        """Swap low/high geometry in place, keeping camera and material."""
        if not (self.compare_low and self.compare_high):
            return
        current = self.viewer.current_path
        showing_low = (current is not None
                       and current.resolve() == self.compare_low.resolve())
        target = self.compare_high if showing_low else self.compare_low
        if not target.is_file():
            self.log(f"Comparison model missing: {target}", "warn")
            self._update_compare_button()
            return
        if self.compare_align is None:
            # Cache the coarse bounds once so both meshes toggle in place.
            self.compare_align = viewer_mod.file_bounds(self.compare_low)
        if self.viewer.load(target, preserve_camera=True,
                            preserve_material=True,
                            align_bounds=self.compare_align):
            which = "low" if target == self.compare_low else "high"
            self.log(f"Showing {target.name} ({which}) geometry.", "info")
        else:
            self.log(f"Could not show {target.name}.", "error")
        self._update_compare_button()

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
        pending_images = self._pending_images
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
        if kind == "stage1":
            # Early-stop output is the decoded Shape-512 mesh; it has no
            # Shape-1024 latent, so Generate Texture is unavailable.
            self.texture_state = None
            self.texture_source = None
            self.log(f"Shape-512 preview finished ({out.name}); 1024 upscale "
                     "and texture were skipped.", "info")
            stage1_state = _shape512_state_path(out)
            if stage1_state.is_file() and pending_images:
                self.stage1_state = stage1_state
                self.stage1_source = out
                self.stage1_images = list(pending_images)
                self.log("Run 2nd stage is available; it will upscale this "
                         "Shape-512 mesh with Shape-1024.", "ok")
            else:
                self.stage1_state = None
                self.stage1_source = None
                self.stage1_images = None
                self.log("The Shape-512 state was not saved; Run 2nd stage "
                         "is unavailable.", "warn")
        elif kind == "geometry":
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
        elif kind == "stage2":
            if pending_state is not None and pending_state.is_file():
                self.texture_state = pending_state
                self.texture_source = out
                self.log("2nd stage finished; Generate Texture is available.",
                         "ok")
            else:
                self.texture_state = None
                self.texture_source = None
                self.log("2nd stage finished, but its texture state was not "
                         "saved; Generate Texture is unavailable.", "warn")
            low = out.with_name("prediction_512.glb")
            if out.is_file() and low.is_file():
                self.compare_low = low
                self.compare_high = out
                self.compare_align = None
                self.log("Show low / Show high is available "
                         "(prediction_512.glb vs prediction.glb).", "ok")
            else:
                self.compare_low = None
                self.compare_high = None
                self.compare_align = None
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

    def _on_hi3d_done(self, output: str, seed=None):
        self._clear_pending_job()
        out = Path(output)
        self.log(f"Hi3D finished -> {output}", "ok")
        if seed is not None:
            try:
                self.last_seed = int(seed)
            except (TypeError, ValueError):
                pass
            else:
                self._update_seed_button()
        self.last_output = out
        self.btn_folder.setVisible(True)
        # The Hi3D mesh is recentered/normalized, so align it to the
        # Shape-512 bounds for the in-place low/high comparison. The saved
        # GLB keeps Hi3D's own scale; only the viewer is aligned.
        low = self.stage1_source
        if low is not None and low.is_file() and low.resolve() != out.resolve():
            self.compare_low = low
            self.compare_high = out
            self.compare_align = viewer_mod.file_bounds(low)
            align = self.compare_align
        else:
            self.compare_low = None
            self.compare_high = None
            self.compare_align = None
            align = None
        self.log("Hi3D recenters the result and normalizes it to unit "
                 "radius; absolute scale is not preserved.", "info")
        if self.viewer.load(output, align_bounds=align):
            self.log(f"Loaded Hi3D result into viewer: {out.name}", "ok")
            if self.compare_low is not None:
                self.log("Show low / Show high is available "
                         f"(Shape-512 vs {out.name}).", "ok")
        else:
            self.log("Could not display the Hi3D result in viewer.", "error")
        # Published next to the GLB, like every other completed run.
        self.converter.convert(out)
        self._refresh_buttons()

    def _on_hi3d_failed(self, msg: str):
        self._clear_pending_job()
        self.log(msg, "error")
        QMessageBox.warning(self, "Hi3D refinement failed", msg)
        self._refresh_buttons()

    # -- misc -----------------------------------------------------------------
    def log(self, msg: str, level: str = "info"):
        self.log_box.appendPlainText(msg)

    def closeEvent(self, event):
        self.rmbg.shutdown()
        self.hi3d.shutdown()
        self.cellocut.shutdown()
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
