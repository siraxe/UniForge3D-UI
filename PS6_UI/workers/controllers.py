"""Background worker controllers: generation, Hi3D, LATO.2, CelloCut,
RMBG, GLB->OBJ conversion."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from paths import (
    ROOT,
    WORKER,
    RMBG_WORKER,
    HI3D_WORKER,
    CELLO_WORKER,
    LATO2_WORKER,
    MODEL_DIR,
)

# ---------------------------------------------------------------------------
# Generation worker plumbing (JSON on stdio, one object per line)
# ---------------------------------------------------------------------------
class _WorkerReader(QThread):
    log_line = Signal(str, str)          # msg, level
    job_done = Signal(str, object, object)  # output path, seed, extra payload
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
                    # "scale_back" is produced by the stage2 job; other jobs
                    # leave it absent, so the payload is simply None.
                    self.job_done.emit(str(msg.get("output", "")),
                                       msg.get("seed"),
                                       msg.get("scale_back"))
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
    log_line = Signal(str, str)
    state_changed = Signal(bool)          # True while generating
    generation_done = Signal(str, object, object)  # output path, seed, extra
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

    def start_stage2(self, images, output: Path, shape_source: Path, *,
                     seed, grid_size: int, texture_size: int, steps: int,
                     texture: bool = False, shape_state: Path | None = None):
        # shape_source is the .shape512.pt checkpoint when present, or any
        # displayed mesh (GLB/OBJ/STL/PLY) to upscale.
        cmd = {
            "cmd": "stage2",
            "images": [str(p) for p in images],
            "output": str(output),
            "shape512_state": str(shape_source),
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

    def _on_done(self, output: str, seed=None, extra=None):
        self._generating = False
        self.state_changed.emit(False)
        self.generation_done.emit(output, seed, extra)

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
    log_line = Signal(str, str)
    state_changed = Signal(bool)         # True while refining
    refine_done = Signal(str, object, object)    # output path, seed, extras
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

    def _on_done(self, output: str, seed=None, extra=None):
        self._refining = False
        self.state_changed.emit(False)
        # ``extra`` carries the worker's scale_back payload (Hi3D output is
        # unit-radius); forward it so the UI can restore the source size.
        self.refine_done.emit(output, seed, extra)

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
# LATO.2 mesh-generation worker plumbing (one-shot process per run)
# ---------------------------------------------------------------------------
class Lato2Controller(QObject):
    log_line = Signal(str, str)
    state_changed = Signal(bool)         # True while generating
    lato2_done = Signal(str, object)     # output path, seed used
    lato2_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: subprocess.Popen | None = None
        self._reader: _WorkerReader | None = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self, mesh: Path, output: Path, *, vert_num: int = 2000,
              seed=None, vflow_steps: int = 24, tflow_steps: int = 50,
              cfg_strength: float = 3.0, fill_quad_rings: bool = False,
              drop_tiny_patches: bool = True,
              shape512_state: Path | None = None):
        if self._running:
            return
        # A finished one-shot worker may still be shutting down; reap it
        # before spawning the next run.
        if self._proc is not None and self._proc.poll() is None:
            self.stop()
        self._proc = subprocess.Popen(
            [sys.executable, str(LATO2_WORKER),
             "--model-dir", str(MODEL_DIR)],
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
        # The worker only reads stdin after its ~5 GB model load, so this
        # small command simply waits in the pipe buffer.
        cmd = {
            "cmd": "generate",
            "mesh": str(mesh),
            "output": str(output),
            "vert_num": int(vert_num),
            "vflow_steps": int(vflow_steps),
            "tflow_steps": int(tflow_steps),
            "cfg_strength": float(cfg_strength),
            "fill_quad_rings": bool(fill_quad_rings),
            "drop_tiny_patches": bool(drop_tiny_patches),
            "seed": seed,
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

    def _on_done(self, output: str, seed=None, extra=None):
        self._running = False
        self.state_changed.emit(False)
        self.lato2_done.emit(output, seed)

    def _on_failed(self, msg: str):
        self._running = False
        self.state_changed.emit(False)
        self.lato2_failed.emit(msg)

    def _on_dead(self, code: int):
        if self._running:
            self._running = False
            self.state_changed.emit(False)
            self.lato2_failed.emit(
                f"LATO.2 worker exited unexpectedly (code {code}).")
        self.log_line.emit(f"LATO.2 worker process exited (code {code}).",
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
        self._running = False


# ---------------------------------------------------------------------------
# CelloCut watertight-remeshing worker plumbing (one-shot process per run)
# ---------------------------------------------------------------------------
class CelloCutController(QObject):
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

    def _on_done(self, output: str, seed=None, extra=None):
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
        try:
            import trimesh

            stem = glb.stem
            obj = glb.with_suffix(".obj")
            # The OBJ is a published copy next to the GLB (it is never used
            # by the pipeline); skip the work when it is already up to date.
            if (obj.is_file() and obj.stat().st_mtime >= glb.stat().st_mtime):
                self.log_line.emit(
                    f"OBJ copy of {glb.name} is up to date "
                    f"({obj.name}); conversion skipped.", "info")
                return
            self.log_line.emit(
                f"Publishing OBJ copy of {glb.name} ({stem}.obj) ...", "info")
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
