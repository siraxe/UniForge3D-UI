"""Diagnostic trace for the move-brush cursor jump (kept, off by default).

Writes one line per mouse/brush event to ``brush_debug.log`` in the repo
root when enabled.  To turn it back on, either flip ``ENABLED`` below to
``True`` or launch with ``UNIFORGE_TRACE=1`` in the environment.  All call
sites stay in place, so enabling needs no code changes.
"""

from __future__ import annotations

import atexit
import os
import time
from pathlib import Path

#: Master switch (kept ``False``: logging disabled, call sites retained).
ENABLED = False

_LOG_PATH = Path(__file__).resolve().parent.parent / "brush_debug.log"

_t0 = time.perf_counter()
_handle = None

_enabled = ENABLED or os.environ.get(
    "UNIFORGE_TRACE", "").strip().lower() in ("1", "true", "yes", "on")
if _enabled:
    try:
        _handle = _LOG_PATH.open("a", encoding="utf-8", buffering=1)
        _handle.write("\n" + "=" * 78 + "\n")
        _handle.write(
            f"=== session start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    except Exception:  # noqa: BLE001 - tracing must never break the app
        _handle = None


def _close():
    try:
        if _handle is not None:
            _handle.write("=== session end ===\n")
            _handle.close()
    except Exception:  # noqa: BLE001
        pass


atexit.register(_close)


def trace(tag: str, **fields) -> None:
    """Append one diagnostic line; never raises."""
    if _handle is None:
        return
    try:
        parts = []
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        stamp = time.perf_counter() - _t0
        _handle.write(f"[{stamp:9.3f}] {tag:22s} " + " ".join(parts) + "\n")
    except Exception:  # noqa: BLE001
        pass


def log_path() -> str:
    return str(_LOG_PATH)


def qt(value):
    """Qt6 enum/flags -> plain int for logging; pass-through otherwise."""
    return getattr(value, "value", value)


def vec3(value):
    """Numpy/sequence point -> rounded tuple for logs; None stays None."""
    try:
        return None if value is None else tuple(round(float(v), 3)
                                                for v in value)
    except Exception:  # noqa: BLE001
        return value


def cam_info(cam) -> dict:
    """Camera position/focal/up fields for a trace call; {} on failure."""
    try:
        return {
            "campos": tuple(round(v, 3) for v in cam.GetPosition()),
            "camfoc": tuple(round(v, 3) for v in cam.GetFocalPoint()),
            "camup": tuple(round(v, 3) for v in cam.GetViewUp()),
        }
    except Exception:  # noqa: BLE001
        return {}
