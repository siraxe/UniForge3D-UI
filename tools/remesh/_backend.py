"""Lazy optional Numba backend for :mod:`remesh`."""
from __future__ import annotations

import os

_MODULE = None
_CHECKED = False


def accelerator():
    """Return the ``remesh.accel`` module, or None when unavailable."""
    global _MODULE, _CHECKED
    if os.environ.get("REMESH_DISABLE_ACCEL", "").strip():
        return None
    if _MODULE is not None:
        return _MODULE
    if _CHECKED:
        return None
    _CHECKED = True
    try:
        from . import accel
    except Exception:
        return None
    _MODULE = accel
    return _MODULE


def backend_name() -> str:
    """Human-readable name of the path the hot stages will use."""
    return "parallel CPU (numba)" if accelerator() is not None else "Python"


__all__ = ["accelerator", "backend_name"]
