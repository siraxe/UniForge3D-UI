"""Resolve model locations for the PS6 UI and its worker processes.

Models use the ``models/<owner>/<repo>/...`` layout written by
``get_models.bat``. ``start_ps6.bat`` points ``UNIFORGE_MODEL_DIR`` at this
project's ``models/xibushu/UniForge3D`` directory.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = ROOT / "models"
CORE_MODEL_DIR = MODELS_ROOT / "xibushu" / "UniForge3D"

DINO_NAME = "dinov3-vitl16-pretrain-lvd1689m"
RMBG_NAME = "RMBG-2.0"

_EXTERNAL_RELATIVE = {
    DINO_NAME: (Path("facebook") / DINO_NAME, Path(DINO_NAME)),
    RMBG_NAME: (Path("briaai") / RMBG_NAME, Path(RMBG_NAME)),
}


def find_model_dir() -> Path:
    """Return the directory that holds ``pipeline.json``.

    ``UNIFORGE_MODEL_DIR`` overrides detection; ``start_ps6.bat`` sets it for
    the UI, and it can be set manually to share a model tree between copies.
    """
    override = os.environ.get("UNIFORGE_MODEL_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return CORE_MODEL_DIR


def _search_roots(model_dir: Path | None) -> list[Path]:
    roots: list[Path] = []
    for base in (model_dir, MODELS_ROOT, ROOT):
        if base is None:
            continue
        resolved = base.expanduser().resolve()
        if resolved not in roots:
            roots.append(resolved)
    if model_dir is not None:
        for parent in model_dir.expanduser().resolve().parents:
            if parent not in roots:
                roots.append(parent)
    return roots


def find_external_dir(name: str, model_dir: Path | None = None) -> Path | None:
    """Locate a local DINOv3 or RMBG-2.0 folder in the models tree."""
    relatives = _EXTERNAL_RELATIVE.get(name)
    if relatives is None:
        raise ValueError(f"unknown external model: {name}")
    for root in _search_roots(model_dir):
        for relative in relatives:
            candidate = root / relative
            if (candidate / "config.json").is_file():
                return candidate
    return None


def find_dino_dir(model_dir: Path | None = None) -> Path | None:
    return find_external_dir(DINO_NAME, model_dir)


def find_rmbg_dir(model_dir: Path | None = None) -> Path | None:
    return find_external_dir(RMBG_NAME, model_dir)


def hf_cache_dir() -> Path:
    """Project-local cache used only if a missing file must be downloaded."""
    return ROOT / ".cache" / "hf_cache"
