"""Path constants and sidecar/output-path helpers for PS6_UI."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from model_paths import find_model_dir

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # for the hot-reloadable tools/ package
    sys.path.insert(0, str(ROOT))
_WORKERS = Path(__file__).resolve().parent / "workers"
WORKER = _WORKERS / "worker.py"
RMBG_WORKER = _WORKERS / "rmbg_worker.py"
HI3D_WORKER = _WORKERS / "hi3d_worker.py"
CELLO_WORKER = _WORKERS / "cello_worker.py"
LATO2_WORKER = _WORKERS / "lato2_worker.py"
MODEL_DIR = find_model_dir()
OUTPUT_DIR = ROOT / "output_dir"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _shape_state_path(glb: Path) -> Path:
    return glb.with_name(glb.stem + ".shape_state.pt")


def _shape512_state_path(glb: Path) -> Path:
    return glb.with_name(glb.stem + ".shape512.pt")


def _model_sidecars(glb: Path) -> list[Path]:
    candidates = (_shape_state_path(glb), _shape512_state_path(glb))
    return [path for path in candidates if path.is_file()]


def _paths_equal(left, right) -> bool:
    if left is None or right is None:
        return False
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return Path(left) == Path(right)


_INVALID_STEM = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_STEMS = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _validate_model_stem(stem: str) -> str | None:
    if not stem:
        return "The name cannot be empty."
    if _INVALID_STEM.search(stem):
        return 'The name cannot contain < > : " / \\ | ? * characters.'
    if stem.endswith((" ", ".")):
        return "The name cannot end with a space or a dot."
    if stem.split(".", 1)[0].upper() in _RESERVED_STEMS:
        return f"{stem!r} is a reserved Windows file name."
    return None


def _textured_output_path(glb: Path) -> Path:
    return glb.with_name(glb.stem + "_textured.glb")


def _strip_stage_suffixes(stem: str, *, cellocut: bool = True) -> str:
    stem = re.sub(r"_hi3d_\d+$", "", stem)
    stem = re.sub(r"_lato2$", "", stem)
    if cellocut and stem.endswith("_cellocut"):
        stem = stem[: -len("_cellocut")]
    return stem


def _stage2_output_path(glb: Path) -> Path:
    stem = _strip_stage_suffixes(glb.stem, cellocut=False)
    marker = ""
    if stem.endswith("_cellocut"):
        marker = "_cellocut"
        stem = stem[: -len("_cellocut")]
    if stem.endswith("_512"):
        stem = stem[: -len("_512")] + "_1024"
    else:
        stem += "_1024"
    return glb.with_name(stem + marker + ".glb")


def _hi3d_output_path(glb: Path, resolution: int) -> Path:
    stem = _strip_stage_suffixes(glb.stem, cellocut=False)
    return glb.with_name(f"{stem}_hi3d_{resolution}.glb")


def _lato2_output_path(glb: Path) -> Path:
    stem = _strip_stage_suffixes(glb.stem, cellocut=False)
    return glb.with_name(f"{stem}_lato2.glb")


def _cellocut_output_path(glb: Path) -> Path:
    stem = _strip_stage_suffixes(glb.stem)
    return glb.with_name(stem + "_cellocut.glb")


def _cellocut_source_path(glb: Path) -> Path | None:
    stem = re.sub(r"_hi3d_\d+$", "", glb.stem)
    stem = re.sub(r"(_edit|_reproject)$", "", stem)
    if not stem.endswith("_cellocut"):
        return None
    return glb.with_name(stem[: -len("_cellocut")] + ".glb")


def _edit_output_path(glb: Path) -> Path:
    stem = re.sub(r"(_edit|_reproject)$", "", glb.stem)
    return glb.with_name(stem + "_edit.glb")
