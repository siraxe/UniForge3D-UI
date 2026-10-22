"""Resolve model locations for the PS6 UI and its worker processes.

Models use the ``models/<owner>/<repo>/...`` layout written by
``get_models.bat``. ``start_ps6.bat`` points ``UNIFORGE_MODEL_DIR`` at this
project's ``models/xibushu/UniForge3D`` directory.

Hi3D v1.0 uses the same tree:

    models/hi3d-official/Hi3Dv1.0/
    models/facebook/dinov2-large/
    models/openai/clip-vit-large-patch14/

LATO.2 stores its checkpoints and DINOv2-L/14-reg encoder under:

    models/0x4c48/LATO.2/
    models/0x4c48/LATO.2/dinov2/checkpoints/
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = ROOT / "models"
CORE_MODEL_DIR = MODELS_ROOT / "xibushu" / "UniForge3D"

DINO_NAME = "dinov3-vitl16-pretrain-lvd1689m"
RMBG_NAME = "RMBG-2.0"
HI3D_NAME = "Hi3Dv1.0"
DINOV2_NAME = "dinov2-large"
CLIP_NAME = "clip-vit-large-patch14"
LATO2_NAME = "LATO.2"

_EXTERNAL_RELATIVE = {
    DINO_NAME: (Path("facebook") / DINO_NAME, Path(DINO_NAME)),
    RMBG_NAME: (Path("briaai") / RMBG_NAME, Path(RMBG_NAME)),
    HI3D_NAME: (Path("hi3d-official") / HI3D_NAME, Path(HI3D_NAME)),
    DINOV2_NAME: (Path("facebook") / DINOV2_NAME, Path(DINOV2_NAME)),
    CLIP_NAME: (Path("openai") / CLIP_NAME, Path(CLIP_NAME)),
    LATO2_NAME: (Path("0x4c48") / LATO2_NAME, Path(LATO2_NAME)),
}

_EXTERNAL_MARKERS = {
    HI3D_NAME: "model_index.json",
    LATO2_NAME: "vflow.pt",
}

HI3D_REQUIRED_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "vae/config.json",
    "vae/model.safetensors",
    "transformer_512/config.json",
    "transformer_512/model.safetensors",
    "transformer_1024/config.json",
    "transformer_1024/model.safetensors",
)

_ENCODER_REQUIRED_FILES = ("config.json", "model.safetensors")

LATO2_REQUIRED_FILES = (
    "vflow.pt",
    "vvae.pt",
    "offset_head.pt",
    "tflow.pt",
    "tvae.pt",
    "voxel_encoder.pt",
    "dinov2/checkpoints/dinov2_vitl14_reg4_pretrain.pth",
)


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
    marker = _EXTERNAL_MARKERS.get(name, "config.json")
    for root in _search_roots(model_dir):
        for relative in relatives:
            candidate = root / relative
            if (candidate / marker).is_file():
                return candidate
    return None


def find_dino_dir(model_dir: Path | None = None) -> Path | None:
    return find_external_dir(DINO_NAME, model_dir)


def find_rmbg_dir(model_dir: Path | None = None) -> Path | None:
    return find_external_dir(RMBG_NAME, model_dir)


def find_hi3d_dir(model_dir: Path | None = None) -> Path | None:
    """Locate the local Hi3D v1.0 artifact directory."""
    return find_external_dir(HI3D_NAME, model_dir)


def find_dinov2_dir(model_dir: Path | None = None) -> Path | None:
    """Locate the local DINOv2-large encoder required by Hi3D."""
    return find_external_dir(DINOV2_NAME, model_dir)


def find_clip_dir(model_dir: Path | None = None) -> Path | None:
    """Locate the local CLIP ViT-L/14 encoder required by Hi3D."""
    return find_external_dir(CLIP_NAME, model_dir)


def find_lato2_dir(model_dir: Path | None = None) -> Path | None:
    """Locate the local LATO.2 checkpoint directory."""
    return find_external_dir(LATO2_NAME, model_dir)


def find_lato2_source() -> Path | None:
    """Locate the LATO.2 *source* checkout.

    The PS6 UI runs the source directly because LATO.2 upstream pins older
    torch/CUDA versions. ``UNIFORGE_LATO2_SOURCE`` can point elsewhere.
    """
    override = os.environ.get("UNIFORGE_LATO2_SOURCE", "").strip()
    candidates = [Path(override).expanduser()] if override else []
    candidates.append(ROOT / "LATO.2")
    for candidate in candidates:
        if (candidate / "scripts" / "e2e_inference.py").is_file():
            return candidate.resolve()
    return None


def missing_lato2_assets(model_dir: Path | None = None) -> list[str]:
    """Return human-readable missing LATO.2 source/checkpoint paths.

    Called before Run LATO.2 so an incomplete download is reported with the
    exact paths instead of failing after loading several GB of weights.
    """
    missing: list[str] = []

    if find_lato2_source() is None:
        missing.append(f"LATO.2 source checkout ({ROOT / 'LATO.2'})")

    lato2_dir = find_lato2_dir(model_dir)
    if lato2_dir is None:
        missing.append(
            "LATO.2 checkpoint directory "
            f"({MODELS_ROOT / '0x4c48' / LATO2_NAME})")
    else:
        for relative in LATO2_REQUIRED_FILES:
            if not (lato2_dir / relative).is_file():
                missing.append(str(lato2_dir / relative))

    return missing


def find_hi3d_source() -> Path | None:
    """Locate the Hi3D v1.0 *source* checkout.

    The PS6 UI runs the source directly instead of pip-installing it because
    Hi3D's package metadata pins Python 3.10 while this app runs on 3.12.
    ``UNIFORGE_HI3D_SOURCE`` can point at a checkout outside the project.
    """
    override = os.environ.get("UNIFORGE_HI3D_SOURCE", "").strip()
    candidates = [Path(override).expanduser()] if override else []
    candidates.append(ROOT / "Hi3Dv1.0")
    for candidate in candidates:
        if (candidate / "sparc3d" / "pipelines" / "sparc3d_pipeline.py").is_file():
            return candidate.resolve()
    return None


def missing_hi3d_assets(model_dir: Path | None = None) -> list[str]:
    """Return human-readable missing Hi3D source/model paths.

    The UI calls this before starting Run Hi3D 1536 so a missing download is
    reported with the exact paths instead of failing after a long model load.
    """
    missing: list[str] = []

    if find_hi3d_source() is None:
        missing.append(
            f"Hi3D v1.0 source checkout ({ROOT / 'Hi3Dv1.0'})")

    hi3d_dir = find_hi3d_dir(model_dir)
    if hi3d_dir is None:
        missing.append(
            "Hi3D v1.0 model directory "
            f"({MODELS_ROOT / 'hi3d-official' / HI3D_NAME})")
    else:
        for relative in HI3D_REQUIRED_FILES:
            if not (hi3d_dir / relative).is_file():
                missing.append(str(hi3d_dir / relative))

    for label, finder in (
        ("DINOv2-large encoder", find_dinov2_dir),
        ("CLIP ViT-L/14 encoder", find_clip_dir),
    ):
        encoder_dir = finder(model_dir)
        if encoder_dir is None:
            missing.append(f"{label} directory")
            continue
        for relative in _ENCODER_REQUIRED_FILES:
            if not (encoder_dir / relative).is_file():
                missing.append(str(encoder_dir / relative))

    return missing


def hf_cache_dir() -> Path:
    """Project-local cache used only if a missing file must be downloaded."""
    return ROOT / ".cache" / "hf_cache"
