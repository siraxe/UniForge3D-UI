import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "submodules"))

from uniforge3d import UniForge3DImageTo3DPipeline
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_MODEL = "xibushu/UniForge3D"
LOCAL_CONFIG = ROOT / "models" / "xibushu" / "UniForge3D" / "pipeline.json"


def _default_config() -> Path | None:
    override = os.environ.get("UNIFORGE_MODEL_DIR", "").strip()
    candidates = [Path(override).expanduser() / "pipeline.json"] if override else []
    candidates.append(LOCAL_CONFIG)
    return next((path for path in candidates if path.is_file()), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "assets")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--model", default=None,
        help="Hugging Face repo id or local model directory. Default: the local "
             "models/ tree when present, else xibushu/UniForge3D.")
    source.add_argument(
        "--config", type=Path,
        help="Pipeline JSON. Default: models/xibushu/UniForge3D/pipeline.json "
             "when present.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True,
                        help="Stage models on/off the GPU and use fp16 weights (default on).")
    parser.add_argument("--low-ram", action="store_true",
                        help="Keep only the current stage's weights in RAM (slower reloads).")
    parser.add_argument("--no-prefetch", dest="prefetch", action="store_false",
                        help="Load stage groups on demand instead of in the background.")
    parser.add_argument("--weights-dtype",
                        choices=("auto", "float16", "bfloat16", "float32"), default="auto",
                        help="Denoiser weight precision; auto follows --low-vram (fp16).")
    parser.add_argument("--vram-debug", action="store_true",
                        help="Print per-stage VRAM/peak and timing lines.")
    parser.add_argument("--texture", action=argparse.BooleanOptionalAction, default=True,
                        help="Bake a PBR texture; use --no-texture for geometry only.")
    parser.add_argument("--save-textures", action="store_true",
                        help="Also write basecolor/metallic-roughness PNGs beside each GLB.")
    args = parser.parse_args()

    cases = sorted(path for path in args.input.iterdir() if path.is_dir())
    cases = [case for case in cases if args.overwrite or not (args.output / case.name / "prediction.glb").is_file()]
    if not cases:
        return

    if args.vram_debug:
        os.environ["UNIFORGE_VRAM_DEBUG"] = "1"
    weights_dtype = None if args.weights_dtype == "auto" else args.weights_dtype
    options = dict(
        device=args.device, low_vram=args.low_vram, weights_dtype=weights_dtype,
        low_ram=args.low_ram, prefetch=args.prefetch,
    )
    config_path = args.config
    model = args.model
    if config_path is None and model is None:
        config_path = _default_config()
        if config_path is None:
            model = DEFAULT_MODEL
    pipeline = (
        UniForge3DImageTo3DPipeline.from_config(
            config_path, weights_root=config_path.parent, **options)
        if config_path else
        UniForge3DImageTo3DPipeline.from_pretrained(model, **options)
    )
    for case in cases:
        images = sorted(path for path in case.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        pipeline.run(images, args.output / case.name / "prediction.glb",
                     overwrite=args.overwrite, texture=args.texture,
                     save_textures=args.save_textures)


if __name__ == "__main__":
    main()
