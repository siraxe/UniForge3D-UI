"""Standalone Hi3D v1.0 refinement worker.

Spawned by PS6_UI/main.py for **Run Hi3D 1536**. It loads the Hi3D refinement
pipeline, refines the saved 1st-stage mesh at 1536, exports a geometry-only
GLB, then exits. Keeping this in a one-shot process means the Hi3D transformers
and VAE are released as soon as the job finishes, instead of competing with the
UniForge3D worker for VRAM.

The protocol is the same JSON-lines format used by worker.py:

  stdin:  {"cmd": "refine", "image": ..., "coarse_mesh": ..., "output": ...,
           "resolution": 1536, "steps": 25, "seed": 0}
  stdout: {"type": "ready"}
          {"type": "log", "msg": ..., "level": "info"|"ok"|"warn"|"error"}
          {"type": "done", "output": "...", "seed": 42, "resolution": 1536}
          {"type": "error", "msg": "..."}

Hi3D is geometry-only. Pass a reference image with its alpha preserved so
``prepare_image`` can frame the subject the way the conditioner expects.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_paths import (  # noqa: E402
    find_clip_dir,
    find_dinov2_dir,
    find_hi3d_dir,
    find_hi3d_source,
    find_model_dir,
    hf_cache_dir,
    missing_hi3d_assets,
)

MODEL_DIR = find_model_dir()
HF_CACHE = hf_cache_dir()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE))

_real_stdout = sys.stdout


def emit(obj) -> None:
    """Write one protocol line; if the UI disappeared (broken pipe), quit."""
    try:
        _real_stdout.write(json.dumps(obj) + "\n")
        _real_stdout.flush()
    except OSError:
        os._exit(1)


def _resolve_seed(cmd: dict) -> int:
    seed = cmd.get("seed")
    return random.randrange(2**31) if seed is None else int(seed)


def _run_refine(pipeline, cmd: dict) -> None:
    import trimesh
    from PIL import Image

    image_path = Path(cmd["image"])
    coarse_path = Path(cmd["coarse_mesh"])
    output_path = Path(cmd["output"])
    resolution = int(cmd.get("resolution") or 1536)
    steps = int(cmd.get("steps") or 0)
    num_steps = steps if steps > 0 else 25
    seed = _resolve_seed(cmd)

    if not image_path.is_file():
        raise FileNotFoundError(f"reference image not found: {image_path}")
    if not coarse_path.is_file():
        raise FileNotFoundError(f"1st-stage mesh not found: {coarse_path}")

    emit({"type": "log", "msg": (
        f"Hi3D settings: resolution={resolution}, steps={num_steps}, "
        f"seed={seed}")})
    emit({"type": "log", "msg": (
        f"Refining {coarse_path.name} with {image_path.name} "
        f"(image-conditioned, geometry only) ...")})

    with Image.open(image_path) as image:
        image = image.copy()
    coarse_mesh = trimesh.load(str(coarse_path), force="mesh", process=False)

    result = pipeline(
        image,
        coarse_mesh,
        resolution=resolution,
        num_inference_steps=num_steps,
        seed=seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.mesh.export(str(output_path))

    emit({"type": "log", "msg": (
        f"Hi3D refinement finished at {result.resolution}; "
        f"{len(result.mesh.faces):,} faces, {len(result.mesh.vertices):,} "
        f"vertices."), "level": "ok"})
    emit({"type": "log", "msg": f"Saved {output_path}", "level": "ok"})
    emit({"type": "done", "output": str(output_path), "seed": seed,
          "resolution": int(result.resolution)})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None,
                        help="UniForge model directory; its parent roots are "
                             "also searched for Hi3D models.")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="ON: sequential CPU offload for the Hi3D modules.")
    args = parser.parse_args()

    # Heavy imports and all pipeline chatter go to stderr so they cannot
    # corrupt the JSON protocol on stdout.
    sys.stdout = sys.stderr

    source = find_hi3d_source()
    if source is None:
        emit({"type": "error", "msg": (
            "Hi3D v1.0 source checkout was not found. Expected "
            f"{ROOT / 'Hi3Dv1.0'} (or set UNIFORGE_HI3D_SOURCE).")})
        return 1
    sys.path.insert(0, str(source))

    model_hint = Path(args.model_dir) if args.model_dir else MODEL_DIR
    hi3d_dir = find_hi3d_dir(model_hint)
    dino_dir = find_dinov2_dir(model_hint)
    clip_dir = find_clip_dir(model_hint)
    missing = missing_hi3d_assets(model_hint)
    if hi3d_dir is None or dino_dir is None or clip_dir is None or missing:
        emit({"type": "error", "msg": (
            "Hi3D models are not ready. Run get_models.bat first. Missing: "
            + "; ".join(missing))})
        return 1

    try:
        import torch
        from sparc3d import Sparc3DPipeline
    except Exception as ex:  # noqa: BLE001 - report import/kernel problems cleanly
        emit({"type": "error", "msg": f"Hi3D import failed: {ex}"})
        return 1

    if not torch.cuda.is_available():
        emit({"type": "error", "msg": "Hi3D requires an NVIDIA CUDA GPU."})
        return 1

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)
    emit({"type": "log", "msg": (
        f"Hi3D GPU: {props.name} ({total_gb:.1f} GB), "
        f"compute capability {props.major}.{props.minor}.")})
    if total_gb < 40:
        emit({"type": "log", "msg": (
            "Hi3D 1536 was validated on a 96 GB GPU. Low VRAM/offload is "
            "enabled by default, but this run may still be memory-limited."),
            "level": "warn"})

    mode = ("sequential CPU offload" if args.low_vram
            else "full GPU residency")
    emit({"type": "log", "msg": f"Loading Hi3D v1.0 from {hi3d_dir} "
                                f"({mode}) ..."})
    try:
        def progress(event):
            emit({"type": "log", "msg": (
                f"Hi3D pass {event.pass_index + 1}/{event.num_passes} "
                f"(structure {event.structure_resolution}): step "
                f"{event.step}/{event.num_inference_steps}")})

        pipeline = Sparc3DPipeline.from_pretrained(
            str(hi3d_dir),
            cache_dir=str(HF_CACHE),
            local_files_only=True,
            dino_model_id=str(dino_dir),
            clip_model_id=str(clip_dir),
            progress_callback=progress,
        )
        if args.low_vram:
            pipeline.enable_sequential_cpu_offload("cuda")
        else:
            pipeline.to("cuda")
    except Exception as ex:  # noqa: BLE001
        emit({"type": "error", "msg": f"Hi3D model load failed: {ex}"})
        return 1

    emit({"type": "log", "msg": "Hi3D model ready.", "level": "ok"})
    emit({"type": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            emit({"type": "error", "msg": f"Bad command: {line[:200]}"})
            continue
        if cmd.get("cmd") != "refine":
            emit({"type": "error",
                  "msg": f"Unknown command: {cmd.get('cmd')!r}"})
            continue
        try:
            _run_refine(pipeline, cmd)
        except Exception as ex:  # noqa: BLE001 - keep the process exit clean
            message = f"Hi3D refinement failed: {ex}"
            if "out of memory" in message.lower():
                message += (" Close other GPU applications or enable Low VRAM. "
                            "Hi3D 1536 is memory-intensive on 32 GB GPUs.")
            emit({"type": "error", "msg": message})
            return 1
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

