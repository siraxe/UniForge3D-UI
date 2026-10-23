"""Standalone Hi3D v1.0 refinement worker.

Spawned by PS6_UI/main.py for **Run Hi3D 1536**. It loads the Hi3D refinement
pipeline, refines the saved 1st-stage mesh at 1536, exports a geometry-only
GLB, then exits. Keeping this in a one-shot process means the Hi3D transformers
and VAE are released as soon as the job finishes, instead of competing with the
UniForge3D worker for VRAM.

The protocol is the same JSON-lines format used by worker.py:

  stdin:  {"cmd": "refine", "image": ..., "coarse_mesh": ..., "output": ...,
           "resolution": 1536, "steps": 25, "seed": 0,
           "start_structure": 192}   # optional; omit for the full ladder
  stdout: {"type": "ready"}
          {"type": "log", "msg": ..., "level": "info"|"ok"|"warn"|"error"}
          {"type": "done", "output": "...", "seed": 42, "resolution": 1536}
          {"type": "error", "msg": "..."}

Without ``start_structure`` the full Hi3D ladder runs (structures 64, 128,
192). With ``start_structure`` 192, the worker voxelizes the supplied mesh at
that rung and runs only the final pass: passes 1-2 are skipped and the source
mesh is used directly. A CUDA OOM triggers one retry with reduced token
budgets for the later passes.

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
    start_structure = cmd.get("start_structure")

    if not image_path.is_file():
        raise FileNotFoundError(f"reference image not found: {image_path}")
    if not coarse_path.is_file():
        raise FileNotFoundError(f"1st-stage mesh not found: {coarse_path}")

    if start_structure is not None:
        from sparc3d.pipelines.sparc3d_pipeline import (
            PASS_BUDGETS,
            STRUCTURE_RESOLUTIONS,
        )
        from sparc3d.ops import adaptive_vox
        from sparc3d.sparcubes.postprocess import validate_mesh

        start_structure = int(start_structure)
        if start_structure not in STRUCTURE_RESOLUTIONS:
            raise ValueError(
                f"start_structure must be one of {STRUCTURE_RESOLUTIONS}, "
                f"got {start_structure}")
        start_index = STRUCTURE_RESOLUTIONS.index(start_structure)
        mode_label = (
            f"pass {start_index + 1}/{len(STRUCTURE_RESOLUTIONS)} only, "
            f"structure {start_structure}")
        if start_index > 0:
            mode_label += " (earlier passes skipped)"
    else:
        mode_label = "full ladder (structures 64, 128, 192)"

    emit({"type": "log", "msg": (
        f"Hi3D settings: {mode_label}, resolution={resolution}, "
        f"steps={num_steps}, seed={seed}")})
    emit({"type": "log", "msg": (
        f"Refining {coarse_path.name} with {image_path.name} "
        f"(image-conditioned, geometry only) ...")})

    with Image.open(image_path) as image:
        image = image.copy()
    coarse_mesh = trimesh.load(str(coarse_path), force="mesh", process=False)

    if start_structure is None:
        result = pipeline(
            image,
            coarse_mesh,
            resolution=resolution,
            num_inference_steps=num_steps,
            seed=seed,
        )
    else:
        # Start the ladder at the requested rung using this mesh directly: the
        # first two passes never run, so their refined geometry is not involved.
        if start_index > 0 and hasattr(pipeline, "transformer_512"):
            # The 64-structure transformer is unused when the ladder starts
            # later. Release it so its ~3.5 GB cannot sit in VRAM next to the
            # memory-intensive 1536 decode (full-GPU mode especially).
            import gc

            import torch

            del pipeline.transformer_512
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            emit({"type": "log", "msg": (
                "Released the 64-structure transformer (not used by "
                "pass 3).")})
        mesh = validate_mesh(coarse_mesh)
        coords = adaptive_vox(
            mesh,
            resolution=start_structure,
            target_num=PASS_BUDGETS[start_index],
            device=pipeline.device,
        )
        skipped = (f"Skipping passes 1-{start_index}."
                   if start_index > 0 else "No earlier passes to skip.")
        emit({"type": "log", "msg": (
            f"Voxelized the source mesh at structure {start_structure}: "
            f"{coords.shape[0]:,} occupied voxels. {skipped}")})
        result = pipeline.refine_voxels(
            image,
            coords,
            structure_resolution=start_structure,
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
                f"Hi3D structure {event.structure_resolution}: step "
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
            try:
                _run_refine(pipeline, cmd)
            except RuntimeError as ex:
                if "out of memory" not in str(ex).lower():
                    raise
                # Last-resort fallback for smaller cards: retry once with fewer
                # occupied voxels in the later passes. The decode's feature count
                # scales with those budgets, so this is the one safe knob left
                # after the decoder's chunking/in-place optimizations.
                import sparc3d.pipelines.sparc3d_pipeline as pipeline_module

                emit({"type": "log", "msg": (
                    "CUDA out of memory during Hi3D refinement; retrying once "
                    "with reduced occupied-voxel budgets for the later "
                    "passes."), "level": "warn"})
                torch.cuda.empty_cache()
                budgets = pipeline_module.PASS_BUDGETS
                pipeline_module.PASS_BUDGETS = tuple(
                    budget if index == 0 else max(1, int(budget * 0.65))
                    for index, budget in enumerate(budgets)
                )
                emit({"type": "log", "msg": (
                    f"Retry budgets: {pipeline_module.PASS_BUDGETS[1:]} "
                    "(was " + ", ".join(str(b) for b in budgets[1:]) + ")")})
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

