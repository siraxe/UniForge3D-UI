"""Standalone UltraTex texturing worker.

Spawned by PS6_UI/workers/controllers.py for **Run UltraTex**. It loads the
FLUX.2 Klein 9B int8-convrot model + UltraTex VAEs/LoRAs, renders the camera
rigs of the UV-unwrapped mesh, samples albedo (and optionally metallic/
roughness), bakes the texture atlases, exports a display-ready textured GLB,
then exits. One-shot like lato2_worker.py: the ~10 GB of weights are released
when the process ends.

The protocol is the same JSON-lines format used by the other workers:

  stdin:  {"cmd": "texture", "mesh": "...", "reference": "...",
           "mask": "...",                  # optional (else cutout alpha)
           "reference_views": {"front": "...", ...},  # optional view tags;
                                           # else inputs/reference_views.json
           "seed": 1, "steps": 8,
           "render_size": 1024, "bake_size": 4096,
           "use_pbr": true, "use_refine": true, "use_second_rig": true,
           "use_distill_lora": true}
  stdout: {"type": "ready"}
          {"type": "log", "msg": ..., "level": "info"|"ok"|"warn"|"error"}
          {"type": "done", "output": "...textured.glb", "seed": 42,
           "albedo": "...", "metallic": "...", "roughness": "...",
           "confidence": "...", "views_dir": "..."}
          {"type": "error", "msg": "..."}
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent    # PS6_UI
ROOT = APP_DIR.parent                               # repo root (tools/)
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from model_paths import find_model_dir, find_ultratex_dir, missing_ultratex_assets

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_DIR = find_model_dir()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_real_stdout = sys.stdout


def emit(obj) -> None:
    """Write one protocol line; if the UI disappeared (broken pipe), quit."""
    try:
        _real_stdout.write(json.dumps(obj) + "\n")
        _real_stdout.flush()
    except OSError:
        os._exit(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    args = parser.parse_args()

    model_hint = Path(args.model_dir) if args.model_dir else MODEL_DIR
    missing = missing_ultratex_assets(model_hint)
    if missing:
        emit({"type": "error", "msg": (
            "UltraTex models are not ready. Run get_models.bat first. "
            "Missing: " + "; ".join(missing))})
        return 1
    model_dir = find_ultratex_dir(model_hint)
    if model_dir is None:
        emit({"type": "error", "msg": "UltraTex model directory not found."})
        return 1

    try:
        import torch
    except Exception as ex:
        emit({"type": "error", "msg": f"UltraTex import failed: {ex}"})
        return 1
    if not torch.cuda.is_available():
        emit({"type": "error", "msg": "UltraTex requires an NVIDIA CUDA GPU."})
        return 1

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
        if cmd.get("cmd") == "bake_ws_normal":
            # No FLUX models needed: comfy-core mesh tools only.
            from tools.ultratex.engine import bake_ws_normal
            try:
                out = bake_ws_normal(
                    cmd["mesh"], cmd["donor"],
                    resolution=int(cmd.get("resolution", 4096)),
                    log=lambda msg, level="info":
                        emit({"type": "log", "msg": msg, "level": level}))
            except Exception as ex:  # noqa: BLE001
                emit({"type": "error", "msg": f"Bake WS Normal failed: {ex}"})
                return 1
            emit({"type": "done", "output": out, "seed": None,
                  "stats": {"kind": "ws_normal",
                            "resolution": int(cmd.get("resolution", 4096)),
                            "donor": cmd.get("donor")}})
            return 0
        if cmd.get("cmd") != "texture":
            emit({"type": "error",
                  "msg": f"Unknown command: {cmd.get('cmd')!r}"})
            continue

        from tools.ultratex import (UltraTexJob, load_reference_views,
                                    run as run_utex)

        # Per-view reference images: what the UI tagged for this run, else
        # the run folder's inputs/reference_views.json (filename -> view).
        reference_views = {str(v): str(p)
                           for v, p in (cmd.get("reference_views") or {}).items()}
        if not reference_views:
            inputs_dir = Path(cmd["mesh"]).parent / "inputs"
            reference_views = {str(v): str(p) for v, p
                               in load_reference_views(inputs_dir).items()}

        seed = cmd.get("seed")
        seed = random.randrange(2**31) if seed is None else int(seed)
        job = UltraTexJob(
            mesh_path=cmd["mesh"],
            render_mesh_path=cmd.get("render_mesh"),
            ws_normal_map=cmd.get("ws_normal_map"),
            reference_image=cmd["reference"],
            reference_mask=cmd.get("mask"),
            reference_views=reference_views or None,
            seed=seed,
            steps=int(cmd.get("steps", 8)),
            use_distill_lora=bool(cmd.get("use_distill_lora", True)),
            use_refine=bool(cmd.get("use_refine", True)),
            use_second_rig=bool(cmd.get("use_second_rig", True)),
            use_pbr=bool(cmd.get("use_pbr", True)),
            render_size=int(cmd.get("render_size", 1024)),
            bake_size=int(cmd.get("bake_size", 4096)),
            model_dir=str(model_dir),
        )
        try:
            job = run_utex(job, log=lambda msg, level="info":
                           emit({"type": "log", "msg": msg, "level": level}),
                           cancel=None)
        except Exception as ex:
            message = f"UltraTex texturing failed: {ex}"
            if "out of memory" in message.lower():
                message += (" Close other GPU applications or lower the "
                            "render/bake resolution.")
            emit({"type": "error", "msg": message})
            return 1
        emit({"type": "done",
              "output": job.textured_glb or job.albedo_atlas,
              "seed": job.seed,
              "albedo": job.albedo_atlas,
              "metallic": job.metallic_atlas,
              "roughness": job.roughness_atlas,
              "confidence": job.confidence_atlas,
              "views_dir": job.stats.get("views_dir")})
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
