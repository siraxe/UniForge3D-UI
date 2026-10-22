"""UniForge3D generation worker.

Owns the GPU pipeline in a separate process so the UI can hard-stop a run by
killing the process (a Python thread cannot be interrupted).

Started by PS6_UI/main.py with the model location pre-configured, then speaks
one JSON object per line on stdio:

  stdin:  {"cmd": "generate", "images": [...], "output": ..., "seed": ...,
           "overwrite": ..., "grid_size": ..., "texture_size": ...,
           "texture": true|false, "shape_state": ...}
          {"cmd": "texture", "images": [...], "output": ..., "state": ...,
           "seed": ..., "overwrite": ..., "texture_size": ..., "steps": ...}
  stdout: {"type": "ready"}                       pipeline loaded, awaiting jobs
          {"type": "log", "msg": ..., "level": "info"|"ok"|"warn"|"error"}
          {"type": "done", "output": "...", "seed": 42}
          {"type": "error", "msg": "..."}

A null/missing "seed" means "random": the worker picks one, reports it in the
done message, and the UI can reuse it with the Use last button.

Pipeline chatter (prints, tqdm, warnings) is redirected to stderr so it can
never corrupt the protocol; it shows up in the terminal that launched the UI.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Keep Hugging Face downloads inside the project, never inside models/.
# This must run before any huggingface_hub import (uniforge3d imports it).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_paths import (  # noqa: E402
    find_external_dir,
    find_model_dir,
    hf_cache_dir,
)


def _model_dir_arg() -> Path | None:
    """Read --model-dir before argparse/HF imports run."""
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        if arg == "--model-dir" and index + 1 < len(args):
            return Path(args[index + 1]).expanduser()
        if arg.startswith("--model-dir="):
            return Path(arg.split("=", 1)[1]).expanduser()
    return None


MODEL_HINT = _model_dir_arg() or find_model_dir()
HF_CACHE = hf_cache_dir()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_CACHE"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE)


# huggingface_hub.hf_file_system installs signal handlers at import time,
# which only works on the main thread; import it eagerly for safety.
import huggingface_hub.hf_file_system  # noqa: F401, E402

from uniforge3d import UniForge3DImageTo3DPipeline  # noqa: E402

# External (gated HF) models -> look for a local copy in the models/ tree first
# (e.g. downloaded by get_models.bat or from modelscope.cn), then the Hub.
# Maps model folder name -> (json sections to patch).
# Note: rembg is read by the pipeline from image_preprocess.rembg_model,
# DINO from external_models.dino (load_model_group in uniforge3d/models/loaders.py).
LOCAL_EXTERNAL_MODELS = {
    "dinov3-vitl16-pretrain-lvd1689m": [("external_models", "dino")],
    "RMBG-2.0": [("image_preprocess", "rembg_model"),
                 ("external_models", "rembg")],
}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def load_pipeline(model_root: Path, *, low_vram: bool, offline: bool):
    """Load the pipeline, preferring local copies of external HF models."""
    config_path = model_root / "pipeline.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} not found. Run get_models.bat to create the models/ "
            "tree, or set UNIFORGE_MODEL_DIR to an existing model directory."
        )
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    patched_any = False
    for folder, sections in LOCAL_EXTERNAL_MODELS.items():
        local = find_external_dir(folder, model_root)
        if local is None:
            continue
        for section, key in sections:
            node = cfg.get(section)
            if node is None:
                cfg[section] = node = {}
            if node.get(key) != str(local):
                node[key] = str(local)
                patched_any = True
    weights_dtype = os.environ.get("UNIFORGE_WEIGHTS_DTYPE", "").strip() or None
    if weights_dtype in ("auto",):
        weights_dtype = None
    options = dict(
        low_vram=low_vram, local_files_only=offline, weights_dtype=weights_dtype,
        low_ram=_env_flag("UNIFORGE_LOW_RAM"),
        prefetch=not _env_flag("UNIFORGE_NO_PREFETCH"),
    )
    if patched_any:
        tmp_cfg = model_root / "_pipeline.local.json"
        tmp_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        try:
            return UniForge3DImageTo3DPipeline.from_config(
                tmp_cfg, weights_root=model_root, **options)
        finally:
            tmp_cfg.unlink(missing_ok=True)
    return UniForge3DImageTo3DPipeline.from_pretrained(model_root, **options)


# ---------------------------------------------------------------------------
# Protocol: JSON on the real stdout, everything else on stderr.
# ---------------------------------------------------------------------------
_real_stdout = sys.stdout
sys.stdout = sys.stderr


def emit(obj) -> None:
    _real_stdout.write(json.dumps(obj) + "\n")
    _real_stdout.flush()


def _shape_state_path(output: str | Path) -> Path:
    output = Path(output)
    return output.with_name(output.stem + ".shape_state.pt")


def _resolve_seed(cmd: dict) -> int:
    """Explicit seed, or a fresh random one for the UI's 'random' option."""
    seed = cmd.get("seed")
    return random.randrange(2**31) if seed is None else int(seed)


def _run_job(pipeline, cmd: dict) -> None:
    grid_size = int(cmd["grid_size"])
    texture_size = int(cmd["texture_size"])
    steps = int(cmd.get("steps") or 0)
    texture = bool(cmd.get("texture", True))
    seed = _resolve_seed(cmd)
    pipeline.config["mesh_postprocess"]["grid_size"] = grid_size
    pipeline.config["texture"]["texture_size"] = texture_size
    if steps:
        # Overrides the default 20 denoising steps of the shape/texture
        # samplers; the 12-step sparse-structure stage is left untouched.
        for stage in ("structured_latent_512", "structured_latent_1024", "texture"):
            pipeline.config[stage]["sampler"]["params"]["steps"] = steps
    settings = (f"geometry output grid={grid_size}, steps={steps or 'default'}, "
                f"seed={seed}")
    if texture:
        emit({"type": "log", "msg": f"Settings: {settings}, "
                                    f"texture size={texture_size}"})
        emit({"type": "log", "msg": "Running pipeline: image -> structure -> "
                                    "shape -> texture ..."})
        state_path = None
    else:
        emit({"type": "log", "msg": f"Settings: {settings}, no texture"})
        emit({"type": "log", "msg": "Running pipeline: image -> structure -> "
                                    "shape (mesh only) ..."})
        state_path = Path(cmd.get("shape_state") or _shape_state_path(cmd["output"]))
    result = pipeline.run(
        [Path(p) for p in cmd["images"]],
        cmd["output"],
        seed=seed,
        overwrite=bool(cmd.get("overwrite", False)),
        texture=texture,
        save_shape_state=state_path,
    )
    if state_path is not None:
        if state_path.is_file():
            emit({"type": "log", "msg": f"Geometry saved for Generate Texture "
                                        f"({state_path.name}).", "level": "ok"})
        else:
            emit({"type": "log", "msg": "Shape state was not saved; Generate "
                                        "Texture is unavailable for this run.",
                  "level": "warn"})
    emit({"type": "done", "output": str(result), "seed": seed})


def _run_texture_job(pipeline, cmd: dict) -> None:
    texture_size = int(cmd["texture_size"])
    steps = int(cmd.get("steps") or 0)
    seed = _resolve_seed(cmd)
    pipeline.config["texture"]["texture_size"] = texture_size
    if steps:
        pipeline.config["texture"]["sampler"]["params"]["steps"] = steps
    emit({"type": "log", "msg": f"Texture settings: size={texture_size}, "
                                f"steps={steps or 'default'}, seed={seed}"})
    emit({"type": "log", "msg": "Running pipeline: texture only "
                                "(reusing saved geometry) ..."})
    result = pipeline.run_texture(
        [Path(p) for p in cmd["images"]],
        cmd["output"],
        cmd["state"],
        seed=seed,
        overwrite=bool(cmd.get("overwrite", False)),
        save_textures=bool(cmd.get("save_textures", False)),
    )
    emit({"type": "done", "output": str(result), "seed": seed})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    emit({"type": "log", "msg": f"Loading model from {args.model_dir} ..."
                                + ("" if args.offline else
                                   " (downloading missing external models on first run)")})
    try:
        pipeline = load_pipeline(model_dir, low_vram=args.low_vram,
                                 offline=args.offline)
    except Exception as ex:
        emit({"type": "error", "msg": f"Model load failed: {ex}"})
        return 1
    emit({"type": "log", "msg": "Model loader ready; stage weights load as each stage runs.",
          "level": "ok"})
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
        kind = cmd.get("cmd")
        if kind not in ("generate", "texture"):
            emit({"type": "error", "msg": f"Unknown command: {kind!r}"})
            continue
        try:
            if kind == "texture":
                _run_texture_job(pipeline, cmd)
            else:
                _run_job(pipeline, cmd)
        except Exception as ex:  # keep the loaded model for a retry
            message = f"Generation failed: {ex}"
            if ("out of memory" in message.lower()
                    and "lower Geometry" not in message):
                message += (" Close other GPU applications, or lower Geometry output "
                            "grid / Texture size and try again.")
            emit({"type": "error", "msg": message})

    return 0


if __name__ == "__main__":
    sys.exit(main())
