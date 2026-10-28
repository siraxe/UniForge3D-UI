"""UniForge3D generation worker.

Owns the GPU pipeline in a separate process so the UI can hard-stop a run by
killing the process (a Python thread cannot be interrupted).

Started by PS6_UI/workers/controllers.py with the model location pre-configured, then speaks
one JSON object per line on stdio:

  stdin:  {"cmd": "generate", "images": [...], "output": ..., "seed": ...,
           "overwrite": ..., "grid_size": ..., "texture_size": ...,
           "texture": true|false, "shape_state": ...,
           "stop_at_stage1": true|false, "shape512_state": ...}
          {"cmd": "stage2", "images": [...], "output": ..., "shape512_state": ...,
           "shape_state": ..., "seed": ..., "overwrite": ..., "grid_size": ...,
           "texture_size": ..., "steps": ..., "texture": true|false}
          {"cmd": "texture", "images": [...], "output": ..., "state": ...,
           "seed": ..., "overwrite": ..., "texture_size": ..., "steps": ...}
  stdout: {"type": "ready"}                       pipeline loaded, awaiting jobs
          {"type": "log", "msg": ..., "level": "info"|"ok"|"warn"|"error"}
          {"type": "done", "output": "...", "seed": 42,
           "scale_back": {...}}                   (scale_back: stage2 only)
          {"type": "error", "msg": "..."}

A null/missing "seed" means "random": the worker picks one, reports it in the
done message, and the UI can reuse it with the Use last button. For "stage2",
"shape512_state" names the mesh to upscale: normally the <model>.shape512.pt
checkpoint, or the displayed GLB/OBJ/STL/PLY when no checkpoint exists.
"scale_back" carries the normalization UniForge applied to that source mesh
(scale + AABB center) when it is not already in the canonical cube, so the UI
can offer to undo it on the 1024 result.

Pipeline chatter (prints, tqdm, warnings) is redirected to stderr so it can
never corrupt the protocol; it shows up in the terminal that launched the UI.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent    # PS6_UI
ROOT = APP_DIR.parent                               # repo root
sys.path.insert(0, str(ROOT))                       # tools/
sys.path.insert(0, str(ROOT / "submodules"))        # uniforge3d source
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_paths import (
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
# Deterministic cuBLAS kernel selection (must be set before torch/CUDA
# init).  Without it cuBLAS may pick different fp16 reduction kernels for
# the same op depending on tensor addresses - and those addresses differ
# between a fresh worker and a reused one - so a re-run with the same seed
# could sample visibly different geometry.  The fixed workspace keeps
# kernel choice (and thus bitwise results) stable across runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


import huggingface_hub.hf_file_system

from uniforge3d import UniForge3DImageTo3DPipeline
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


def _shape512_state_path(output: str | Path) -> Path:
    output = Path(output)
    return output.with_name(output.stem + ".shape512.pt")


def _resolve_seed(cmd: dict) -> int:
    """Explicit seed, or a fresh random one for the UI's 'random' option."""
    seed = cmd.get("seed")
    return random.randrange(2**31) if seed is None else int(seed)


def _run_job(pipeline, cmd: dict) -> None:
    grid_size = int(cmd["grid_size"])
    texture_size = int(cmd["texture_size"])
    steps = int(cmd.get("steps") or 0)
    texture = bool(cmd.get("texture", True))
    stop_at_stage1 = bool(cmd.get("stop_at_stage1", False))
    seed = _resolve_seed(cmd)
    # Prefetch only what this job needs; a mesh-only run never pulls the
    # texture stage's weights into RAM while the GPU samples.
    if stop_at_stage1:
        pipeline.plan_prefetch(
            "preprocess", "structure", "conditioner", "shape512")
    elif texture:
        pipeline.plan_prefetch(
            "preprocess", "structure", "conditioner", "shape512",
            "shape1024", "shape1024_decode", "texture")
    else:
        pipeline.plan_prefetch(
            "preprocess", "structure", "conditioner", "shape512",
            "shape1024", "shape1024_decode")
    pipeline.config["mesh_postprocess"]["grid_size"] = grid_size
    pipeline.config["texture"]["texture_size"] = texture_size
    if steps:
        # Overrides the default 20 denoising steps of the shape/texture
        # samplers; the 12-step sparse-structure stage is left untouched.
        for stage in ("structured_latent_512", "structured_latent_1024", "texture"):
            pipeline.config[stage]["sampler"]["params"]["steps"] = steps
    settings = (f"geometry output grid={grid_size}, steps={steps or 'default'}, "
                f"seed={seed}")
    stage1_state_path: Path | None = None
    if stop_at_stage1:
        # The Shape-512 mesh is exported directly; no Shape-1024 latent (and
        # thus no texture state) exists for this run. The sidecar lets Run 2nd
        # stage resume the 1024 upscale without redoing stages 1.
        texture = False
        state_path = None
        stage1_state_path = Path(cmd.get("shape512_state")
                                 or _shape512_state_path(cmd["output"]))
        emit({"type": "log", "msg": f"Settings: {settings}, stop after "
                                    "Shape-512 (no 1024 upscale, no texture)"})
        emit({"type": "log", "msg": "Running pipeline: image -> structure -> "
                                    "shape512 -> decode (stop) ..."})
    elif texture:
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
        stop_after_shape512=stop_at_stage1,
        save_shape512_state=stage1_state_path,
    )
    if stage1_state_path is not None:
        if stage1_state_path.is_file():
            emit({"type": "log", "msg": f"Shape-512 state saved; Run 2nd "
                                        f"stage is available "
                                        f"({stage1_state_path.name}).",
                  "level": "ok"})
        else:
            emit({"type": "log", "msg": "Shape-512 state was not saved; Run "
                                        "2nd stage is unavailable for this run.",
                  "level": "warn"})
    if state_path is not None:
        if state_path.is_file():
            emit({"type": "log", "msg": f"Geometry saved for Generate Texture "
                                        f"({state_path.name}).", "level": "ok"})
        else:
            emit({"type": "log", "msg": "Shape state was not saved; Generate "
                                        "Texture is unavailable for this run.",
                  "level": "warn"})
    emit({"type": "done", "output": str(result), "seed": seed})


def _run_stage2_job(pipeline, cmd: dict) -> None:
    grid_size = int(cmd["grid_size"])
    texture_size = int(cmd["texture_size"])
    steps = int(cmd.get("steps") or 0)
    texture = bool(cmd.get("texture", False))
    seed = _resolve_seed(cmd)
    pipeline.config["mesh_postprocess"]["grid_size"] = grid_size
    pipeline.config["texture"]["texture_size"] = texture_size
    if steps:
        for stage in ("structured_latent_1024", "texture"):
            pipeline.config[stage]["sampler"]["params"]["steps"] = steps
    shape_source = Path(cmd["shape512_state"])
    if not shape_source.is_file():
        raise FileNotFoundError(
            f"Shape-512 mesh not found: {shape_source}. Load or generate a "
            "mesh first.")
    # Geometry-only 2nd stage: the texture flow model/decoder are never built.
    pipeline.plan_prefetch(
        "preprocess", "conditioner", "shape1024", "shape1024_decode",
        *(("texture",) if texture else ()))
    state_path = Path(cmd["shape_state"]) if cmd.get("shape_state") else None
    emit({"type": "log",
          "msg": f"2nd stage settings: geometry output grid={grid_size}, "
                 f"steps={steps or 'default'}, seed={seed}"
                 + (f", texture size={texture_size}" if texture else "")})
    emit({"type": "log", "msg": f"Running pipeline: {shape_source.name} -> "
                                "shape1024 -> "
                                + ("texture ..." if texture else "mesh ...")})
    result = pipeline.run_stage2(
        [Path(p) for p in cmd["images"]],
        cmd["output"],
        shape_source,
        seed=seed,
        overwrite=bool(cmd.get("overwrite", False)),
        texture=texture,
        save_shape_state=state_path,
    )
    scale_back = None
    normalization = getattr(pipeline, "last_stage2_normalization", None)
    if normalization:
        try:
            # Hot-reloadable tool; doing the detection here keeps the UI free
            # of the numpy/tool import and uses the exact transform the
            # pipeline applied to the source mesh.
            from tools._loader import load_tool
            scale_back = load_tool("mesh_edit").scale_back_info(normalization)
        except Exception as error:  # noqa: BLE001 - optional UI convenience
            emit({"type": "log", "msg": f"Scale-back detection failed: {error}",
                  "level": "warn"})
    if state_path is not None:
        if state_path.is_file():
            emit({"type": "log", "msg": "Shape-1024 state saved; Generate "
                                        "Texture is available.", "level": "ok"})
        else:
            emit({"type": "log", "msg": "Shape-1024 state was not saved; "
                                        "Generate Texture is unavailable.",
                  "level": "warn"})
    emit({"type": "done", "output": str(result), "seed": seed,
          "scale_back": scale_back})


def _run_texture_job(pipeline, cmd: dict) -> None:
    texture_size = int(cmd["texture_size"])
    steps = int(cmd.get("steps") or 0)
    seed = _resolve_seed(cmd)
    pipeline.config["texture"]["texture_size"] = texture_size
    if steps:
        pipeline.config["texture"]["sampler"]["params"]["steps"] = steps
    # A geometry-only run before this one likely narrowed the prefetch plan;
    # re-widen it so the texture weights build as this job starts.
    pipeline.plan_prefetch(
        "preprocess", "conditioner", "shape1024_decode", "texture")
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
    mode = "low VRAM (fp16 weights/compute)" if args.low_vram else \
           "full precision (fp32 weights, bf16 compute)"
    emit({"type": "log", "msg": f"Pipeline memory mode: {mode}."})

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
        if kind not in ("generate", "texture", "stage2"):
            emit({"type": "error", "msg": f"Unknown command: {kind!r}"})
            continue
        try:
            if kind == "texture":
                _run_texture_job(pipeline, cmd)
            elif kind == "stage2":
                _run_stage2_job(pipeline, cmd)
            else:
                _run_job(pipeline, cmd)
        except Exception as ex:  # keep the loaded model for a retry
            label = {"stage2": "2nd stage",
                     "texture": "Texture generation"}.get(kind, "Generation")
            message = f"{label} failed: {ex}"
            if ("out of memory" in message.lower()
                    and "lower Geometry" not in message):
                message += (" Close other GPU applications, or lower Geometry output "
                            "grid / Texture size and try again.")
            emit({"type": "error", "msg": message})

    return 0


if __name__ == "__main__":
    sys.exit(main())
