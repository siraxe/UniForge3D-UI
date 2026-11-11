"""UltraTex texture generation core (self-contained).

Extracts the sampling/VAE/LoRA core out of ComfyUI (reference commit
61e524bdb, CORE-606) and re-implements the ultratex_setget_01.json recipe
without the ComfyUI server. The heavy lifting is done by the comfy core
package vendored under ``submodules/ComfyUI/comfy`` used as a library:

- model loading + int8-convrot kernels: comfy.sd, comfy.ops, comfy.quant_ops
  (kernels come from the ``comfy-kitchen`` pip package; needs torch cu130,
  which install.bat already pins)
- sampling: comfy.model_management, comfy.sample, comfy.samplers
- VAE: comfy.sd.VAE
- LoRA: comfy.sd.load_lora_for_models
- flux multi-view token grid + background token drop: comfy.ldm.flux.model
  (views_to_grid + the UltraTex paths added by CORE-606)

This module currently holds the recipe constants and the pipeline skeleton.
M2 fills in the implementations (see ULTRATEX_PLAN.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# -- recipe constants (from ultratex_setget_01.json) -------------------------
# All passes sample with cfg = 1.0 (UltraTex trains with an empty prompt),
# euler / simple scheduler. The albedo branch additionally runs the Klein
# distill LoRA at strength 0.5, which is what allows 4-step sampling.

DEFAULT_RENDER_SIZE = 1024      # world-normal + texture view renders
DEFAULT_BAKE_SIZE = 4096        # BakeTextureFromViews atlas resolution
DEFAULT_BAKE_PADDING = 4        # UV gutter used at bake time
DISTILL_LORA_STRENGTH = 0.5
REFINE_DENOISE_RIG1 = 0.1       # 2-step albedo refine
REFINE_DENOISE_RIG2 = 0.2       # 2-step refine of the second-rig pass
REFINE_DENOISE_MR = 0.2         # 2-step metallic/roughness refine

# Camera rigs (CreateCameraInfo 'orbit' params: elevation, roll, distance,
# yaw offsets, fov 35.49, perspective):
#   rig 1: azimuth 0, orbit 4 sides + top and bottom  -> 6 canonical views
#   rig 2: azimuth 45, orbit 4 sides, plus elev 0/45 top-bottom pair
#     (batched) -> 4 extra views re-sampling low-confidence texels
RIG1 = dict(elevation=0.0, orbit_sides=4, extras="top and bottom")
RIG2 = dict(elevation=45.0, orbit_sides=4, extras="none")
RIG2_TOP_BOTTOM = dict(elevation=0.0, orbit_sides=2, yaw=-90.0, extras="none")

SAMPLER = dict(sampler="euler", scheduler="simple", cfg=1.0)

# -- reference views ----------------------------------------------------------
# Canonical view order: matches UltraTexConditioning (comfy_extras/
# nodes_ultratex.py) and the rig-1 camera order (save_views names).
VIEW_ORDER = ("front", "right", "back", "left", "top", "bottom")
VIEWS_FILENAME = "reference_views.json"

# Closest-view fallbacks for views without a tagged image (front falls back
# to the side views first, back to top/bottom, top/bottom to each other, and
# the side views to each other).
VIEW_FALLBACKS = {
    "front":  ("left", "right", "back", "top", "bottom"),
    "right":  ("left", "front", "back", "top", "bottom"),
    "left":   ("right", "front", "back", "top", "bottom"),
    "back":   ("top", "bottom", "left", "right", "front"),
    "top":    ("bottom", "front", "back", "left", "right"),
    "bottom": ("top", "front", "back", "left", "right"),
}


def resolve_reference_views(tagged: dict) -> dict:
    """Fill every canonical view from *tagged* (view -> image path).

    Views tagged directly keep their image; the rest fall back to the
    closest tagged view per :data:`VIEW_FALLBACKS`. Views with no tagged
    image anywhere are omitted."""
    tagged = {v: p for v, p in (tagged or {}).items() if v in VIEW_ORDER}
    resolved = {}
    for view in VIEW_ORDER:
        path = tagged.get(view)
        if path is None:
            path = next((tagged[c] for c in VIEW_FALLBACKS[view]
                         if c in tagged), None)
        if path is not None:
            resolved[view] = path
    return resolved


def load_reference_views(inputs_dir) -> dict:
    """Read ``reference_views.json`` ({filename: view}) from *inputs_dir*.

    Returns ``{view: Path}`` for files that exist next to the JSON (the
    UI writes the ``<stem>_src.png`` copies it made for the run). The
    background-removed ``<stem>_cutout.png`` twin is preferred when it
    exists (its alpha channel doubles as the object mask)."""
    import json
    path = Path(inputs_dir) / VIEWS_FILENAME
    if not path.is_file():
        return {}
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(saved, dict):
        return {}
    tagged = {}
    for name, view in saved.items():
        if view not in VIEW_ORDER:
            continue
        stem = name.rsplit(".", 1)[0]
        if stem.endswith("_src"):
            stem = stem[:-4]
        img = Path(inputs_dir) / f"{stem}_cutout.png"
        if not img.is_file():
            img = Path(inputs_dir) / name
        if img.is_file():
            tagged[view] = img
    return tagged


@dataclass
class UltraTexJob:
    """One texturing job: mesh (+UVs) + reference image in, atlases out."""

    mesh_path: str                    # GLB with TEXCOORD_0 (UV accordion)
    reference_image: str              # stage-1 input image
    render_mesh_path: str | None = None  # optional geometry donor for the
                                      # normal renders (circle-marked ghost
                                      # mesh); defaults to mesh_path
    ws_normal_map: str | None = None  # prebaked world-space normal map
                                      # (<mesh>_wsnormal.png); when set the
                                      # conditioning renders show the baked
                                      # high-poly normals on the displayed
                                      # low-poly mesh (render_mesh_path is
                                      # ignored in this mode)
    reference_mask: str | None = None # RMBG cutout; RemoveBackground fallback
    reference_views: dict | None = None  # {view name: image path} tagged in
                                      # the UI; missing views fall back to
                                      # the closest tagged view (VIEW_FALLBACKS)
    seed: int = 1
    steps: int = 8                    # albedo/MR pass steps (rig 1 uses 4
                                      # when the distill LoRA is enabled)
    use_distill_lora: bool = True
    use_refine: bool = True
    use_second_rig: bool = True
    use_pbr: bool = True              # metallic/roughness LoRA pass
    render_size: int = DEFAULT_RENDER_SIZE
    bake_size: int = DEFAULT_BAKE_SIZE
    bake_padding: int = DEFAULT_BAKE_PADDING
    model_dir: str = "models/Comfy-Org/UltraTex"
    log = None                        # callable(message, level)

    # outputs (filled by run())
    albedo_atlas: str | None = None
    metallic_atlas: str | None = None
    roughness_atlas: str | None = None
    confidence_atlas: str | None = None
    textured_glb: str | None = None   # mesh + UVs + albedo atlas (display)
    stats: dict = field(default_factory=dict)


# -- pipeline skeleton (implemented over M2/M3) ------------------------------
#
# 1. load_core()
#    - import comfy core from submodules/ComfyUI (comfy.options defaults are
#      library-safe: cli args are NOT parsed)
#    - load_diffusion_model(flux-2-klein-base-9b_int8_convrot)
#    - Branch A: distill LoRA 0.5 -> albedo LoRA 1.0
#    - Branch B: metallic/roughness LoRA 1.0
#    - ultratex_flux2_vae (encode/decode), flux2-vae (refine branch encode)
# 2. conditioning()
#    - ConditioningLoader equivalent: read the empty-prompt embedding
#      safetensors directly (no text encoder on disk at all)
#    - port UltraTexConditioning (comfy_extras/nodes_ultratex.py):
#      world-normal views -> views_to_grid latent, reference latents,
#      token_masks, view_grid, background latent, reference_views
# 3. render_views()
#    - cameras from the rigs above; port RenderMesh 'world normal' + mask
#      outputs (pure-torch path in comfy_extras/nodes_mesh_postprocess.py,
#      no GL context needed in the worker)
# 4. sample() per pass, in recipe order:
#    - rig 1 albedo: EmptyFlux2LatentImage(render_size, batch 6),
#      4 steps (distill) -> bake
#    - refine rig 1: render texture views -> VAEEncode -> 2 steps @ 0.1
#      denoise -> rebake
#    - rig 2 albedo (if use_second_rig): confidence bake of rig 1 becomes a
#      SetLatentNoiseMask via ImageToMask(red)->invert; DifferentialDiffusion
#      model wrapper; 8 steps -> batch with rig-1 views -> rebake -> refine
#    - MR pass (if use_pbr): 8 steps -> bake -> refine @ 0.2 -> split
#      metallic (green channel) / roughness (blue channel) atlases
# 5. bake()
#    - port BakeTextureFromViews (comfy_extras/nodes_mesh_postprocess.py):
#      back-project every view, JFA gutter fill, 4096 atlas, padding
# 6. outputs: albedo (+ metallic/roughness + confidence) PNG atlases


def run(job: UltraTexJob, log=None, cancel=None) -> UltraTexJob:
    """Execute the recipe (M2 engine in ``tools.ultratex.engine``).

    Loads the FLUX.2 Klein int8 model + LoRAs, renders the camera-rig views
    of the UV-unwrapped mesh, samples, and bakes the texture atlases. Fills
    ``job.albedo_atlas`` (+ metallic/roughness/confidence) and returns job.
    """
    from .engine import run as _engine_run

    return _engine_run(job, log=log, cancel=cancel)
