"""UltraTex engine (M2): self-contained FLUX.2 Klein 9B int8 texturing.

Imports the ComfyUI core from the vendored ``submodules/ComfyUI`` clone as a
library -- no ComfyUI server, no node graph -- and replicates the
``ultratex_setget_01.json`` recipe by calling the reference node
implementations directly:

- ``comfy_extras.nodes_mesh_postprocess``: ``_render_view`` (pure-torch
  ray-cast rasterizer + BVH), ``_bake_texture_from_views``,
  ``_mesh_view_camera``, ``_compute_vertex_normals``,
  ``_normalize_uvs_to_unit``, ``_prepare_gpu_mesh_processing``
- ``comfy_extras.nodes_ultratex``: ``UltraTexConditioning.execute``
  (view grid, token masks, reference/background latents)
- ``comfy_extras.nodes_camera``: ``CreateCameraInfo`` / ``CreateCameraOrbit``
- ``comfy_extras.nodes_sparse_attention``: BlockSparseAttention (sla)
- ``comfy_extras.nodes_differential_diffusion``: rig-2 masked pass
- ``comfy.sd`` / ``comfy.sample``: model/VAE/LoRA loading + sampling

Pass order (see ULTRATEX_PLAN.md section 1b):
rig-1 albedo (6 views) -> optional refine -> optional rig-2 (confidence-
masked re-sample, bake 12 views) -> optional metallic/roughness pass.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import (DISTILL_LORA_STRENGTH, REFINE_DENOISE_MR, REFINE_DENOISE_RIG1,
               VIEW_ORDER, resolve_reference_views)
from . import SAMPLER, UltraTexJob

_REPO = Path(__file__).resolve().parents[2]
_COMFY = _REPO / "submodules" / "ComfyUI"

_bootstrapped = False
_mm = None    # comfy.model_management
_sd = None    # comfy.sd
_sample = None
_utils = None  # comfy.utils
_mp = None    # nodes_mesh_postprocess
_cam = None   # nodes_camera
_utx = None   # nodes_ultratex
_bsa = None   # nodes_sparse_attention
_dd = None    # nodes_differential_diffusion


def _bootstrap():
    """Import the vendored comfy core + reference node modules (once)."""
    global _bootstrapped, _mm, _sd, _sample, _utils, _mp, _cam, _utx, _bsa, _dd
    if _bootstrapped:
        return
    if not _COMFY.is_dir():
        raise RuntimeError(
            f"Vendored ComfyUI not found at {_COMFY}; clone it with: git clone "
            "https://github.com/Comfy-Org/ComfyUI.git submodules/ComfyUI")
    if str(_COMFY) not in sys.path:
        sys.path.insert(0, str(_COMFY))
    # Match the reference ComfyUI install (--use-sage-attention): without it
    # comfy falls back to attention_basic, which materializes the whole
    # multi-view token grid and spills tens of GB into shared VRAM.
    import comfy.cli_args
    comfy.cli_args.args.use_sage_attention = True
    # Inference only. ComfyUI always runs under no_grad; as a library we must
    # do the same, else the VAE/model parameters (which load with PyTorch's
    # default requires_grad=True) build a full autograd graph per
    # encode/decode/sample and retain every intermediate activation
    # (~4 GB per VAE encode, ~7.5 GB per decode, linear growth -> OOM).
    torch.set_grad_enabled(False)
    # Keep ~1 GB of dedicated VRAM free (comfy's --vram-headroom): DynamicVRAM
    # then sheds weight to avoid peaks spilling into Windows shared VRAM,
    # which stalls the whole run.
    comfy.cli_args.args.vram_headroom = 1.0
    # surface comfy's own load decisions ("Requested to load", "loaded
    # completely / partially", eviction messages) in our worker log
    import logging
    logging.basicConfig(level=logging.INFO)
    import comfy.model_management
    import comfy.sample
    import comfy.sd
    import comfy.utils
    from comfy_extras import nodes_camera
    from comfy_extras import nodes_differential_diffusion
    from comfy_extras import nodes_mesh_postprocess
    from comfy_extras import nodes_sparse_attention
    from comfy_extras import nodes_ultratex
    _mm = comfy.model_management
    _sd = comfy.sd
    _sample = comfy.sample
    _utils = comfy.utils
    _mp = nodes_mesh_postprocess
    _cam = nodes_camera
    _utx = nodes_ultratex
    _bsa = nodes_sparse_attention
    _dd = nodes_differential_diffusion
    _bootstrapped = True


def _unwrap(node_output):
    """comfy_api NodeOutput -> first positional result."""
    args = getattr(node_output, "args", None)
    return args[0] if args else node_output


# -- model loading ------------------------------------------------------------

_KLEIN = "flux-2-klein-base-9b_int8_convrot.safetensors"
_DISTILL = "flux-2-klein-9b_distill_lora_avg_rank_345_bf16.safetensors"
_ALBEDO_LORA = "ultratex_flux2_klein_9b_albedo_lora_bf16.safetensors"
_MR_LORA = "ultratex_flux2_klein_9b_metallic_roughness_lora_bf16.safetensors"
_VAE_UTEX = "ultratex_flux2_vae_fp32.safetensors"
_VAE_REFINE = "flux2-vae.safetensors"
_EMPTY_PROMPT = "flux-2-klein-9b_empty_prompt.safetensors"


class UltraTexModels:
    """FLUX.2 Klein int8 + BlockSparseAttention + LoRA branches + VAEs."""

    def __init__(self, model_dir, use_distill=True, use_pbr=True, log=None):
        _bootstrap()
        self.log = log or (lambda msg, level="info": None)
        base_dir = Path(model_dir)

        def path(sub, name, required=True):
            p = base_dir / sub / name
            if not p.is_file():
                if required:
                    raise FileNotFoundError(
                        f"Missing UltraTex model file: {p} (run get_models.bat)")
                return None
            return str(p)

        self.log("Loading FLUX.2 Klein 9B int8-convrot diffusion model...")
        base = _sd.load_diffusion_model(path("diffusion_models", _KLEIN))
        # BlockSparseAttention, the recipe's model patch: sla, keep 20%,
        # exact_kv_and_rows, min_tokens 12288, extra_tokens 256.
        self.base = _unwrap(_bsa.BlockSparseAttention.execute(
            base, selection={"selection": "sla", "keep_percent": 20},
            start_percent=0.0, end_percent=1.0, dense_blocks="",
            min_tokens=12288, extra_tokens=256,
            sink_conditioning="exact_kv_and_rows", verbose=False))

        distill = path("loras", _DISTILL, required=False)
        self.use_distill = use_distill and distill is not None
        if use_distill and distill is None:
            self.log("distill LoRA not found; albedo pass uses full steps",
                     "warning")
        self.albedo = self.base
        if self.use_distill:
            self.albedo = self._lora(self.albedo, distill, DISTILL_LORA_STRENGTH)
        self.albedo = self._lora(self.albedo,
                                 path("loras", _ALBEDO_LORA), 1.0)
        self.mr = None
        if use_pbr:
            self.mr = self._lora(self.base, path("loras", _MR_LORA), 1.0)

        self.log("Loading UltraTex VAE + refine VAE + empty-prompt embedding...")
        self.vae = _sd.VAE(sd=_utils.load_torch_file(
            path("vae", _VAE_UTEX)))
        self.vae_refine = _sd.VAE(sd=_utils.load_torch_file(
            path("vae", _VAE_REFINE)))
        emb = _utils.load_torch_file(path("embeddings", _EMPTY_PROMPT))
        cond = emb.get("conditioning")
        if cond is None:
            raise RuntimeError("empty-prompt embedding has no 'conditioning' "
                               f"tensor (keys: {list(emb)})")
        opts = {}
        if emb.get("attention_mask") is not None:
            opts["attention_mask"] = emb["attention_mask"]
        self.conditioning = [[cond, opts]]

    def _lora(self, model, lora_path, strength):
        self.log(f"Applying LoRA {Path(lora_path).name} @ {strength}")
        lora = _utils.load_torch_file(lora_path)
        patched, _clip = _sd.load_lora_for_models(model, None, lora,
                                                  strength, 0.0)
        return patched


# -- mesh / reference / camera helpers ----------------------------------------

def load_mesh_arrays(mesh_path, require_uvs=True):
    """(vertices [n,3], faces [k,3], uvs [n,2] normalized, vertex normals
    [n,3]) as float32/int64 CPU tensors. Requires TEXCOORD_0 (the UV
    accordion writes it) unless *require_uvs* is False (render-donor
    geometry, e.g. the circle-marked ghost mesh); raises otherwise."""
    import trimesh

    loaded = trimesh.load(str(mesh_path), force="scene", process=False)
    geoms = (list(loaded.geometry.values())
             if hasattr(loaded, "geometry") else [loaded])
    mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vis = mesh.visual
    uv = getattr(vis, "uv", None)
    _bootstrap()
    if uv is None or len(uv) != len(verts):
        if require_uvs:
            raise ValueError(
                f"{mesh_path}: no per-vertex UVs -- run Unwrap UVs "
                "in the UV accordion first (TEXCOORD_0)")
        uvs = None            # donor only provides geometry
    else:
        uvs = _mp._normalize_uvs_to_unit(np.asarray(uv, dtype=np.float32))
    # xatlas duplicates vertices along UV seams; a straight per-vertex normal
    # computation on the duplicated topology makes every seam a visible
    # normal discontinuity (shells "change angle"). Weld by position, compute
    # smooth area-weighted normals on the welded mesh, map back per duplicate.
    eps = 1e-5 * float(np.linalg.norm(verts.max(0) - verts.min(0))) + 1e-12
    key = np.round(verts / eps).astype(np.int64)
    _uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    corners = inverse[faces]                     # (F, 3) welded ids
    face_n = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                      verts[faces[:, 2]] - verts[faces[:, 0]])
    acc = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
    for k in range(3):
        np.add.at(acc, corners[:, k], face_n)
    welded = acc / (np.linalg.norm(acc, axis=1, keepdims=True) + 1e-12)
    normals = welded[inverse].astype(np.float32)
    return (torch.from_numpy(verts), torch.from_numpy(faces),
            (None if uvs is None else torch.from_numpy(uvs)),
            torch.from_numpy(normals))


def _camera(yaw=0.0, pitch=0.0, distance=1.6, fov=35.49):
    """One CreateCameraInfo('orbit') camera_info (recipe: dist 1.6, fov 35.49)."""
    return _unwrap(_cam.CreateCameraInfo.execute(
        mode={"mode": "orbit", "yaw": float(yaw), "pitch": float(pitch),
              "distance": float(distance)},
        target_x=0.0, target_y=0.0, target_z=0.0, roll=0.0,
        fov=float(fov), zoom=1.0, camera_type="perspective"))


def rig1_cameras():
    """6 canonical views: orbit 4 + top and bottom."""
    return list(_unwrap(_cam.CreateCameraOrbit.execute(
        _camera(), 4, 360.0, 0.0, "top and bottom")))


def rig2_cameras():
    """Second rig: orbit 4 at azimuth 45 + elev 0/45 top-bottom pair."""
    sides = list(_unwrap(_cam.CreateCameraOrbit.execute(
        _camera(yaw=45.0), 4, 360.0, 0.0, "none")))
    pair = list(_unwrap(_cam.CreateCameraOrbit.execute(
        _camera(pitch=45.0), 2, 180.0, -90.0, "none")))
    return sides + pair


def load_reference(image_path, mask_path=None):
    """Reference image [1,H,W,3] + optional object mask [1,H,W] (comfy
    Mask layout), 0..1 floats.

    With no explicit mask, a cutout PNG's alpha channel (the app's RMBG-2.0
    cutouts) is used as the object mask automatically."""
    from PIL import Image
    img = Image.open(str(image_path))
    mask = None
    if mask_path is not None:
        m = Image.open(str(mask_path)).convert("L").resize(img.size)
        mask = torch.from_numpy(
            np.asarray(m, dtype=np.float32)[None] / 255.0)
    rgba = img.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
    if float(alpha.min()) < 0.999:      # a real cutout, not a flat photo
        mask = torch.from_numpy(alpha)[None] if mask is None else mask
    ref = torch.from_numpy(
        np.asarray(rgba.convert("RGB"), dtype=np.float32) / 255.0)[None]
    return ref, mask


# -- render / conditioning / sampling / bake ----------------------------------

def render_views(verts, faces, uvs, vnorms, cameras, size,
                 mode="world normal", texture=None):
    """RenderMesh equivalent (pure torch, no GL). mode 'world normal' gives
    the canonical normal renders over white; 'texture' needs ``texture``
    [H,W,3] and renders it over black. Returns (images [V,H,W,3],
    masks [V,H,W]) on the mesh device."""
    dev = verts.device
    tri = verts[faces]
    bvh = _mp._build_triangle_bvh(tri)
    # view background: #282828 for the saved screenshots (cosmetic --
    # UltraTexConditioning re-composites over white via the hit mask).
    bg = torch.full((3,), 0x28 / 255.0, device=dev) if mode == "world normal" \
        else torch.zeros(3, device=dev)
    render_normal = None
    if mode == "world normal":
        # World axes turned by the first camera's yaw (recipe: normals with
        # the up axis negated are applied inside UltraTexConditioning).
        fwd, right = _mp._mesh_view_camera(cameras[0], dev)[1:3]
        y = fwd.new_tensor([0.0, 1.0, 0.0])
        x = torch.cross(fwd, y, dim=-1)
        x = (torch.nn.functional.normalize(x, dim=-1, eps=1e-6)
             if float(x.norm()) > 1e-6 else right)
        render_normal = torch.stack([x, y, torch.cross(x, y, dim=-1)])
    if vnorms is None and mode == "world normal":
        vnorms = _mp._compute_vertex_normals(verts, faces)
    imgs, masks = [], []
    for cam_info in cameras:
        eye, fwd, right, up, fov_rad, ortho_half = \
            _mp._mesh_view_camera(cam_info, dev)
        img, hit, _depth = _mp._render_view(
            tri, bvh, uvs, faces, texture, eye, fwd, right, up, fov_rad,
            int(size), int(size), ray_chunk=1 << 22,
            vertex_colors=None, vertex_normals=vnorms,
            render_normal=render_normal, ortho_half=ortho_half)
        imgs.append(torch.where(hit[..., None], img.clamp(0.0, 1.0), bg))
        masks.append(hit.to(torch.float32))
    return torch.stack(imgs), torch.stack(masks)


def build_conditioning(conditioning, vae, normals, masks, reference,
                       reference_mask=None, extra_views=None):
    """UltraTexConditioning port: optional reference mask plus per-view
    extra references (*extra_views*: ``{view name: image [1,H,W,3]}`` for
    right/back/left/top/bottom)."""
    out = _utx.UltraTexConditioning.execute(
        conditioning, vae, normals, masks, reference,
        extra_views=extra_views or {}, reference_mask=reference_mask)
    return _unwrap(out)


def sample_pass(model, conditioning, latent, steps, seed,
                denoise=1.0, noise_mask=None):
    """One cfg-1.0 euler/simple pass (empty prompt: cond == uncond)."""
    noise = _sample.prepare_noise(latent, seed)
    return _sample.sample(
        model, noise, int(steps), SAMPLER["cfg"], SAMPLER["sampler"],
        SAMPLER["scheduler"], conditioning, conditioning, latent,
        denoise=float(denoise), noise_mask=noise_mask, seed=int(seed))


def bake_views(verts, faces, uvs, vnorms, images, cameras, resolution,
               fill_unseen=True, base_texture=None, blend_exponent=4.0):
    """BakeTextureFromViews port. Returns (atlas [H,W,3], confidence [H,W])
    as numpy arrays in 0..1. fill_unseen=True grows unseen texels from seen
    surface (the recipe's setting); False composites over base_texture."""
    dev = _mm.get_torch_device()
    idev, idtype = _mm.intermediate_device(), _mm.intermediate_dtype()
    images = images.to(idev, idtype)
    _mp._prepare_gpu_mesh_processing(
        dev, max(int(resolution) * int(resolution) * 256,
                 int(faces.shape[0]) * 512))
    tex, conf = _mp._bake_texture_from_views(
        verts, faces, uvs, vnorms,
        None if fill_unseen else base_texture, None,
        images, None, list(cameras), int(resolution), float(blend_exponent))
    # newer comfy returns GPU tensors, older numpy -- normalize to tensors
    tex = tex if torch.is_tensor(tex) else torch.from_numpy(
        np.ascontiguousarray(tex))
    conf = conf if torch.is_tensor(conf) else torch.from_numpy(
        np.ascontiguousarray(conf))
    out = tex.to(dev).float(), conf.to(dev).float()
    # release the bake scratch buffers back to the driver; leaving them in
    # torch's caching allocator starves the next sampling pass, which then
    # demotes the transformer to partial (streamed) residency and re-computes
    # its LoRA weight patches on every layer forward -- minutes per step.
    _mm.soft_empty_cache()
    return out


# -- orchestration -------------------------------------------------------------

def save_atlas(arr, path, flip=True):
    """Save a [H,W(,C)] float 0..1 numpy/tensor array as PNG.

    The in-memory bake atlas uses comfy's top-origin texel mapping
    (``_sample_image01``: v = row from the top) while mesh UVs loaded
    through trimesh are lower-left (glTF files store 1-v). Everything
    that consumes the saved PNG -- glTF viewers and OBJ tools alike --
    expects the other orientation, so rows are flipped on save. The
    in-memory atlas is untouched (the refine pass re-samples it raw).
    Camera screenshots (save_views) must NOT be flipped: their row 0
    is the top of the render.
    """
    from PIL import Image
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if flip:
        arr = np.ascontiguousarray(arr[::-1])   # flip rows (vertical mirror)
    if arr.ndim == 2:
        img = Image.fromarray(
            (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8), "L")
    else:
        img = Image.fromarray(
            (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))


def _view_tag(i):
    """File-name tag of rig view *i* (order matches the camera rigs)."""
    return VIEW_ORDER[i] if i < len(VIEW_ORDER) else str(i)


def save_views(images, out_dir, prefix):
    """Save a [V,H,W(,C)] view batch as per-view PNGs (backend screenshots:
    ``<out_dir>/<prefix>_<view>.png``). Order matches the camera rig:
    front, right, back, left, top, bottom."""
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(len(images)):
        p = out / f"{prefix}_{_view_tag(i)}.png"
        save_atlas(images[i], p, flip=False)
        paths.append(str(p))
    return paths


def load_views(out_dir, prefix, count, size):
    """Inverse of save_views for [V,H,W,3] / [V,H,W] batches: returns the
    views as a float 0..1 torch tensor (cpu), or None if any file is
    missing or was saved at a different resolution."""
    out = Path(out_dir)
    planes = []
    for i in range(count):
        p = out / f"{prefix}_{_view_tag(i)}.png"
        if not p.is_file():
            return None
        try:
            from PIL import Image
            img = Image.open(str(p))
            if img.size != (int(size), int(size)):
                return None
            planes.append(np.asarray(img, dtype=np.float32) / 255.0)
        except Exception:  # noqa: BLE001 - a corrupt cache entry re-renders
            return None
    return torch.from_numpy(np.stack(planes))


def normal_views_cache_key(verts, faces, vnorms, size):
    """Stable key for the world-normal view cache: render geometry (the
    donor mesh when one is set) + vertex normals + render size. Seed,
    steps and reference images do NOT affect the normal renders, so a
    re-run with a different seed reuses them."""
    import hashlib
    h = hashlib.sha256()
    h.update(b"ultratex-normals-v1")
    for arr in (verts, faces, vnorms):
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
        arr = np.ascontiguousarray(np.asarray(arr))
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    h.update(str(int(size)).encode())
    return h.hexdigest()[:16]


def cached_or_render_normals(views_dir, rig_tag, key, cameras, size,
                             verts, faces, vnorms, render_fn, log):
    """World-normal views for one rig, cached on disk.

    The normal renders depend only on geometry + camera rig + render size,
    so they are reused across runs (e.g. seed changes) as long as the
    ``normals_cache.json`` sidecar still matches *key*. Saves both the
    composited normal renders and the raw hit masks the conditioning node
    needs."""
    views_dir = Path(views_dir)
    saved = {}
    meta = views_dir / "normals_cache.json"
    if meta.is_file():
        try:
            import json
            saved = json.loads(meta.read_text())
        except Exception:  # noqa: BLE001 - unreadable meta -> re-render
            saved = {}
    if saved.get(rig_tag) == key:
        normals = load_views(views_dir, f"normals_{rig_tag}",
                             len(cameras), size)
        masks = load_views(views_dir, f"masks_{rig_tag}",
                           len(cameras), size)
        if normals is not None and masks is not None:
            log(f"Reusing cached world-normal views ({rig_tag}, "
                f"{len(cameras)} views @ {size}px).", "info")
            dev = verts.device
            return (normals.to(dev), masks.to(dev)[..., 0]
                    if masks.ndim == 4 else masks.to(dev))

    normals, masks = render_fn()
    save_views(normals, views_dir, f"normals_{rig_tag}")
    save_views(masks, views_dir, f"masks_{rig_tag}")
    saved[rig_tag] = key
    try:
        import json
        views_dir.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps(saved))
    except Exception:  # noqa: BLE001 - caching is best-effort
        pass
    return normals, masks


def export_textured_glb(mesh_path, verts, faces, uvs, atlas_path,
                        out_path) -> str:
    """Write a display-ready GLB: the mesh with TEXCOORD_0 plus the baked
    albedo atlas as its base-color texture (viewer.add_textured preset),
    plus an OBJ twin (+ texture + mtl) like the other pipeline stages."""
    import numpy as np
    import trimesh
    from PIL import Image
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals
    mesh = trimesh.Trimesh(
        vertices=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False, validate=False)
    img = Image.open(str(atlas_path))
    # Explicit PBR values: trimesh's defaults (baseColorFactor 0.4 gray,
    # roughness 0.9, implicit metallic 1.0) render the albedo far too dark.
    material = PBRMaterial(
        baseColorTexture=img,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],   # texture at full strength
        metallicFactor=0.0,                     # dielectric: no env map needed
        roughnessFactor=0.55,                   # slight spec, not mirror-flat
        name=out_path.stem)
    mesh.visual = TextureVisuals(
        uv=np.asarray(uvs, dtype=np.float64),
        material=material,
        # trimesh already converts UVs to glTF's top-left origin on write
        # (TEXCOORD_0 = 1-v) and embeds the image bytes unmodified, so the
        # RAW image is correct for the GLB. The OBJ exporter writes vt and
        # the texture file as-is, so it gets the same raw image.
        image=img)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # trimesh derives the exported texture file name from the material name;
    # match it to the output stem so the GLB/OBJ sidecars agree.
    material = getattr(mesh.visual, "material", None)
    if material is not None and hasattr(material, "name"):
        material.name = out_path.stem
    mesh.export(str(out_path))
    # OBJ twin: restore the raw (unflipped) texture so the sidecar PNG and
    # the vt coordinates agree.
    if material is not None and hasattr(material, "image"):
        material.image = img
    try:
        mesh.export(str(out_path.with_suffix(".obj")),
                    include_texture=True,
                    mtl_name=f"{out_path.stem}.mtl")
    except Exception:  # noqa: BLE001 - the GLB is the primary result
        pass
    return str(out_path)


def bake_ws_normal(target_path, donor_path, resolution=4096,
                   cage_distance=0.05, log=None) -> str:
    """World-space normal atlas (Maya transfer-style bake).

    The displayed mesh is the low-poly target: its welded, area-weighted
    (Maya 'Average Normals'-style) vertex normals steer the cage rays. The
    donor mesh (the circle-marked Outliner entry) is the high-poly source:
    per covered texel, a cage ray along the smoothed low-poly normal finds
    the donor surface (closest-point fallback), whose interpolated smooth
    normal is encoded as (n * 0.5 + 0.5) and JFA-filled into the UV gutter.
    Saved as <target>_wsnormal.png next to the target mesh.
    """
    log = log or (lambda msg, level="info": None)
    v, f, uvs, vnorms = load_mesh_arrays(target_path)
    hv, hf, _hu, hn = load_mesh_arrays(donor_path, require_uvs=False)
    dev = _mm.get_torch_device()
    v, f, vnorms = v.to(dev), f.to(dev), vnorms.to(dev)
    hv, hf, hn = hv.to(dev), hf.to(dev), hn.to(dev)
    H = W = int(resolution)
    faces_np = f.detach().cpu().numpy().astype(np.int64)
    uvs_np = uvs.detach().cpu().numpy().astype(np.float32)
    tick0 = time.time()
    log(f"Rasterizing UVs ({H}x{W}) ...", "info")
    face_idx, bary_uv, mask = _mp._rasterize_uv_barycentric(faces_np, uvs_np, H)
    out = torch.full((H, W, 3), 0.5, device=dev, dtype=torch.float32)
    if bool(mask.any()):
        fsel = face_idx[mask]                    # [K] texel -> target face
        bsel = bary_uv[mask]                     # [K,3]
        vtri = f[fsel]                           # [K,3] vertex ids

        def interp(attr):
            return (bsel[:, :, None] * attr[vtri]).sum(1)

        P = interp(v)
        Nl = torch.nn.functional.normalize(interp(vnorms), dim=-1, eps=1e-6)
        tri = hv[hf]
        log(f"Building BVH of {Path(str(donor_path)).name} "
            f"({tri.shape[0]:,} tris) ...", "info")
        bvh = _mp._build_triangle_bvh(tri)
        diag = float((hv.amax(0) - hv.amin(0)).norm().clamp_min(1e-6))
        cage = max(1e-6, float(cage_distance) * diag)
        origin = P + Nl * cage
        t_hit, f_hit, ray_hit = _mp._closest_hit_rays_bvh(
            origin, -Nl, tri, bvh, tmin=0.0, tmax=2.0 * cage,
            cull_backface=True)
        bface = f_hit.clamp_min(0)
        hitpoint = origin - t_hit[:, None] * Nl
        miss = ~ray_hit
        if bool(miss.any()):
            log(f"Cage rays missed {int(miss.sum())} texels; "
                "closest-point fallback ...", "info")
            bp, bf = _mp._closest_points_on_mesh_bvh(P[miss], tri, bvh,
                                                     return_face=True)
            bface = bface.clone()
            hitpoint = hitpoint.clone()
            bface[miss] = bf.clamp_min(0)
            hitpoint[miss] = bp
        bary = _mp._barycentric(hitpoint, tri[bface])
        Nh = torch.nn.functional.normalize(
            (bary[:, :, None] * hn[hf[bface]]).sum(1), dim=-1, eps=1e-6)
        # a back surface grabbed in a deep crevice: fall back to the local
        # low-poly smooth normal instead of a wrong-facing one
        nz = (Nh * Nl).sum(-1)
        Nh = torch.where(nz[:, None] < 0.0, Nl, Nh)
        out[mask] = Nh.mul(0.5).add(0.5).clamp(0.0, 1.0)
    filled = _mp._jfa_fill_gpu(out.cpu().numpy(), mask.cpu().numpy())
    out_path = Path(target_path).with_name(
        Path(target_path).stem + "_wsnormal.png")
    save_atlas(filled, out_path)     # row-flip like the other atlases
    log(f"Bake WS Normal done in {time.time() - tick0:.1f}s: {out_path}",
        "ok")
    return str(out_path)


def run(job: UltraTexJob, log=None, cancel=None) -> UltraTexJob:
    """Execute the full recipe for one job; fills job.albedo_atlas (+ PBR /
    confidence atlas paths) and job.stats."""
    _bootstrap()
    log = log or (lambda msg, level="info": None)

    def tick(stage):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("UltraTex cancelled")
        log(stage)

    def vram(tag):
        alloc = torch.cuda.memory_allocated() / 1e9
        rsvd = torch.cuda.memory_reserved() / 1e9
        free = _mm.get_free_memory(_mm.get_torch_device()) / 1e9
        log(f"vram[{tag}]: alloc {alloc:.1f} reserved {rsvd:.1f} "
            f"free {free:.1f} GB")

    tick("Loading models")
    models = UltraTexModels(job.model_dir, use_distill=job.use_distill_lora,
                            use_pbr=job.use_pbr, log=log)

    tick("Loading mesh + reference")
    v, f, uvs, vnorms = load_mesh_arrays(job.mesh_path)
    # Conditioning-render geometry: the circle-marked ghost mesh when set
    # (screenshots only, no UVs needed); the textured mesh otherwise. The
    # textured mesh stays the bake/projection target (its UVs + geometry
    # produce the atlases and the display GLB).
    vr, fr, vnr = v, f, vnorms
    if job.render_mesh_path and str(job.render_mesh_path) != str(job.mesh_path):
        vr, fr, _ru, vnr = load_mesh_arrays(job.render_mesh_path,
                                            require_uvs=False)
        log(f"Normal renders use the circle-marked mesh "
            f"{Path(str(job.render_mesh_path)).name}.", "info")
    dev = _mm.get_torch_device()
    v, f = v.to(dev), f.to(dev)
    uvs_d = uvs.to(dev)
    # cpu copies for the display GLB export at the end of the run
    v_np = v.detach().cpu().numpy()
    f_np = f.detach().cpu().numpy()
    uvs_np = (uvs.detach().cpu().numpy() if torch.is_tensor(uvs)
              else np.asarray(uvs))
    vnorms = vnorms.to(dev)
    vr, fr, vnr = vr.to(dev), fr.to(dev), vnr.to(dev)
    ref, ref_mask = load_reference(job.reference_image, job.reference_mask)
    ref = ref.to(dev)
    if ref_mask is not None:
        ref_mask = ref_mask.to(dev)

    # Per-view reference images (the UI's view tags). Direct image for a
    # tagged view, closest tagged view for the rest (front -> left/right
    # sides, back -> top/bottom, ...); the run's plain reference image stays
    # the front fallback when nothing is tagged.
    extras = {}
    tagged = {view: Path(p) for view, p in (job.reference_views or {}).items()
              if view in VIEW_ORDER and p and Path(p).is_file()}
    if tagged:
        resolved = resolve_reference_views(tagged)
        front = resolved.get("front")
        if front is not None and front.resolve() != Path(
                job.reference_image).resolve():
            ref, ref_mask = load_reference(front)
            ref = ref.to(dev)
            ref_mask = None if ref_mask is None else ref_mask.to(dev)
        for view in VIEW_ORDER[1:]:
            path = resolved.get(view)
            # a view that fell back to the front image would just duplicate
            # the reference slot -- skip it
            if path is None or (front is not None and path.resolve()
                                == front.resolve()):
                continue
            img, _mask = load_reference(path)   # mask: cutout alpha / mesh
            extras[view] = img.to(dev)
        log("Reference views: "
            + ", ".join(f"{v}={Path(p).name}" for v, p in resolved.items())
            + (f" (+ {len(extras)} extra view ref(s))"
               if extras else ""), "info")

    size = int(job.render_size)
    latent_hw = size // 16
    # backend screenshots land next to the atlases: <mesh>_ultratex/views/
    out_dir = Path(str(job.mesh_path)).with_suffix("").parent / \
        (Path(str(job.mesh_path)).with_suffix("").name + "_ultratex")
    views_dir = out_dir / "views"

    def empty_latent(batch):
        return torch.zeros([batch, 128, latent_hw, latent_hw],
                           device=_mm.intermediate_device())

    def decode(samples):
        # VAE decode through comfy's memory management: tiled above 1536px
        # so VRAM stays flat, then release the sampler's cache fragments.
        if latent_hw >= 96:
            out = models.vae.decode_tiled(samples, tile_x=1024, tile_y=1024,
                                          overlap=64)
        else:
            out = models.vae.decode(samples)
        _mm.soft_empty_cache()
        return out

    def texture_latent(atlas, cameras):
        atlas = torch.as_tensor(atlas, dtype=torch.float32, device=dev)
        imgs, _m = render_views(v, f, uvs_d, None, cameras, size,
                                mode="texture", texture=atlas)
        return models.vae_refine.encode(imgs), imgs

    def pin_full(model):
        # Keep the transformer fully resident before sampling. Left alone,
        # comfy's load_models_gpu sizes the residency from whatever VRAM is
        # still free; after renders/encodes/bakes that can drop it to partial
        # load where every layer forward re-computes the LoRA weight patches
        # (orders of magnitude slower -- the rig-2 "hang").
        _mm.load_models_gpu([model], force_full_load=True)

    def rebake(images, cameras):
        return bake_views(v, f, uvs_d, vnorms, images, cameras,
                          job.bake_size, fill_unseen=True,
                          blend_exponent=4.0)

    # -- pass 1: rig-1 albedo -------------------------------------------------
    cameras1 = rig1_cameras()
    # world-normal renders depend only on geometry + rig + render size, so
    # they are cached across runs (seed/steps/reference changes reuse them;
    # a render-size or geometry change re-renders).
    normals_key = normal_views_cache_key(vr, fr, vnr, size)
    tick(f"Rendering rig-1 world normals ({size}x{size}, 6 views)")
    normals1, masks1 = cached_or_render_normals(
        views_dir, "rig1", normals_key, cameras1, size, vr, fr, vnr,
        lambda: render_views(vr, fr, None, vnr, cameras1, size,
                             mode="world normal"),
        log)
    tick("Building UltraTex conditioning")
    cond = build_conditioning(models.conditioning, models.vae, normals1,
                              masks1, ref, ref_mask, extra_views=extras)
    steps1 = 4 if models.use_distill else int(job.steps)
    tick(f"Albedo rig-1 sampling ({steps1} steps @ {size}x{size}, batch 6)")
    pin_full(models.albedo)
    vram("rig1_pre")
    images1 = decode(
        sample_pass(models.albedo, cond, empty_latent(6), steps1, job.seed))
    save_views(images1, views_dir, "albedo_rig1")
    tick(f"Baking albedo atlas ({int(job.bake_size)}px)")
    tex, conf = rebake(images1, cameras1)
    vram("bake1")

    # -- pass 2: rig-1 refine --------------------------------------------------
    images_r1 = images1
    if job.use_refine:
        tick(f"Albedo refine (2 steps @ denoise {REFINE_DENOISE_RIG1})")
        lat, _imgs = texture_latent(tex, cameras1)
        pin_full(models.albedo)
        vram("refine_pre")
        images_r1 = decode(sample_pass(
            models.albedo, cond, lat, 2, job.seed + 1,
            denoise=REFINE_DENOISE_RIG1))
        save_views(images_r1, views_dir, "albedo_refined")
        tex, conf = rebake(images_r1, cameras1)

    # -- pass 3: second rig (confidence-masked re-sample) ----------------------
    images_all, cameras_all = images_r1, cameras1
    vram("post_refine")
    if job.use_second_rig:
        cameras2 = rig2_cameras()
        tick(f"Rendering rig-2 world normals ({len(cameras2)} views)")
        normals2, masks2 = cached_or_render_normals(
            views_dir, "rig2", normals_key, cameras2, size, vr, fr, vnr,
            lambda: render_views(vr, fr, None, vnr, cameras2, size,
                                 mode="world normal"),
            log)
        cond2 = build_conditioning(models.conditioning, models.vae,
                                   normals2, masks2, ref, ref_mask,
                                   extra_views=extras)
        tick("Rendering rig-2 texture + confidence-mask views")
        lat2, _imgs = texture_latent(tex, cameras2)
        conf_rgb = torch.as_tensor(conf, dtype=torch.float32,
                                   device=dev)[..., None] \
            .expand(-1, -1, 3).contiguous()
        conf_views, _m = render_views(v, f, uvs_d, None, cameras2, size,
                                      mode="texture", texture=conf_rgb)
        # mask: re-sample where rig 1 was poorly seen (inverted confidence)
        noise_mask = 1.0 - conf_views[..., 0]
        noise_mask = torch.nn.functional.interpolate(
            noise_mask[:, None], size=(latent_hw, latent_hw),
            mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        model2 = _unwrap(_dd.DifferentialDiffusion.execute(models.albedo,
                                                           strength=1.0))
        pin_full(model2)
        vram("rig2_pre")
        tick(f"Albedo rig-2 sampling ({int(job.steps)} steps, masked)")
        sampled2 = sample_pass(model2, cond2, lat2.to(_mm.intermediate_device()),
                               int(job.steps), job.seed + 2, denoise=1.0,
                               noise_mask=noise_mask.to(_mm.intermediate_device()))
        images2 = decode(sampled2)
        vram("rig2_post")
        save_views(images2, views_dir, "albedo_rig2")
        images_all = torch.cat([images_r1, images2], dim=0)
        cameras_all = list(cameras1) + list(cameras2)
        tick(f"Rebaking albedo ({len(cameras_all)} views)")
        tex, conf = rebake(images_all, cameras_all)

    # -- pass 4: metallic/roughness --------------------------------------------
    if job.use_pbr:
        tick(f"PBR sampling ({int(job.steps)} steps, metallic/roughness LoRA)")
        pin_full(models.mr)
        vram("pbr_pre")
        mr_imgs = decode(sample_pass(
            models.mr, cond, empty_latent(6), int(job.steps), job.seed + 3))
        save_views(mr_imgs, views_dir, "pbr_rig1")
        tex_mr, _conf_mr = rebake(mr_imgs, cameras1)
        if job.use_refine:
            tick(f"PBR refine (2 steps @ denoise {REFINE_DENOISE_MR})")
            lat_mr, _i = texture_latent(tex_mr, cameras1)
            pin_full(models.mr)
            mr_imgs = decode(sample_pass(
                models.mr, cond, lat_mr, 2, job.seed + 4,
                denoise=REFINE_DENOISE_MR))
            tex_mr, _conf_mr = rebake(mr_imgs, cameras1)
        # recipe: metallic = green channel, roughness = blue channel
        metallic = tex_mr[..., 1]
        roughness = tex_mr[..., 2]

    # -- outputs ----------------------------------------------------------------
    job.albedo_atlas = str(out_dir / "albedo.png")
    job.confidence_atlas = str(out_dir / "confidence.png")
    save_atlas(tex, job.albedo_atlas)
    save_atlas(conf, job.confidence_atlas)
    if job.use_pbr:
        job.metallic_atlas = str(out_dir / "metallic.png")
        job.roughness_atlas = str(out_dir / "roughness.png")
        save_atlas(metallic, job.metallic_atlas)
        save_atlas(roughness, job.roughness_atlas)
    # display-ready GLB (+ OBJ twin) next to the source mesh, where the
    # Outliner lists it; the _ultratex folder keeps the atlases and views
    src = Path(str(job.mesh_path))
    job.textured_glb = export_textured_glb(
        job.mesh_path, v_np, f_np, uvs_np, job.albedo_atlas,
        src.with_name(src.stem + "_textured.glb"))
    obj_twin = src.with_name(src.stem + "_textured.obj")
    log(f"Saved {Path(job.textured_glb).name}"
        + (f" (+ {obj_twin.name}, texture, mtl)" if obj_twin.is_file()
           else " (mesh only, no texture)"), "ok")
    job.stats = {
        "views": len(cameras_all),
        "views_dir": str(views_dir),
        "render_size": size,
        "bake_size": int(job.bake_size),
        "distill_lora": bool(models.use_distill),
        "second_rig": bool(job.use_second_rig),
        "pbr": bool(job.use_pbr),
        "view_refs": ["front"] + sorted(extras) if tagged else [],
    }
    log(f"UltraTex done: {job.albedo_atlas}")
    return job
