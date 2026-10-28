"""Standalone LATO.2 mesh-generation worker.

Spawned by PS6_UI/workers/controllers.py for **Run LATO.2**. It loads the
LATO.2 checkpoint set (V-Flow, V-VAE, OffsetHead, T-Flow, T-VAE and the voxel
encoder), regenerates the displayed mesh with a target vertex count, exports a
geometry-only GLB, then exits. Keeping this in a one-shot process releases the
~4.8 GB of weights (plus the DINOv2 encoder) as soon as the job finishes.

The protocol is the same JSON-lines format used by worker.py /
hi3d_worker.py / cello_worker.py:

  stdin:  {"cmd": "generate", "mesh": "...", "output": "...",
           "vert_num": 2000, "seed": 0,
           "vflow_steps": 24, "tflow_steps": 50, "cfg_strength": 3.0,
           "fill_quad_rings": false, "drop_tiny_patches": true,
           "shape512_state": "...",         # optional; regenerated too
           "img_res": 518, "render_azimuth": 45.0,
           "render_elevation": 30.0}
  stdout: {"type": "ready"}
          {"type": "log", "msg": ..., "level": "info"|"ok"|"warn"|"error"}
          {"type": "done", "output": "...", "obj": "...", "seed": 42,
           "vertex_count": 2000, "face_count": 4000}
          {"type": "error", "msg": "..."}

LATO.2 conditions on a single white-model render of the source mesh (the
same view used by scripts/e2e_inference.py) and generates new vertices and
connectivity, so the result is a retopology rather than a refinement of the
source topology. Its output is normalized to the canonical [-0.5, 0.5] cube,
like UniForge's own geometry outputs.
"""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent    # PS6_UI
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_paths import (
    SUBMODULES_DIR,
    find_lato2_dir,
    find_lato2_source,
    find_model_dir,
    missing_lato2_assets,
)

MODEL_DIR = find_model_dir()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Deterministic cuBLAS kernel selection (must be set before torch/CUDA
# init); see worker.py for why this matters for same-seed re-runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# The DINOv2 torch.hub repo auto-enables xformers whenever it imports. The
# optional xformers wheel in this venv is built for an older torch/CUDA and
# its memory_efficient_attention kernels reject fp32 on sm_120, so force the
# repo's native scaled_dot_product_attention path instead.
os.environ.setdefault("XFORMERS_DISABLED", "1")

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


def _ensure_dino_hub(hub_dir: Path) -> None:
    """Place the DINOv2 torch.hub repo next to the downloaded checkpoint.

    DinoV2Encoder calls torch.hub.load with source="local" when the repo is
    present, which avoids a GitHub download. A repository already cached in
    the user's global torch hub is copied over.
    """
    local_repo = hub_dir / "facebookresearch_dinov2_main"
    if local_repo.is_dir():
        return
    cached = (Path.home() / ".cache" / "torch" / "hub"
              / "facebookresearch_dinov2_main")
    if not cached.is_dir():
        emit({"type": "log", "msg": (
            "DINOv2 hub repository is not cached locally; torch.hub will "
            "download it from GitHub once."), "level": "warn"})
        return
    try:
        hub_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cached, local_repo)
        emit({"type": "log", "msg": (
            f"Copied the cached DINOv2 hub repository to {local_repo}.")})
    except OSError as ex:
        emit({"type": "log", "msg": (
            f"Could not copy the DINOv2 hub repository ({ex}); torch.hub "
            "will download it from GitHub once."), "level": "warn"})


def _load_models(lato2_dir: Path, device):
    """Load the full LATO.2 checkpoint set and return a bundle of modules."""
    from models import (
        DinoV2Encoder,
        OffsetHead,
        TopoFlowEulerSampler,
        TopologySiTFlow,
        TopologyVAE,
        VertexSLatFlowModel,
        VertFlowEulerCfgSampler,
        VertexVAE,
        VoxelFieldConditioner,
    )
    from utils.load import load_latov2_model

    def load(label, cls, filename):
        emit({"type": "log", "msg": f"Loading LATO.2 {label} ..."})
        return load_latov2_model(cls, lato2_dir / filename, device)

    vflow, vflow_cfg = load("V-Flow", VertexSLatFlowModel, "vflow.pt")
    vvae, vvae_cfg = load("V-VAE", VertexVAE, "vvae.pt")
    offset_head, _ = load("OffsetHead", OffsetHead, "offset_head.pt")
    tflow, tflow_cfg = load("T-Flow", TopologySiTFlow, "tflow.pt")
    tvae, _ = load("T-VAE", TopologyVAE, "tvae.pt")
    voxel_encoder, venc_cfg = load(
        "voxel encoder", VoxelFieldConditioner, "voxel_encoder.pt")

    res = int(vvae_cfg["resolution"])
    min_res = int(vvae_cfg["min_resolution"])
    latent_dim = int(vflow_cfg["latent_dim"])
    density_max = float(vflow_cfg["max_vertex_num"])
    z_dim = int(tflow_cfg["args"]["z_dim"])
    num_discrete = int(tflow_cfg["args"]["num_discrete"])
    max_vertices = int(tflow_cfg["args"]["max_vertices"])
    latent_scale = float(tflow_cfg["latent_scale"])
    voxel_res = int(venc_cfg["resolution"])

    if num_discrete != res:
        raise ValueError(
            f"T-Flow num_discrete={num_discrete} != V-VAE resolution={res}; "
            "the generated vertex voxels would be in the wrong coordinate "
            "space.")
    if voxel_res != min_res:
        raise ValueError(
            f"voxel encoder resolution={voxel_res} != V-VAE "
            f"min_resolution={min_res}; both stages must share the same "
            "active-voxel conditioning grid.")

    dino_version = str(vflow_cfg["dino_version"])
    dino_img_res = int(vflow_cfg["image_resolution"])
    dino_hub_dir = lato2_dir / "dinov2"
    _ensure_dino_hub(dino_hub_dir)
    emit({"type": "log", "msg": (
        f"Loading DINOv2 ({dino_version}) from {dino_hub_dir} ...")})
    dino = (DinoV2Encoder(model_name=dino_version,
                          hub_dir=str(dino_hub_dir),
                          img_res=dino_img_res)
            .to(device).eval())

    return {
        "vflow": vflow,
        "vvae": vvae,
        "offset_head": offset_head,
        "tflow": tflow,
        "tvae": tvae,
        "voxel_encoder": voxel_encoder,
        "dino": dino,
        "vertex_sampler": VertFlowEulerCfgSampler(),
        "topo_sampler": TopoFlowEulerSampler(),
        "res": res,
        "min_res": min_res,
        "latent_dim": latent_dim,
        "density_max": density_max,
        "z_dim": z_dim,
        "max_vertices": max_vertices,
        "latent_scale": latent_scale,
        "voxel_res": voxel_res,
        "dino_img_res": dino_img_res,
    }


def _build_batch(mesh_path: Path, bundle: dict, cmd: dict):
    """Preprocess one mesh and render its DINOv2 conditioning view."""
    from dataset.voxel_dataset import VoxelVertexDataset, collate_fn

    res = bundle["res"]
    min_res = bundle["min_res"]
    img_res = int(cmd.get("img_res") or bundle["dino_img_res"])
    azimuth = float(cmd.get("render_azimuth", 45.0))
    elevation = float(cmd.get("render_elevation", 30.0))

    # VoxelVertexDataset expects a directory of meshes; stage a copy so its
    # basename becomes "input" regardless of the source name.
    with tempfile.TemporaryDirectory(prefix="lato2_") as tmp:
        tmp_dir = Path(tmp)
        staged = tmp_dir / f"input{mesh_path.suffix.lower()}"
        shutil.copy2(mesh_path, staged)
        dataset = VoxelVertexDataset(
            root_dir=str(tmp_dir),
            resolution=res,
            min_resolution=min_res,
            need_encoder_inputs=False,
            render=True,
            img_res=img_res,
            render_azimuth=azimuth,
            render_elevation=elevation,
        )
        emit({"type": "log", "msg": (
            f"Preprocessing {mesh_path.name} at {res}^3 and rendering the "
            f"conditioning view (azimuth {azimuth:g}, elevation "
            f"{elevation:g}) ...")})
        item = dataset[0]
        if "error" in item:
            raise RuntimeError(
                "LATO.2 preprocessing failed: "
                + str(item["error"]).splitlines()[0])
        batch = collate_fn([item], resolution=res, min_resolution=min_res)

    if "name" not in batch:
        raise RuntimeError("LATO.2 preprocessing produced no usable mesh.")

    active = batch[f"active_voxels_{min_res}"]
    if active.shape[0] == 0:
        raise RuntimeError(
            "LATO.2 preprocessing extracted no active voxels; the mesh may "
            "be empty or degenerate.")
    return batch


def _clean_faces(faces):
    """Drop degenerate and duplicate triangles from a decoded topology."""
    import numpy as np

    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if faces.shape[0] == 0:
        return faces
    valid = ((faces[:, 0] != faces[:, 1])
             & (faces[:, 1] != faces[:, 2])
             & (faces[:, 0] != faces[:, 2]))
    faces = faces[valid]
    if faces.shape[0] == 0:
        return faces
    # Keep the first copy of each unordered triangle; T-VAE occasionally
    # predicts the same wedge twice.
    _, first = np.unique(np.sort(faces, axis=1), axis=0,
                         return_index=True)
    return faces[np.sort(first)]


def _orient_mesh_faces(mesh) -> None:
    """Make winding consistent and point the surface outward.

    ``edges_to_faces`` builds triangles from an undirected edge graph, so
    neighbouring triangles routinely use opposite windings. That shows up in
    the viewer as every other triangle being backface-culled (dark/flipped,
    with apparent holes). The decoded topology also contains small patches
    that share only vertices with the main surface; their orientation is
    arbitrary, so align each of them with the nearest main-surface normal.
    """
    import numpy as np
    import trimesh

    trimesh.repair.fix_winding(mesh)
    comps = trimesh.graph.connected_components(mesh.face_adjacency, min_len=1)
    if not comps:
        return
    comps.sort(key=len, reverse=True)
    faces = np.array(mesh.faces, dtype=np.int64, copy=True)

    main = np.asarray(comps[0], dtype=np.int64)
    main_mesh = trimesh.Trimesh(vertices=mesh.vertices,
                                faces=faces[main], process=False)
    volume = float(main_mesh.volume)
    if abs(volume) > 1e-12:
        flip_main = volume < 0.0
    else:
        # Nearly flat/open body: use whether normals point away from the
        # centroid instead of the (meaningless) enclosed volume.
        tris = mesh.vertices[faces[main]]
        normals = np.cross(tris[:, 1] - tris[:, 0],
                           tris[:, 2] - tris[:, 0])
        centers = tris.mean(axis=1)
        flip_main = bool(np.einsum(
            "ij,ij->i", normals,
            centers - mesh.vertices.mean(axis=0)).sum() < 0.0)
    if flip_main:
        faces[main] = np.fliplr(faces[main])

    if len(comps) > 1:
        main_mesh = trimesh.Trimesh(vertices=mesh.vertices,
                                    faces=faces[main], process=False)
        others = np.concatenate(
            [np.asarray(c, dtype=np.int64) for c in comps[1:]])
        tris = mesh.vertices[faces[others]]
        normals = np.cross(tris[:, 1] - tris[:, 0],
                           tris[:, 2] - tris[:, 0])
        centers = tris.mean(axis=1)
        _, _, tri_id = trimesh.proximity.closest_point(main_mesh, centers)
        dots = np.einsum("ij,ij->i", normals,
                         main_mesh.face_normals[tri_id])
        start = 0
        for comp in comps[1:]:
            comp = np.asarray(comp, dtype=np.int64)
            if dots[start:start + len(comp)].sum() < 0.0:
                faces[comp] = np.fliplr(faces[comp])
            start += len(comp)

    mesh.faces = faces


def _drop_tiny_patches(mesh, min_faces: int = 10) -> tuple[int, int]:
    """Remove small disconnected surface islands from a decoded topology.

    The edge decoder sometimes emits tiny patches (a handful of triangles)
    whose vertices sit close to, but are not welded into, the main surface.
    In the viewer they show up as overlapping slivers, so the app drops
    them by default. ``face_adjacency`` only links across edges used by
    exactly two faces, which is exactly what isolates slivers that share
    only non-manifold edges with the body; hole filling runs afterwards and
    repairs the small gaps left behind.
    Returns (components removed, faces removed).
    """
    import numpy as np
    import trimesh

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.shape[0] == 0:
        return 0, 0
    comps = trimesh.graph.connected_components(
        mesh.face_adjacency, min_len=1,
        nodes=np.arange(faces.shape[0], dtype=np.int64))
    keep = [c for c in comps if len(c) >= min_faces]
    dropped = [c for c in comps if len(c) < min_faces]
    if not dropped or not keep:
        return 0, 0
    kept_faces = np.sort(
        np.concatenate([np.asarray(c, dtype=np.int64) for c in keep]))
    mesh.faces = faces[kept_faces]
    mesh.remove_unreferenced_vertices()
    return len(dropped), int(sum(len(c) for c in dropped))


def _close_small_holes(mesh, max_vertices: int = 32) -> int:
    """Fan-triangulate small boundary loops left by the edge decoder.

    Only loops up to ``max_vertices`` are closed; larger gaps are left for
    the explicit Close Holes / CelloCut tools instead of bridging them with
    a giant fan.
    """
    import numpy as np

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.shape[0] == 0:
        return 0
    half = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    ordered = np.sort(half, axis=1)
    span = int(faces.max()) + 1
    keys = ordered[:, 0] * span + ordered[:, 1]
    _, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True)
    boundary = half[counts[inverse] == 1]
    if boundary.shape[0] < 3:
        return 0

    outgoing: dict[int, list[int]] = {}
    for a, b in boundary.tolist():
        outgoing.setdefault(a, []).append(b)
    remaining = {tuple(edge) for edge in boundary.tolist()}
    new_faces: list[list[int]] = []

    while remaining:
        start, nxt = next(iter(remaining))
        cur = start
        loop = [start]
        complete = False
        for _ in range(max_vertices + 1):
            remaining.discard((cur, nxt))
            loop.append(nxt)
            if nxt == start:
                complete = True
                break
            options = [v for v in outgoing.get(nxt, ())
                       if (nxt, v) in remaining]
            if not options:
                break
            cur, nxt = nxt, options[0]
        if not complete:
            continue
        length = len(loop) - 1
        if 3 <= length <= max_vertices:
            for i in range(1, length - 1):
                new_faces.append([loop[0], loop[i + 1], loop[i]])

    if new_faces:
        # Never add a triangle whose vertex set already exists: filling the
        # same 3-vertex loop from both directions would create a duplicate
        # closed "pillow" that renders as a non-manifold sliver.
        existing = {tuple(sorted(f)) for f in faces.tolist()}
        kept: list[list[int]] = []
        for f in new_faces:
            key = tuple(sorted(f))
            if key in existing:
                continue
            existing.add(key)
            kept.append(f)
        if kept:
            mesh.faces = np.concatenate(
                [faces, np.asarray(kept, dtype=np.int64)], axis=0)
        return len(kept)
    return 0


def _save_shape512_state(path: Path, vertices, faces) -> None:
    """Write the Run-2nd-stage checkpoint, mirroring the pipeline's format."""
    import numpy as np
    import torch

    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 1,
        "vertices": torch.from_numpy(
            np.asarray(vertices, dtype=np.float32)),
        "faces": torch.from_numpy(np.asarray(faces, dtype=np.int64)),
    }
    temporary = target.with_name(target.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _generate(bundle: dict, cmd: dict, device) -> dict:
    import numpy as np
    import torch

    from modules.sparse import SparseTensor
    from utils.inference import (
        build_voxel_fields,
        decode_vertices,
        edges_to_faces,
        pad_verts,
    )

    mesh_path = Path(cmd["mesh"])
    output_path = Path(cmd["output"])
    if not mesh_path.is_file():
        raise FileNotFoundError(f"input mesh not found: {mesh_path}")

    res = bundle["res"]
    min_res = bundle["min_res"]
    vflow_steps = int(cmd.get("vflow_steps") or 24)
    tflow_steps = int(cmd.get("tflow_steps") or 50)
    cfg_strength = float(cmd.get("cfg_strength") or 3.0)
    inference_threshold = float(cmd.get("inference_threshold", 0.5))
    edge_threshold = float(cmd.get("edge_threshold", 0.0))
    chunk_size = int(cmd.get("chunk_size") or 20000)
    fill_quad_rings = bool(cmd.get("fill_quad_rings", False))
    drop_tiny_patches = bool(cmd.get("drop_tiny_patches", True))
    seed = _resolve_seed(cmd)
    vert_num = int(cmd.get("vert_num") or 2000)
    vert_num = max(3, min(vert_num, bundle["max_vertices"]))

    emit({"type": "log", "msg": (
        f"LATO.2 settings: verts={vert_num}, V-Flow steps={vflow_steps}, "
        f"CFG={cfg_strength:g}, T-Flow steps={tflow_steps}, seed={seed}, "
        f"quad rings={'on' if fill_quad_rings else 'off'}, "
        f"drop patches={'on' if drop_tiny_patches else 'off'}")})

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    batch = _build_batch(mesh_path, bundle, cmd)
    min_active = batch[f"active_voxels_{min_res}"].to(device)
    density = torch.tensor(
        [float(vert_num) / bundle["density_max"] * 1000.0],
        dtype=torch.float32, device=device)
    images = torch.from_numpy(np.stack(batch["image"]))
    cond = bundle["dino"](images).float()
    neg_cond = torch.zeros_like(cond)

    emit({"type": "log", "msg": (
        f"Generating {vert_num} vertices (V-Flow, {vflow_steps} Euler "
        "steps) ...")})
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        noise = SparseTensor(
            coords=min_active.int(),
            feats=torch.randn(
                min_active.shape[0], bundle["latent_dim"], device=device),
        )
        z_pred = bundle["vertex_sampler"].sample(
            model=bundle["vflow"],
            noise=noise,
            cond=cond,
            neg_cond=neg_cond,
            steps=vflow_steps,
            cfg_strength=cfg_strength,
            rescale_t=float(cmd.get("rescale_t") or 1.0),
            density=density,
        )
    emit({"type": "log", "msg": "Decoding vertices (V-VAE + OffsetHead) ..."})
    # The sparse transformer's full-attention path calls flash_attn directly
    # (no fp32 fallback), so the decode must stay inside bf16 autocast just
    # like scripts/e2e_inference.py.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_coords, pred_offsets = decode_vertices(
            bundle["vvae"], bundle["offset_head"], z_pred,
            inference_threshold)

    num_pred = int(pred_coords.shape[0])
    emit({"type": "log", "msg": (
        f"V-Flow produced {num_pred:,} vertices.")})
    if num_pred < 3:
        raise RuntimeError(
            f"LATO.2 generated only {num_pred} vertices; the mesh is too "
            "small to build topology. Try a different vertex count or seed.")
    if num_pred > bundle["max_vertices"]:
        raise RuntimeError(
            f"LATO.2 generated {num_pred:,} vertices, above T-Flow's "
            f"max_vertices={bundle['max_vertices']:,}. Try a different seed.")

    verts = pred_coords[:, 1:].long()
    offsets = pred_offsets.float()

    emit({"type": "log", "msg": (
        f"Generating connectivity (T-Flow, {tflow_steps} Euler steps) ...")})
    with torch.no_grad():
        verts_pad, mask, lengths = pad_verts([verts], device)
        voxel_list = [min_active[min_active[:, 0] == 0, 1:].long()]
        field = build_voxel_fields(voxel_list, bundle["voxel_res"], device)
        cond_vox = bundle["voxel_encoder"](field)
        z0 = torch.randn(1, verts_pad.shape[1], bundle["z_dim"],
                         device=device)
        z_flow = bundle["topo_sampler"].sample(
            model=bundle["tflow"],
            noise=z0,
            verts=verts_pad,
            mask=mask,
            cond=cond_vox,
            steps=tflow_steps,
        )
        z = z_flow.float() / bundle["latent_scale"]

    emit({"type": "log", "msg": (
        f"Decoding edges (T-VAE, chunk {chunk_size:,}) ...")})
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        edges_list = bundle["tvae"].decode(
            z,
            verts=verts_pad,
            verts_mask=mask,
            chunk_size=chunk_size,
            threshold=edge_threshold,
        )

    faces = _clean_faces(
        edges_to_faces(edges_list[0], lengths[0], fill_quad_rings))
    if faces.shape[0] == 0:
        raise RuntimeError(
            "LATO.2 decoded no faces; try a different seed or vertex count.")

    res_f = float(res)
    vertices = (
        verts.cpu().numpy().astype(np.float64) / res_f
        - 0.5
        + offsets.cpu().numpy().astype(np.float64) / (res_f * 2.0)
    )

    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if drop_tiny_patches:
        dropped_comps, dropped_faces = _drop_tiny_patches(mesh)
        if dropped_comps:
            emit({"type": "log", "msg": (
                f"Dropped {dropped_comps} tiny disconnected "
                f"patch{'es' if dropped_comps != 1 else ''} "
                f"({dropped_faces} faces).")})
    # The edge decoder emits unordered triangles; without this the viewer's
    # backface culling shows every other face as inverted/invisible.
    _orient_mesh_faces(mesh)
    filled = _close_small_holes(mesh)
    if filled:
        # Filled fans can disagree with their neighbours; re-run the
        # orientation pass after they are in place.
        _orient_mesh_faces(mesh)
        emit({"type": "log", "msg": (
            f"Closed small gaps in the decoded topology "
            f"({filled} faces added).")})

    # Undo the dataset's bbox normalization (dataset/voxel_dataset.py maps
    # the source into [-0.5, 0.5] via center and max extent) so the result
    # keeps the source mesh's size and position instead of landing centered
    # in a unit cube.
    source = trimesh.load(str(mesh_path), process=False, force="mesh")
    source_v = np.asarray(source.vertices, dtype=np.float64)
    center = (source_v.min(axis=0) + source_v.max(axis=0)) / 2.0
    extent = max(
        float((source_v.max(axis=0) - source_v.min(axis=0)).max()), 1e-7
    )
    mesh.vertices = mesh.vertices * extent + center
    emit({"type": "log", "msg": (
        "Restored the source mesh's size and position "
        f"(scale x{extent:.4g}).")})
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_count = int(vertices.shape[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))

    # Write a geometry-only OBJ twin next to the GLB for importers that only
    # read OBJ (the ZBrush plugin). Welding duplicate vertices and dropping
    # unreferenced ones mirrors the pipeline's OBJ staging and keeps ZBrush's
    # Import from raising its unused-vertex modal notes.
    obj_path = output_path.with_suffix(".obj")
    try:
        mesh.merge_vertices(merge_norm=True)
        mesh.remove_unreferenced_vertices()
        mesh.export(str(obj_path))
        emit({"type": "log", "msg": f"Saved {obj_path}"})
    except Exception as ex:  # noqa: BLE001 - the GLB is the primary result
        emit({"type": "log", "msg": (
            f"Could not write {obj_path.name}: {ex}"), "level": "warn"})
        obj_path = None

    state_path = (Path(cmd["shape512_state"])
                  if cmd.get("shape512_state") else None)
    if state_path is not None:
        try:
            _save_shape512_state(state_path, vertices, faces)
            emit({"type": "log", "msg": (
                f"Wrote {state_path.name}; Run 2nd stage can upscale this "
                "LATO.2 mesh.")})
        except Exception as ex:  # noqa: BLE001 - geometry output is primary
            emit({"type": "log", "msg": (
                f"Could not write {state_path.name}: {ex}"), "level": "warn"})

    emit({"type": "log", "msg": (
        f"LATO.2 finished: {vertex_count:,} vertices, {len(faces):,} faces. "
        "The result is normalized to the canonical cube."), "level": "ok"})
    emit({"type": "log", "msg": f"Saved {output_path}", "level": "ok"})
    return {
        "output": str(output_path),
        "obj": str(obj_path) if obj_path else "",
        "seed": seed,
        "vertex_count": vertex_count,
        "face_count": int(len(faces)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None,
                        help="UniForge model directory; its parent roots are "
                             "also searched for LATO.2 models.")
    args = parser.parse_args()

    # Heavy imports and all library chatter go to stderr so they cannot
    # corrupt the JSON protocol on stdout.
    sys.stdout = sys.stderr

    source = find_lato2_source()
    if source is None:
        emit({"type": "error", "msg": (
            "LATO.2 source checkout was not found. Expected "
            f"{SUBMODULES_DIR / 'LATO.2'} (or set UNIFORGE_LATO2_SOURCE).")})
        return 1
    sys.path.insert(0, str(source))
    # PS6_UI has its own regular ``utils`` package; it would shadow LATO.2's
    # namespace ``utils`` package (utils.load) even though LATO.2 is earlier
    # on sys.path. model_paths is already imported, so drop APP_DIR now.
    try:
        sys.path.remove(str(APP_DIR))
    except ValueError:
        pass

    model_hint = Path(args.model_dir) if args.model_dir else MODEL_DIR
    lato2_dir = find_lato2_dir(model_hint)
    missing = missing_lato2_assets(model_hint)
    if lato2_dir is None or missing:
        emit({"type": "error", "msg": (
            "LATO.2 models are not ready. Run get_models.bat first. Missing: "
            + "; ".join(missing))})
        return 1

    try:
        import torch
    except Exception as ex:
        emit({"type": "error", "msg": f"LATO.2 import failed: {ex}"})
        return 1

    if not torch.cuda.is_available():
        emit({"type": "error", "msg": "LATO.2 requires an NVIDIA CUDA GPU."})
        return 1

    props = torch.cuda.get_device_properties(0)
    emit({"type": "log", "msg": (
        f"LATO.2 GPU: {props.name} "
        f"({props.total_memory / (1024 ** 3):.1f} GB), compute capability "
        f"{props.major}.{props.minor}.")})
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
        if cmd.get("cmd") != "generate":
            emit({"type": "error",
                  "msg": f"Unknown command: {cmd.get('cmd')!r}"})
            continue

        device = torch.device("cuda")
        try:
            bundle = _load_models(lato2_dir, device)
        except Exception as ex:
            emit({"type": "error", "msg": f"LATO.2 model load failed: {ex}"})
            return 1
        emit({"type": "log", "msg": "LATO.2 model ready.", "level": "ok"})

        try:
            result = _generate(bundle, cmd, device)
        except Exception as ex:
            message = f"LATO.2 mesh generation failed: {ex}"
            if "out of memory" in message.lower():
                message += (" Close other GPU applications or lower the "
                            "target vertex count.")
            emit({"type": "error", "msg": message})
            return 1
        emit({"type": "done", **result})
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
