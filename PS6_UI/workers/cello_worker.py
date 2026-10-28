"""Standalone CelloCut watertight remeshing worker.

Spawned by PS6_UI/workers/controllers.py for **Run CelloCut (watertight)**. It loads the
compiled CelloCut modules (torchcumesh2sdf + cppmodules), runs the
constructive watertight remeshing on the saved 1st-stage mesh, exports a
geometry-only GLB, then exits. CelloCut is not a learned model - it needs no
network weights - but it does use the GPU for the 512^3 UDF rasterization.

The protocol is the same JSON-lines format used by worker.py / hi3d_worker.py:

  stdin:  {"cmd": "watertight", "mesh": "...", "output": "...",
           "shape512_state": "...",            # optional; regenerated too
           "voxel_res": 512, "decimate_ratio": 0.95, "lambda_fill": 20}
  stdout: {"type": "ready"}
          {"type": "log", "msg": ..., "level": "info"|"ok"|"warn"|"error"}
          {"type": "done", "output": "...", "watertight": true}
          {"type": "error", "msg": "..."}
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent    # PS6_UI
ROOT = APP_DIR.parent
CELLOCUT_DIR = ROOT / "submodules" / "CelloCut"
sys.path.insert(0, str(CELLOCUT_DIR))
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Deterministic cuBLAS kernel selection (must be set before torch/CUDA
# init); see worker.py for why this matters for same-seed re-runs.
import os as _os

_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_real_stdout = sys.stdout


def emit(obj) -> None:
    """Write one protocol line; if the UI disappeared (broken pipe), quit."""
    try:
        _real_stdout.write(json.dumps(obj) + "\n")
        _real_stdout.flush()
    except OSError:
        import os

        os._exit(1)


def _load_input_mesh(path: Path, tmp_dir: Path) -> Path:
    """CelloCut's remesh expects an OBJ file; convert anything else (GLB,
    STL, ...) to a temporary OBJ with trimesh first."""
    import trimesh

    if path.suffix.lower() == ".obj":
        return path
    loaded = trimesh.load(str(path), force="mesh", process=False)
    obj_path = tmp_dir / (path.stem + ".obj")
    loaded.export(str(obj_path))
    return obj_path


def _save_shape512_state(path: Path, mesh) -> None:
    """Write the run_stage2 checkpoint, mirroring the pipeline's
    ``_save_shape512_state`` (format 1, float32 vertices, int64 faces)."""
    import torch

    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 1,
        "vertices": torch.from_numpy(
            mesh.vertices.astype("float32", copy=False)),
        "faces": torch.from_numpy(mesh.faces.astype("int64", copy=False)),
    }
    temporary = target.with_name(target.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _run_watertight(cmd: dict) -> None:
    import numpy as np
    import trimesh

    from model import CelloCut

    mesh_path = Path(cmd["mesh"])
    output_path = Path(cmd["output"])
    state_path = (Path(cmd["shape512_state"])
                  if cmd.get("shape512_state") else None)
    voxel_res = int(cmd.get("voxel_res") or 512)
    decimate_ratio = float(cmd.get("decimate_ratio") or 0.95)
    lambda_fill = int(cmd.get("lambda_fill") or 20)

    if not mesh_path.is_file():
        raise FileNotFoundError(f"input mesh not found: {mesh_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    emit({"type": "log", "msg": (
        f"CelloCut settings: voxel_res={voxel_res}, "
        f"decimate_ratio={decimate_ratio}, lambda_fill={lambda_fill}")})

    with tempfile.TemporaryDirectory(prefix="cellocut_") as tmp_dir:
        obj_path = _load_input_mesh(mesh_path, Path(tmp_dir))
        if obj_path != mesh_path:
            emit({"type": "log", "msg": f"Converted {mesh_path.name} -> OBJ."})

        import cppmodules as cpp

        model = CelloCut(voxel_res, lambda_fill)

        emit({"type": "log", "msg": "Computing the thickened UDF proxy ..."})
        thick_v, thick_f, sdf = model.thick_mesh(obj_path, decimate_ratio)

        emit({"type": "log", "msg": "Tetrahedralizing the proxy ..."})
        tet_verts, tets = cpp.tetrahedralize(thick_v)
        internal_faces, outer_faces = cpp.build_adjacency_map(tet_verts, tets)
        out_tets = cpp.search_out_tet(internal_faces, int(tets.shape[0]))
        sdf_np = (sdf.detach().cpu().numpy().astype(np.float64).reshape(-1))
        initial_labels = cpp.init_labels(
            internal_faces, tet_verts, tets, sdf_np, voxel_res,
            model.origin_bbox_min, model.origin_bbox_max)

        emit({"type": "log", "msg": "Solving the graph-cut labeling ..."})
        new_labels = model.graph_cut(internal_faces, initial_labels, out_tets)

        emit({"type": "log", "msg": "Extracting the watertight surface ..."})
        gc_v, gc_f = cpp.surface_extraction(
            internal_faces, new_labels, tet_verts, tets, outer_faces)

        emit({"type": "log", "msg": "Running the thinning cleanup pass ..."})
        thin_v, thin_f = model.thin_mesh(gc_v, gc_f)

    repaired = trimesh.Trimesh(thin_v, thin_f)
    repaired.export(str(output_path))
    watertight = bool(repaired.is_watertight)

    if state_path is not None:
        # Regenerate the Run-2nd-stage checkpoint from the repaired mesh so
        # the 1024 upscale continues from the fixed geometry.
        try:
            _save_shape512_state(state_path, repaired)
            emit({"type": "log", "msg": f"Updated {state_path.name} from "
                                        "the repaired mesh."})
        except Exception as ex:
            emit({"type": "log", "msg": f"Could not update "
                                        f"{state_path.name}: {ex}",
                  "level": "warn"})

    emit({"type": "log", "msg": (
        f"CelloCut finished: {len(repaired.faces):,} faces, "
        f"{len(repaired.vertices):,} vertices, watertight={watertight}."),
        "level": "ok"})
    emit({"type": "log", "msg": f"Saved {output_path}", "level": "ok"})
    emit({"type": "done", "output": str(output_path),
          "watertight": watertight})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction,
                        default=True, help="accepted for symmetry; unused")
    args = parser.parse_args()

    # Heavy imports and all library chatter go to stderr so they cannot
    # corrupt the JSON protocol on stdout.
    sys.stdout = sys.stderr

    try:
        import torch

        import cppmodules
        import diso
        import maxflow
        import torchcumesh2sdf
    except Exception as ex:
        emit({"type": "error", "msg": (
            "CelloCut modules failed to import. Run install.bat to get the "
            f"CelloCut wheels: {ex}")})
        return 1

    if not torch.cuda.is_available():
        emit({"type": "error", "msg": "CelloCut requires an NVIDIA CUDA GPU."})
        return 1

    props = torch.cuda.get_device_properties(0)
    emit({"type": "log", "msg": (
        f"CelloCut GPU: {props.name} "
        f"({props.total_memory / (1024 ** 3):.1f} GB).")})
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
        if cmd.get("cmd") != "watertight":
            emit({"type": "error",
                  "msg": f"Unknown command: {cmd.get('cmd')!r}"})
            continue
        try:
            _run_watertight(cmd)
        except Exception as ex:
            message = f"CelloCut remeshing failed: {ex}"
            if "out of memory" in message.lower():
                message += (" Close other GPU applications or lower the "
                            "voxel resolution.")
            emit({"type": "error", "msg": message})
            return 1
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
