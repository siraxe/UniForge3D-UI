"""Array-level entry point for the UI: in-memory quad remeshing."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .cut import (augment_cut_for_singularities, complete_cut_for_field_consistency,
                  trace_separatrices, unwrap_field_branches)
from .curvature import compute_vertex_curvature, face_trust_and_direction
from .extract import extract_quads
from .integer import RemeshCancelled, solve_integer_uv
from .field import (_edge_frame_angles, smooth_cross_field,
                    vertex_singularity_indices)
from .integer import close_cone_skip
from .locator import binned_locator
from .mesh import TriMesh
from .param import solve_uv

MIN_QUADS = 50
MAX_QUADS = 150000

LogFn = Callable[[str, str], None]


def status() -> dict:
    try:
        import scipy.sparse
        import scipy.optimize
        ready = True
        error = ""
    except Exception as ex:
        ready = False
        error = str(ex)
    return {"ready": ready, "error": error}


def _auto_quads(n_faces: int) -> int:
    return int(min(MAX_QUADS, max(MIN_QUADS, n_faces // 2)))


def _triangulate(quads: np.ndarray, tris: np.ndarray) -> np.ndarray:
    out = []
    if len(tris):
        out.append(np.asarray(tris, dtype=np.int64))
    if len(quads):
        q = np.asarray(quads, dtype=np.int64)
        out.append(q[:, [0, 1, 2]])
        out.append(q[:, [0, 2, 3]])
    if not out:
        return np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(out, axis=0)


def remesh_arrays(vertices, faces, *, target_quads: int = 0,
                  symmetry: str = "", freeze_borders: bool = False,
                  var_density_ratio: float = 1.0,
                  log=None, progress=None, cancel_check=None) -> dict:
    def say(message: str, level: str = "info") -> None:
        if log is not None:
            log(message, level)

    def report(percent: int, stage: str) -> None:
        if cancel_check is not None and cancel_check():
            raise RemeshCancelled("Remesh cancelled")
        if progress is not None:
            progress(max(0, min(100, int(percent))), stage)

    started = time.perf_counter()
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if not len(v) or not len(f):
        raise ValueError("the input mesh has no triangles")
    if symmetry:
        say("Remesh: symmetry is not applied by the native remesher yet.",
            "warn")
    if freeze_borders:
        say("Remesh: freeze-borders is not applied by the native remesher "
            "yet.", "warn")

    quads_target = (int(target_quads) if target_quads and int(target_quads) > 0
                    else _auto_quads(len(f)))
    quads_target = int(var_density_ratio ** 2 * quads_target)
    quads_target = int(min(MAX_QUADS, max(MIN_QUADS, quads_target)))

    mesh = TriMesh(v, f)
    say(f"Remesh: {mesh.n_vertices:,} vertices, {mesh.n_faces:,} triangles, "
        f"target {quads_target:,} quads ...", "info")

    report(2, "curvature")
    k1, k2, d1 = compute_vertex_curvature(
        mesh, progress=lambda t: report(2 + 13.0 * t, "curvature"))
    alpha, trust, frame_u, frame_v = face_trust_and_direction(mesh, k1, k2, d1)
    report(15, "field solve")
    theta, transitions, _ = smooth_cross_field(mesh, alpha, trust,
                                               frame_u, frame_v)
    sing = vertex_singularity_indices(mesh, theta, frame_u, frame_v)
    sing_v = np.flatnonzero(sing)
    n_sing = int(np.sum(np.mod(sing, 4) != 0))
    say(f"Remesh: field solved ({n_sing} singularities) ...", "info")
    report(22, "cut graph")

    traced = trace_separatrices(
        mesh, theta, transitions, frame_u, frame_v, sing_v,
        progress=lambda t: report(22 + 18.0 * t, "cut graph"))
    cut = augment_cut_for_singularities(mesh, traced["cut_edges"], sing_v)
    cut = complete_cut_for_field_consistency(mesh, theta, frame_u, frame_v,
                                             cut)
    fl, fr, m4, inner = _edge_frame_angles(mesh, frame_u, frame_v)
    thg = unwrap_field_branches(mesh, theta, frame_u, frame_v, cut)
    kmap = np.zeros(mesh.n_edges, dtype=int)
    eids = np.flatnonzero(inner)
    kmap[eids] = np.mod(np.round(
        (thg[fl] - thg[fr] - 0.25 * m4) / (np.pi / 2)).astype(int), 4)
    seam_k = {int(e): int(kmap[e]) for e in cut}

    report(40, "UV solve")
    uvres = solve_uv(mesh, theta, frame_u, frame_v, cut,
                     target_quads=quads_target, seam_k=seam_k)
    say(f"Remesh: UV layout ({uvres['n_patches']} patches, "
        f"{len(uvres['cut_pairs'])} seams) ...", "info")

    report(55, "integer offsets")
    snap_skip = close_cone_skip(mesh, uvres, sing_v)

    def integer_report(frac, msg=""):
        # solve_integer_uv reports 0..1 across its trials; map onto the
        # 55..65% band of the whole pipeline
        report(55 + 10.0 * max(0.0, min(1.0, float(frac))),
               f"integer offsets ({msg})" if msg else "integer offsets")

    ires = solve_integer_uv(mesh, uvres, snap_skip=snap_skip,
                            log=say, report=integer_report,
                            cancel_check=cancel_check)
    report(65, "quad extraction")
    quads = extract_quads(
        mesh, ires, seed_poles=False, insert_poles=True,
        progress=lambda t: report(65 + 32.0 * t, "quad extraction"),
        locate=binned_locator(
            ires["u_complex"][ires["corner_uv"]],
            np.asarray(ires["patch_of_face"], dtype=np.int64),
            int(ires["n_patches"]), *mesh.edge_faces(),
            mesh.face_edge_ids,
            {int(p[0]): (1j ** int(kk), complex(nn)) for p, kk, nn in
             zip(ires["cut_pairs"], ires["seam_k"],
                 ires["integer_offsets"])},
            V=mesh.V, F=mesh.F,
            max_dist=4.0 * float(ires["h"]) if ires.get("h") else None))
    if quads["n_fail"] > max(16, quads["n_cells"] // 32):
        say("Remesh: extraction struggled; retrying with snapped pole "
            "offsets ...", "warn")
        ires = solve_integer_uv(mesh, uvres, log=say, report=integer_report,
                                cancel_check=cancel_check)
        report(65, "quad extraction (retry)")
        quads = extract_quads(
            mesh, ires, seed_poles=False, insert_poles=True,
            progress=lambda t: report(65 + 32.0 * t,
                                      "quad extraction (retry)"))

    out_v = np.asarray(quads["vertices"], dtype=np.float64)
    out_f = _triangulate(quads["quads"], quads["tris"])
    report(100, "done")
    seconds = time.perf_counter() - started
    say(f"Remesh done: {quads['n_quads']:,} quads (+{quads['n_tris']:,} "
        f"tris), {len(out_v):,} vertices in {seconds:.1f}s; result kept in "
        "memory (not saved).", "ok")

    return {
        "mode_label": "Remesh",
        "vertices": out_v,
        "faces": out_f,
        "vertex_count": int(len(out_v)),
        "face_count": int(len(out_f)),
        "input_vertex_count": int(len(v)),
        "input_face_count": int(len(f)),
        "target_quads": quads_target,
        "symmetry": symmetry or "none",
        "n_quads": int(quads["n_quads"]),
        "n_tris": int(quads["n_tris"]),
        "flips": int(ires.get("flips", 0)),
        "seconds": round(seconds, 2),
    }
