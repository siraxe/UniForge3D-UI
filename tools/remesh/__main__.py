"""Stage CLI: run pipeline stages and dump diagnostics."""
import argparse
import sys

import numpy as np

from .mesh import TriMesh
from .curvature import compute_vertex_curvature, face_trust_and_direction
from .field import (smooth_cross_field, vertex_singularity_indices,
                    _edge_frame_angles)
from .cut import (trace_separatrices, augment_cut_for_singularities,
                  complete_cut_for_field_consistency, unwrap_field_branches)
from .param import solve_uv
from .integer import solve_integer_uv, close_cone_skip
from .extract import extract_quads, write_quad_obj


def main(argv=None):
    ap = argparse.ArgumentParser(prog="remesher")
    ap.add_argument("input", help="input .obj (triangulated)")
    ap.add_argument("--stages", default="curvature,field",
                    help="comma list: curvature,field,uv,integer,extract")
    ap.add_argument("--num-quad", type=int, default=1000)
    ap.add_argument("--out", default=None, help="diagnostic .npz output")
    ap.add_argument("--quad-out", default=None, help="output quad .obj")
    ap.add_argument("--seed-poles", action="store_true",
                    help="experimental GridPole cell seeding")
    ap.add_argument("--no-grid-poles", action="store_true",
                    help="skip GridPole fan insertion")
    args = ap.parse_args(argv)
    stages = set(s.strip() for s in args.stages.split(",") if s.strip())
    need_field = bool(stages & {"field", "uv", "integer", "extract"})
    need_uv = bool(stages & {"uv", "integer", "extract"})

    mesh = TriMesh.load_obj(args.input)
    print(f"mesh: {mesh.n_vertices} verts, {mesh.n_faces} tris, "
          f"{mesh.n_edges} edges")

    result = {}
    if stages & {"curvature", "uv", "integer", "extract"} or need_field:
        k1, k2, d1 = compute_vertex_curvature(mesh)
        print(f"curvature: k range [{np.nanmin(k1):.3g},{np.nanmax(k2):.3g}]")
        result.update(k1=k1, k2=k2, d1=d1)
        alpha, trust, u, v = face_trust_and_direction(mesh, k1, k2, d1)
        result.update(alpha=alpha, trust=trust)

    if need_field:
        theta, trans, ufield = smooth_cross_field(
            mesh, result["alpha"], result["trust"], u, v)
        sing = vertex_singularity_indices(mesh, theta, u, v)
        nsing = int(np.sum(np.mod(sing, 4) != 0))
        print(f"field: solved; singular vertices: {nsing} "
              f"({100.0*nsing/mesh.n_vertices:.2f}%)")
        result.update(theta=theta, transitions=trans, singularity=sing,
                      frame_u=u, frame_v=v)

    if need_uv:
        singv = np.flatnonzero(sing)
        tra = trace_separatrices(mesh, theta, trans, u, v, singv)
        cut = augment_cut_for_singularities(mesh, tra["cut_edges"], singv)
        nbase = len(cut)
        cut = complete_cut_for_field_consistency(mesh, theta, u, v, cut)
        print(f"cut: {len(cut)} seams (+{len(cut) - nbase} for "
              f"branch consistency)")
        fl, fr, m4, inner = _edge_frame_angles(mesh, u, v)
        thg = unwrap_field_branches(mesh, theta, u, v, cut)
        eids = np.flatnonzero(inner)
        kbase = np.zeros(mesh.n_edges, dtype=int)
        kbase[eids] = np.mod(np.round(
            (thg[fl] - thg[fr] - 0.25 * m4) / (np.pi / 2)).astype(int), 4)
        kmap = {int(e): int(kbase[e]) for e in cut}
        uvres = solve_uv(mesh, theta, u, v, cut, target_quads=args.num_quad,
                         seam_k=kmap)
        print(f"uv: patches {uvres['n_patches']} seams {len(uvres['cut_pairs'])} "
              f"h {uvres['h']:.4g} flips {uvres['flips']}")
        snap_skip = close_cone_skip(mesh, uvres, singv)
        if snap_skip:
            print(f"cones: unsnapped {sorted(snap_skip)}")
        result.update(cut=np.array(sorted(cut)), h=uvres["h"],
                      seam_k=uvres["seam_k"],
                      snap_skip=np.array(sorted(snap_skip), dtype=int))

    if stages & {"integer", "extract"}:
        uvres_cont = uvres
        iout = solve_integer_uv(mesh, uvres, snap_skip=snap_skip)
        print(f"integer: flips {iout['flips']} min_area "
              f"{iout['min_uv_area']:.3g} seam_resid "
              f"{iout['seam_int_resid']:.2g} dev "
              f"{np.median(iout['round_resid_lattice']):.3g}")
        uvres = iout
        result.update(integer_offsets=iout["integer_offsets"],
                      u_complex=iout["u_complex"])

    if "extract" in stages:
        q = extract_quads(mesh, uvres, seed_poles=args.seed_poles,
                          insert_poles=not args.no_grid_poles)
        if snap_skip and (q["n_tris"] > 0 or
                          any(k != 2 for k in q["edge_incidence"])):
            print("cones: fallback to snapped poles")
            iout = solve_integer_uv(mesh, uvres_cont)
            uvres = iout
            result.update(integer_offsets=iout["integer_offsets"],
                          u_complex=iout["u_complex"])
            q = extract_quads(mesh, uvres, seed_poles=args.seed_poles,
                              insert_poles=not args.no_grid_poles)
        print(f"extract: {q['n_quads']} quads, {q['n_tris']} tris, "
              f"{q['n_vertices']} verts, fail {q['n_fail']}, "
              f"fill {q['n_fill']}, merge {q['n_merge']}, "
              f"repair {q['n_repair']}, pair {q['n_pair']}, "
              f"pole {q['n_pole']}, "
              f"incidence {q['edge_incidence']}, euler {q['euler']}")
        if args.quad_out:
            write_quad_obj(args.quad_out, q["vertices"], q["quads"],
                           q["tris"])
            print(f"quads: {args.quad_out}")

    if args.out:
        np.savez_compressed(args.out, **result)
        print(f"diagnostics: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
