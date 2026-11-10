"""CLI prototype for the UltraTex engine (M2 gate).

    python -m tools.ultratex --mesh mesh.glb --reference image.png
        [--mask mask.png] [--seed 1] [--steps 8] [--render 2048]
        [--bake 4096] [--no-refine] [--no-second-rig] [--no-pbr]
        [--no-distill] [--model-dir models/Comfy-Org/UltraTex]

Writes the albedo (+ metallic/roughness/confidence) atlases and the
per-view renders to ``<mesh>_ultratex/`` next to the mesh.
"""

from __future__ import annotations

import argparse
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m tools.ultratex",
        description="Self-contained UltraTex texturing (FLUX.2 Klein 9B int8)")
    ap.add_argument("--mesh", required=True, help="GLB with TEXCOORD_0 UVs")
    ap.add_argument("--reference", required=True, help="reference image")
    ap.add_argument("--mask", default=None, help="reference object mask (white=object)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--render", type=int, default=2048, dest="render_size")
    ap.add_argument("--bake", type=int, default=4096, dest="bake_size")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--no-second-rig", action="store_true")
    ap.add_argument("--no-pbr", action="store_true")
    ap.add_argument("--no-distill", action="store_true")
    ap.add_argument("--model-dir", default="models/Comfy-Org/UltraTex")
    args = ap.parse_args(argv)

    from . import UltraTexJob, run

    def log(msg, level="info"):
        print(f"[ultratex][{level}] {msg}", flush=True)

    job = UltraTexJob(
        mesh_path=str(args.mesh),
        reference_image=str(args.reference),
        reference_mask=args.mask,
        seed=args.seed,
        steps=args.steps,
        use_distill_lora=not args.no_distill,
        use_refine=not args.no_refine,
        use_second_rig=not args.no_second_rig,
        use_pbr=not args.no_pbr,
        render_size=args.render_size,
        bake_size=args.bake_size,
        model_dir=args.model_dir,
    )
    started = time.perf_counter()
    job = run(job, log=log)
    took = time.perf_counter() - started
    print(f"[ultratex] done in {took:.1f}s")
    for key, val in job.stats.items():
        print(f"[ultratex]   {key}: {val}")
    for name in ("albedo_atlas", "metallic_atlas", "roughness_atlas",
                 "confidence_atlas"):
        if getattr(job, name):
            print(f"[ultratex]   {name}: {getattr(job, name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
