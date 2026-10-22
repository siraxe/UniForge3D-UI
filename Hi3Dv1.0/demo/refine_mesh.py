#!/usr/bin/env python3
"""Command-line image-conditioned refinement of an existing coarse mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh
from PIL import Image

from sparc3d import Sparc3DPipeline


def parse_args() -> argparse.Namespace:
    """Parse mesh-refinement arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local artifact path or Hub model ID.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--coarse-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resolution",
        type=int,
        choices=(512, 1024, 1536),
        default=1024,
    )
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequential-cpu-offload", action="store_true")
    parser.add_argument("--remove-background", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Hugging Face cache directory, meaning the one that holds the "
        "models--* trees. That is $HF_HOME/hub, not $HF_HOME.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--dino-model",
        help="DINOv2 repository ID or local directory. The conditioning encoders "
        "are not part of the SPARC3D artifact; point this at a local copy to run "
        "without network access.",
    )
    parser.add_argument("--clip-model", help="CLIP repository ID or local directory.")
    return parser.parse_args()


def main() -> int:
    """Load the pipeline, refine one mesh, and export the result."""
    args = parse_args()
    image = Image.open(args.image)
    if args.remove_background:
        from sparc3d.utils.image import remove_background

        image = remove_background(image)
    coarse_mesh = trimesh.load(args.coarse_mesh, force="mesh", process=False)
    pipeline = Sparc3DPipeline.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        dino_model_id=args.dino_model,
        clip_model_id=args.clip_model,
    )
    if args.sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload(args.device)
    else:
        pipeline.to(args.device)
    result = pipeline(
        image,
        coarse_mesh,
        resolution=args.resolution,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.mesh.export(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
