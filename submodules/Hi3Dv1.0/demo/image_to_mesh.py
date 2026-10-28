#!/usr/bin/env python3
"""End-to-end image-to-mesh demo: coarse generation, then SPARC3D refinement.

The coarse stage is Hunyuan3D-2 mini standing in for "any coarse generator"; see
``coarse_stage.py``. Only the refinement half is SPARC3D. Run ``refine_mesh.py``
instead when you already have a coarse mesh.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from sparc3d import Sparc3DPipeline

from coarse_stage import generate_coarse_mesh, load_coarse_generator


def parse_args() -> argparse.Namespace:
    """Parse end-to-end generation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="SPARC3D artifact directory or Hub model ID.",
    )
    parser.add_argument("--coarse-model", default=None, help="Hunyuan3D path or ID.")
    parser.add_argument(
        "--coarse-subfolder",
        default=None,
        help="Variant subfolder inside --coarse-model.",
    )
    parser.add_argument(
        "--coarse-output",
        type=Path,
        help="Optional path to also save the intermediate coarse mesh to.",
    )
    parser.add_argument("--resolution", type=int, choices=(512, 1024, 1536), default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequential-cpu-offload", action="store_true")
    parser.add_argument(
        "--remove-background",
        action="store_true",
        help="Run rembg first. Needed for photos; skip it for cut-out renders.",
    )
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
    """Generate a coarse mesh, refine it, and export the result."""
    args = parse_args()

    image = Image.open(args.image)
    if args.remove_background:
        from sparc3d.utils.image import remove_background

        image = remove_background(image)

    coarse_kwargs = {"device": args.device}
    if args.coarse_model is not None:
        coarse_kwargs["model"] = args.coarse_model
    if args.coarse_subfolder is not None:
        coarse_kwargs["subfolder"] = args.coarse_subfolder
    coarse_generator = load_coarse_generator(**coarse_kwargs)
    coarse_mesh = generate_coarse_mesh(coarse_generator, image, args.seed)
    if args.coarse_output is not None:
        args.coarse_output.parent.mkdir(parents=True, exist_ok=True)
        coarse_mesh.export(args.coarse_output)

    # Both stages want the whole GPU at 1024 and above, and the coarse generator is
    # finished with by now.
    coarse_generator.to("cpu")

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
    print(f"wrote {args.output} at {result.resolution}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
