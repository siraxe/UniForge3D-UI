#!/usr/bin/env python3
"""Adapter turning an image into a coarse mesh with Hunyuan3D-2 mini.

SPARC3D refines a coarse mesh; it does not generate one. This adapter is a
demonstration of one way to produce that input, not part of the library and not a
declared dependency. Any generator that returns a triangle ``trimesh.Trimesh``
in a roughly unit-scale frame works just as well -- swap this file out.

Requires `hy3dgen` to be importable and the Hunyuan3D-2 mini weights to be
available; see https://github.com/Tencent-Hunyuan/Hunyuan3D-2 for both. Neither is
installed by ``sparc3d[demo]``. Its ``shapegen`` modules also import einops,
diffusers, opencv-python-headless, pymeshlab, scikit-image, tqdm and pyyaml, which
resolve NumPy 2.x unless constrained -- SPARC3D requires ``numpy<2``, so install
them under ``PIP_CONSTRAINT``. ``docs/installation.md`` spells this out.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image

HUNYUAN_REPO = "tencent/Hunyuan3D-2mini"
"""Hub repository holding the demo coarse generator."""

HUNYUAN_SUBFOLDER = "hunyuan3d-dit-v2-mini-turbo"
"""Turbo variant inside :data:`HUNYUAN_REPO`, chosen for its short step schedule."""

COARSE_STEPS = 30
"""Denoising steps; the turbo variant is distilled for roughly this many."""

COARSE_OCTREE_RESOLUTION = 256
"""Octree resolution of the coarse extraction. SPARC3D re-voxelizes to 64 anyway."""


def load_coarse_generator(
    model: str = HUNYUAN_REPO,
    subfolder: str = HUNYUAN_SUBFOLDER,
    device: str = "cuda",
) -> Any:
    """Load the Hunyuan3D coarse generator.

    Args:
        model: Local path or Hub repository holding the coarse generator.
        subfolder: Variant subfolder within ``model``.
        device: Device to place the generator on.

    Returns:
        A callable Hunyuan3D pipeline.

    Raises:
        ImportError: ``hy3dgen`` is not installed.
    """
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    except ImportError as error:
        raise ImportError(
            "the coarse-stage demo needs Hunyuan3D-2; install hy3dgen from "
            "https://github.com/Tencent-Hunyuan/Hunyuan3D-2. It is a demonstration "
            "of one coarse generator and is not a SPARC3D dependency."
        ) from error

    return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model,
        subfolder=subfolder,
        use_safetensors=True,
        device=device,
    )


def generate_coarse_mesh(generator: Any, image: Image.Image, seed: int = 0) -> Any:
    """Generate one coarse mesh from ``image``.

    Args:
        generator: Pipeline returned by :func:`load_coarse_generator`.
        image: Source image. Pass it with its alpha channel intact if it has one;
            Hunyuan3D does its own framing and background handling.
        seed: Seed for the coarse sampler.

    Returns:
        Coarse ``trimesh.Trimesh`` ready to hand to ``Sparc3DPipeline``.
    """
    return generator(
        image=image,
        num_inference_steps=COARSE_STEPS,
        octree_resolution=COARSE_OCTREE_RESOLUTION,
        generator=torch.manual_seed(seed),
        output_type="trimesh",
    )[0]


def parse_args() -> argparse.Namespace:
    """Parse coarse-generation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=HUNYUAN_REPO)
    parser.add_argument("--subfolder", default=HUNYUAN_SUBFOLDER)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    """Generate one coarse mesh and export it."""
    args = parse_args()
    generator = load_coarse_generator(args.model, args.subfolder, args.device)
    mesh = generate_coarse_mesh(generator, Image.open(args.image), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
