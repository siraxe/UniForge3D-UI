"""Sparse PTv3 mesh decoder."""

# Adapted from Pointcept / Point Transformer V3 under the MIT License.
# Modified for SPARC3D: the decoder half only, reading a latent grid instead of
# an encoded point cloud.

from __future__ import annotations

from typing import Sequence

from torch import nn

from sparc3d.ops import sparse_backend
from sparc3d.models.autoencoders.blocks import (
    ConvOut,
    SparseAttentionBlock,
    SparseResnetBlock,
    SparseZToF,
    Upsample,
)
from sparc3d.structures.point import Point
from sparc3d.structures.point_modules import PointModule, PointSequential

DECODE_PROGRESS_HOOK = None
"""Optional callable ``(step, total, label)`` invoked before each decoder stage
runs and once more with ``label == "activate"`` after the head. Inference-time
decodes at 1536 run for minutes with the GPU mostly idle in bounded chunks, so
host applications can install this to surface decode progress. Assign the
module attribute (``ptv3.DECODE_PROGRESS_HOOK = fn``) and set it back to
``None`` afterwards; exceptions from the hook are swallowed."""


class PTv3MeshDecoder(PointModule):
    """Decode a coarse latent point cloud back into per-corner SDF and deformation.

    A serialized attention stack mixes globally at the coarsest level, then residual
    stages double the resolution and prune children by predicted occupancy.
    """

    def __init__(
        self,
        out_channels: int = 32,
        order: str | Sequence[str] = ("z", "z-trans"),
        depths: Sequence[int] = (2, 2, 2, 2),
        channels: Sequence[int] = (128, 256, 512, 512),
        mid_depth: int = 2,
        mid_num_head: int = 32,
        z_channels: int = 32,
        patch_size: int = 48,
        mlp_ratio: float = 4,
    ):
        """
        Args:
            out_channels: Attribute channels emitted, on top of the eight sign logits.
            order: Space-filling curve orders; attention blocks cycle through them.
            depths: Residual blocks per stage.
            channels: Feature width per stage.
            mid_depth: Attention blocks at the coarsest level.
            mid_num_head: Attention heads at the coarsest level.
            z_channels: Latent channels consumed.
            patch_size: Points per attention window.
            mlp_ratio: Attention-block MLP expansion ratio.

        Raises:
            ValueError: ``depths`` and ``channels`` describe different stage counts.
        """
        super().__init__()

        self.num_stages = len(depths)
        self.order = [order] if isinstance(order, str) else list(order)

        if len(channels) != self.num_stages:
            raise ValueError(
                f"channels has {len(channels)} entries but depths describes "
                f"{self.num_stages} stages"
            )

        mid_ch = channels[-1]
        self.z_to_f = SparseZToF(z_channels, mid_ch)

        self.mid = PointSequential()
        self.mid.add(SparseResnetBlock(mid_ch, mid_ch), name="block_in")
        for i in range(mid_depth):
            self.mid.add(
                SparseAttentionBlock(
                    channels=mid_ch,
                    num_heads=mid_num_head,
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    order_index=i % len(self.order),
                ),
                name=f"block{i}",
            )
        self.mid.add(SparseResnetBlock(mid_ch, mid_ch), name="block_out")

        self.dec = PointSequential()
        for s in reversed(range(self.num_stages)):
            dec = PointSequential()
            block_in = channels[s]
            block_out = channels[s - 1] if s > 0 else channels[0]
            for i in range(depths[s]):
                dec.add(SparseResnetBlock(block_in, block_out), name=f"block{i}")
                block_in = block_out
            if s != 0:
                dec.add(Upsample(block_out), name="up")
            self.dec.add(module=dec, name=f"dec{s}")

        # Eight extra output channels carry the per-corner SDF sign logits.
        self.conv_out = ConvOut(block_out, 8 + out_channels)

    def _decode_stages(self) -> list[tuple[str, nn.Module]]:
        """Flat ``(label, module)`` list of the decode pass, in execution order."""
        stages: list[tuple[str, nn.Module]] = [("embed", self.z_to_f)]
        stages.extend(
            (f"mid.{name}", module)
            for name, module in self.mid._modules.items()
        )
        for stage_name, dec in self.dec._modules.items():
            stages.extend(
                (f"{stage_name}.{name}", module)
                for name, module in dec._modules.items()
            )
        stages.append(("head", self.conv_out))
        return stages

    def forward(self, point: Point) -> Point:
        """Decode latents into per-corner SDF signs, distances and deformations.

        Args:
            point: Sparse point carrying latent ``feat`` on its grid. Sparsified and
                serialized in place if the caller has not done so already.

        Returns:
            The same point, with ``sdf_sign`` set and ``feat`` holding the remaining
            per-corner attributes at the final resolution.
        """
        if not point.is_sparsified:
            point.sparsify()

        hook = DECODE_PROGRESS_HOOK
        stages = self._decode_stages()
        total = len(stages)
        for step, (label, module) in enumerate(stages, 1):
            if hook is not None:
                try:
                    hook(step, total, label)
                except Exception:
                    pass
            if isinstance(module, PointModule):
                point = module(point)
            elif sparse_backend.is_sparse_conv(module):
                point.sparse_conv_feat = module(point.sparse_conv_feat)
                point.feat = sparse_backend.get_feats(point.sparse_conv_feat)
            else:
                point.update_feat(module(point.feat))
            if label == "embed" and not point.is_serialized:
                # The serialized order must exist before the attention stack.
                point.serialization(order=self.order)

        if hook is not None:
            try:
                hook(total, total, "activate")
            except Exception:
                pass
        return point
