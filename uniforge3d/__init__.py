"""UniForge3D multi-view image-to-3D inference."""

# Must run before transformers/accelerate import torchao.
from . import _compat  # noqa: F401
from .pipelines import UniForge3DImageTo3DPipeline

__all__ = ["UniForge3DImageTo3DPipeline"]
