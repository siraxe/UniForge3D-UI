"""Mesh edit operations.

Panel tools (loaded via tools._loader): subdivision, hole filling, face
deletion and scale-back.  The 3D-view brush stack (brush_smooth,
history, smooth_operator) lives here too; the UI imports those submodules
directly, so the package __init__ keeps them out of the panel-tool API."""

from .close_holes import close_holes_arrays
from .delete_faces import delete_faces_arrays
from .mirror_merge import mirror_merge_arrays
from .scale_back import (
    radial_scale_back_info,
    scale_back_arrays,
    scale_back_info,
)
from .subdivide import subdivide_arrays
__version__ = "0.1.0"
