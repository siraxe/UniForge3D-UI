"""Headless smoke test - no Qt window or GL context needed.

Verifies that a generated prediction.glb can be brought into VTK through the
same path ModelViewer.load() uses (direct import, or trimesh re-export
fallback for GLBs that require extensions VTK does not know, e.g.
EXT_texture_webp on outputs made before extension_webp=False).

Run: .venv/Scripts/python.exe PS6_UI/smoke_test.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vtk

glbs = sorted((ROOT / "output_dir").glob("*/prediction.glb"))
assert glbs, "no prediction.glb found - generate something first"
glb = glbs[0]
print("testing:", glb)

import viewer as viewer_mod


class _FakeWindow:
    """Stand-in exposing just what the import path touches."""

    def Render(self):
        pass


class _FakeViewer:
    """Runs ModelViewer's GLB logic against a bare vtkRenderer."""

    def __init__(self):
        self._ren = vtk.vtkRenderer()

    _load_gltf = viewer_mod.ModelViewer._load_gltf
    _adopt = viewer_mod.ModelViewer._adopt

    def clear(self):
        self._ren.RemoveAllViewProps()

    render_now = lambda self: None


v = _FakeViewer()
direct = viewer_mod.ModelViewer._collect_importer_actors(
    viewer_mod.ModelViewer._gltf_importer(glb))
print("direct VTK import:", bool(direct), "| actors:", len(direct))
if direct:
    actors = direct
else:
    # Same path ModelViewer._load_gltf uses: trimesh re-export fallback.
    actors = viewer_mod.ModelViewer._read_gltf_actors(glb)
    print("trimesh fallback: OK | actors:", len(actors))
assert actors, "GLB import failed (direct and trimesh fallback)"
for actor in actors:
    v._ren.AddActor(actor)
print("SMOKE OK")
