# UniForge3D UI — Windows Desktop Edition

A Windows-focused packaging of [UniForge3D](https://github.com/xibushu/UniForge3D) with a
PySide6 desktop UI (`PS6_UI/`). Generates textured 3D assets (GLB) from sparse, unposed
multi-view images. Apache-2.0, same as upstream.

## Requirements

- Windows 10/11, NVIDIA GPU (CUDA 13.0 wheels)
- [Python 3.12 x64](https://www.python.org/downloads/) (`py -3.12` must work)
- The prebuilt `.whl/` folder (included) — no native builds needed, but the wheels can also be built from source if you prefer. It includes custom `cumm-cu130` / `spconv-cu130` builds for Hi3D; the PyPI spconv wheels have no Blackwell (sm_120) kernels.

## Setup

<table>
  <tr>
    <th width="25%">Step</th>
    <th width="75%">UI</th>
  </tr>
  <tr>
    <td valign="top"><code>install.bat</code> — create <code>.venv</code>, install pinned requirements + local wheels<br><br><code>get_models.bat</code> — download model weights into <code>models\&lt;owner&gt;\&lt;repo&gt;\</code><br><br><code>start_UI.bat</code> — launch the desktop UI</td>
    <td valign="top"><img src="assets/screen.jpg" alt="screenshot"></td>
  </tr>
</table>

Run in that order. `get_models.bat` fetches from public Hugging Face and ModelScope
mirrors and is resumable — re-run it if a download is interrupted. `start_UI.bat`
auto-detects the model tree and applies VRAM safeguards.

### Hi3D 1536 refinement (optional)

The **Upscale settings** panel below the UniForge3D settings is always
visible but stays disabled until an **Early stop** run has saved a
Shape-512 mesh. **Run Hi3D 1536** refines the saved `prediction_512.glb`
with Hi3D v1.0, conditioned on the image selected as **REF** (falling back to
that run's first input image). The dropdown next to the button picks which
passes of the refinement ladder run:

- **1-3 pass** runs the full Hi3D ladder: structures 64 → 128 → 192, output
  1536.
- **2-3 pass** voxelizes your saved mesh at structure 128 and runs passes 2-3,
  output 1536.
- **1 pass**, **2 pass**, **3 pass** run a single rung. Each pass decodes its
  own mesh: pass 1 outputs 512, pass 2 outputs 1024, pass 3 outputs 1536. A
  single-pass run at 2 or 3 voxelizes your saved mesh at that structure and
  uses it directly; the final 1536 decode has the same memory cost as the
  full run.

The result is saved as `prediction_hi3d_<output resolution>.glb`. The decode
phase (the long silent tail after the last sampling step) reports per-stage
progress in the log.

A settings row above the button exposes three experiment knobs:

- **Steps** — sampling steps per pass (default 25, the validated value).
  Fewer steps (e.g. 12–15) sample proportionally faster with somewhat
  rougher geometry.
- **Voxel budget** — scale for the occupied-voxel budgets per pass (default
  100% = 80k/90k/110k). Lower values mean fewer tokens, so sampling and the
  memory-heavy decode both get faster and smaller at the cost of fine
  detail. The automatic OOM retry already drops to 65%.
- **Full GPU residency** — OFF (default) Hi3D follows the global
  **Low VRAM** switch: with it on, layers are streamed on/off the GPU per
  forward pass, so GPU utilization looks low and the run is slow. ON keeps
  all Hi3D weights resident on the GPU — much faster, but the 1536 decode
  peaks several GB higher and can OOM on 24–32 GB cards (one automatic
  retry at 65% budgets).

- **Run 2nd stage** stays in the same panel: it runs the UniForge Shape-1024
  upscale from the saved Shape-512 mesh.

Integration notes:

- Keep the Hi3D v1.0 source checkout at `Hi3Dv1.0/` (or point
  `UNIFORGE_HI3D_SOURCE` at it). It is run directly because its package
  metadata pins Python 3.10 while this app uses 3.12.
- `get_models.bat` also downloads Hi3D v1.0, DINOv2-large and CLIP ViT-L/14
  (~10.7 GB total) into `models/`.
- The result is geometry only and is written next to the preview as
  `prediction_hi3d_1536.glb`. Hi3D unloads as soon as refinement finishes.
- After refinement, **Show low / Show high** compares the saved Shape-512 mesh
  with the Hi3D result in place, keeping the current camera and material.
  Hi3D normalizes its output, so the viewer aligns it to the Shape-512 bounds
  for the comparison; the saved GLB keeps Hi3D's own scale.
- Hi3D's decoder in this checkout is patched for lower peak VRAM: group norms,
  Swish and the unpooling projection run in bounded chunks, residual adds are
  in-place during inference, and decoder caches/features are released before
  mesh extraction and hole filling.
- 1536 is still GPU-memory intensive. Low VRAM uses sequential CPU offload,
  but a 24–32 GB card may remain limited (Hi3D's 1536 mode was validated on a
  96 GB GPU).

### CelloCut watertight repair (optional)

After an **Early stop** run, **Run CelloCut (watertight)** in the
**Upscale settings** panel repairs the saved `prediction_512.glb` with
CelloCut (vendored under `CelloCut/`): it rebuilds the mesh from a 512³
unsigned-distance volume, tetrahedralizes it, and solves a graph-cut cell
labeling, which yields a strictly watertight solid (holes, self-intersections
and non-manifold edges included). Notes:

- Constructive method — **no model download**; the GPU is only used for the
  512³ UDF rasterization. The two compiled extensions ship as local wheels in
  `.whl/` (`torchcumesh2sdf`, `cppmodules`), installed by `install.bat`.
- The repaired mesh **replaces `prediction_512.glb` in place** and the 3D
  view reloads it; the untouched original is kept once as
  `prediction_512_original.glb`. Its `prediction_512.shape512.pt` state is
  regenerated from the repaired mesh (original kept as
  `prediction_512_original.shape512.pt`), so **Run 2nd stage** and the Hi3D
  buttons then start from the repaired geometry, and
  **Show low / Show high** compares original vs watertight.

## Viewer

The 3D view has **Backface culling** (toolbar checkbox, on by default):
polygons facing away from the camera are hidden. A closed, consistently
oriented mesh looks unchanged, while double-sided sheets, flipped patches and
holes become see-through — useful for checking whether a result is watertight
before running CelloCut or the refinements.

## Credits & License

This is a third-party UI/packaging fork. All model and pipeline code comes from the
original **[UniForge3D](https://github.com/xibushu/UniForge3D)** repo, which builds on
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) and
[VGGT](https://github.com/facebookresearch/vggt):

- Upstream repo: [github.com/xibushu/UniForge3D](https://github.com/xibushu/UniForge3D)
- Project page: [xishuxishu.github.io/uniforge3d-project](https://xishuxishu.github.io/uniforge3d-project/)
- Models: [huggingface.co/xibushu/UniForge3D](https://huggingface.co/xibushu/UniForge3D)
- Paper: [arXiv:2610.15357](https://arxiv.org/abs/2610.15357)

### License

- **Everything else** (this UI, the packaging, and upstream UniForge3D):
  [Apache License 2.0](LICENSE), matching the upstream project.
- **Vendored VGGT code** (`uniforge3d/models/vggt/`): VGGT's
  [CC BY-NC 4.0](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt) —
  non-commercial use only.
- **BRIA RMBG-2.0** (background removal weights fetched by `get_models.bat`):
  BRIA's [custom license](https://huggingface.co/briaai/RMBG-2.0) — free for
  non-commercial use; commercial use requires a license from BRIA AI.
- **Hi3D v1.0** (`Hi3Dv1.0/` source and the `hi3d-official/Hi3Dv1.0` weights):
  the [Hi3D Open License Agreement Version 1.0](Hi3Dv1.0/LICENSE). Commercial
  use is permitted below its scale threshold (200,000 MAU / USD 1M monthly
  revenue); commercial integrations must display **Powered by Hi3D 1.0**
  linking to [hi3d.ai](http://hi3d.ai/).
- **DINOv2-large** ([facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large))
  and **CLIP ViT-L/14** ([openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)):
  third-party encoders used only by Hi3D, under their respective model licenses.
- **spconv** and **cumm** (Hi3D dependencies): Apache License 2.0. The wheels in
  `.whl/` are local CUDA 13 / sm_120 builds of spconv 2.3.8 and cumm 0.8.2.

Please cite the original paper if you use this work:

```bibtex
@article{chen2026uniforge3d,
  title   = {UniForge3D: Joint 3D Reconstruction and Generation in a Shared Anchor-Guided Space},
  author  = {Chen, Wenjue and Chen, Wenyue and Zheng, Heliang and Jia, Rongfei and Liu, Yuan and Li, Meng},
  journal = {arXiv preprint arXiv:2610.15357},
  year    = {2026}
}
```
