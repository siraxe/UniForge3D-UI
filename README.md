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

After an **Early stop** run, **Run Hi3D 1536** refines the saved
`prediction_512.glb` with Hi3D v1.0, conditioned on the image selected as
**REF** (falling back to that run's first input image).

- Keep the Hi3D v1.0 source checkout at `Hi3Dv1.0/` (or point
  `UNIFORGE_HI3D_SOURCE` at it). It is run directly because its package
  metadata pins Python 3.10 while this app uses 3.12.
- `get_models.bat` also downloads Hi3D v1.0, DINOv2-large and CLIP ViT-L/14
  (~10.7 GB total) into `models/`.
- The result is geometry only and is written next to the preview as
  `prediction_hi3d_1536.glb`. Hi3D unloads as soon as refinement finishes.
- 1536 is GPU-memory intensive. Low VRAM uses sequential CPU offload, but a
  24–32 GB card may still be limited (Hi3D's 1536 mode was validated on a
  96 GB GPU).

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
  linking to [hi3d.ai](http://hi3d.ai/). The UI shows this credit next to the
  Hi3D button.
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
