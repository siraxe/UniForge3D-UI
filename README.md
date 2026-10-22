# UniForge3D UI — Windows Desktop Edition

A Windows-focused packaging of [UniForge3D](https://github.com/xibushu/UniForge3D) with a
PySide6 desktop UI (`PS6_UI/`). Generates textured 3D assets (GLB) from sparse, unposed
multi-view images. Apache-2.0, same as upstream.

## Requirements

- Windows 10/11, NVIDIA GPU (CUDA 13.0 wheels)
- [Python 3.12 x64](https://www.python.org/downloads/) (`py -3.12` must work)
- The prebuilt `.whl/` folder (included) — no native builds needed, but the wheels can also be built from source if you prefer

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

Please cite the original paper if you use this work:

```bibtex
@article{chen2026uniforge3d,
  title   = {UniForge3D: Joint 3D Reconstruction and Generation in a Shared Anchor-Guided Space},
  author  = {Chen, Wenjue and Chen, Wenyue and Zheng, Heliang and Jia, Rongfei and Liu, Yuan and Li, Meng},
  journal = {arXiv preprint arXiv:2610.15357},
  year    = {2026}
}
```
