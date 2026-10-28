"""Download the UniForge3D inference files into ``models/<owner>/<repo>``.

Only public, ungated sources are used:

* Hugging Face for the open UniForge3D, Microsoft TRELLIS, Hi3D v1.0,
  LATO.2, DINOv2-large, and CLIP ViT-L/14 repositories.
* ModelScope public mirrors for DINOv3 and RMBG-2.0, whose Hugging Face
  repositories are gated.
* dl.fbaipublicfiles.com for the LATO.2 DINOv2-L/14-with-registers backbone
  (public, no token required).

The Hi3D v1.0 refinement models and their two conditioning encoders are
downloaded to::

    models/hi3d-official/Hi3Dv1.0/
    models/facebook/dinov2-large/
    models/openai/clip-vit-large-patch14/

The LATO.2 V-Flow/T-Flow checkpoints and their DINOv2 encoder are
downloaded to::

    models/0x4c48/LATO.2/
    models/0x4c48/LATO.2/dinov2/checkpoints/  (torch.hub cache layout)

The original model groups are downloaded straight to the destination with
urllib, so no global Hugging Face cache (and no ``models/hf_cache``) is
created. The Hi3D v1.0 model and its DINOv2/CLIP encoders are fetched with
``huggingface_hub.snapshot_download`` into their final ``models/<owner>/<repo>``
directories, using an allow-list so duplicate encoder checkpoints are not
downloaded. A patched local pipeline config with checkpoint paths relative to
the models tree is written to:
    models/xibushu/UniForge3D/pipeline.json

The script is resumable: existing files with the expected size are skipped,
``.part`` files are continued with an HTTP Range request, and snapshot
downloads resume from their local metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "models"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UniForge3D-ModelDownloader/1.0"

CORE_BASE = (
    "https://huggingface.co/xibushu/UniForge3D/resolve/"
    "a084333f2aae0213e85b9a4541752d1221d618db/"
)
TRELLIS2_BASE = (
    "https://huggingface.co/microsoft/TRELLIS.2-4B/resolve/"
    "af44b45f2e35a493886929c6d786e563ec68364d/"
)
TRELLIS_IMG_BASE = (
    "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/"
    "25e0d31ffbebe4b5a97464dd851910efc3002d96/"
)
LATO2_BASE = (
    "https://huggingface.co/0x4c48/LATO.2/resolve/"
    "a91090e8077b9318ab87ac08fd9eb905903d4515/"
)
DINOV2_REG4_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/"
    "dinov2_vitl14_reg4_pretrain.pth"
)
DINO_MS = (
    "https://modelscope.cn/models/facebook/"
    "dinov3-vitl16-pretrain-lvd1689m/resolve/"
)
RMBG_MS = "https://modelscope.cn/models/AI-ModelScope/RMBG-2.0/resolve/"


@dataclass(frozen=True)
class DownloadFile:
    path: str                 # relative to models/<owner>/<repo>/
    url: str
    size: int
    sha256: str | None = None


@dataclass(frozen=True)
class ModelGroup:
    name: str                 # owner/repo directory under models/
    source: str
    files: tuple[DownloadFile, ...]


@dataclass(frozen=True)
class RepoGroup:
    """Whole Hugging Face repository fetched with ``snapshot_download``.

    ``allow_patterns`` keeps the download to the runtime files we need. For
    the encoders this avoids fetching the duplicate ``pytorch_model.bin``
    weights that a literal whole-repo download would include.
    """

    name: str                 # owner/repo directory under models/
    source: str
    repo_id: str
    revision: str
    allow_patterns: tuple[str, ...]
    required_files: tuple[str, ...]


REPO_GROUPS: tuple[RepoGroup, ...] = (
    RepoGroup(
        "hi3d-official/Hi3Dv1.0",
        "Hugging Face, open (revision 705fc090)",
        "hi3d-official/Hi3Dv1.0",
        "705fc0901c5d896cdcbf1dbf1feaa173e62bdd3a",
        (
            "model_index.json",
            "scheduler/*",
            "vae/*",
            "transformer_512/*",
            "transformer_1024/*",
        ),
        (
            "model_index.json",
            "scheduler/scheduler_config.json",
            "vae/config.json",
            "vae/model.safetensors",
            "transformer_512/config.json",
            "transformer_512/model.safetensors",
            "transformer_1024/config.json",
            "transformer_1024/model.safetensors",
        ),
    ),
    RepoGroup(
        "facebook/dinov2-large",
        "Hugging Face, open (revision 47b73eef); Hi3D conditioning encoder",
        "facebook/dinov2-large",
        "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
        (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
        ),
        (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
        ),
    ),
    RepoGroup(
        "openai/clip-vit-large-patch14",
        "Hugging Face, open (revision 32bd6428); Hi3D conditioning encoder",
        "openai/clip-vit-large-patch14",
        "32bd64288804d66eefd0ccbe215aa642df71cc41",
        (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
        ),
        (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
        ),
    ),
)


GROUPS: tuple[ModelGroup, ...] = (
    ModelGroup(
        "xibushu/UniForge3D",
        "Hugging Face, open (revision a084333f)",
        (
            DownloadFile(
                "uniforge3d/stage1/uniforge3d_stage1_structure_joint_model.pt",
                CORE_BASE + "uniforge3d/stage1/uniforge3d_stage1_structure_joint_model.pt",
                1_234_004_742,
                "a4079cc6f0b0748d3afb301717eddf532ed4f5ddb80f37bb9294a290ee89f824",
            ),
            DownloadFile(
                "uniforge3d/vggt/uniforge3d_stage1_structure_vggt_backbone.pt",
                CORE_BASE + "uniforge3d/vggt/uniforge3d_stage1_structure_vggt_backbone.pt",
                4_632_306_609,
                "a3844cfc95b5ab6d4f002e0bce967bb7184489169531963b45134cb51eb3c5c1",
            ),
            DownloadFile(
                "uniforge3d/vggt/uniforge3d_shape_texture_vggt_conditioner.pt",
                CORE_BASE + "uniforge3d/vggt/uniforge3d_shape_texture_vggt_conditioner.pt",
                3_898_562_754,
                "ecc0915beb86ee0b027ac65ca78e2abb29ad6fe140d904e487c9c3f4120f6902",
            ),
            DownloadFile(
                "uniforge3d/shape512/uniforge3d_stage2_shape512_denoiser.pt",
                CORE_BASE + "uniforge3d/shape512/uniforge3d_stage2_shape512_denoiser.pt",
                5_468_197_230,
                "121abea6b1fc1ec2e1c47ef0f102089574a6a718e8ca8dbc2426ebee738e0f51",
            ),
            DownloadFile(
                "uniforge3d/shape512/uniforge3d_stage2_shape512_vggt_adapter.pt",
                CORE_BASE + "uniforge3d/shape512/uniforge3d_stage2_shape512_vggt_adapter.pt",
                251_868_386,
                "dcaffa7cbea19b0670ae0fded3864a65e151ac0d663c338f374492d20d65e55d",
            ),
            DownloadFile(
                "uniforge3d/shape1024/uniforge3d_stage2_shape1024_denoiser.pt",
                CORE_BASE + "uniforge3d/shape1024/uniforge3d_stage2_shape1024_denoiser.pt",
                5_468_197_230,
                "dbe0699d7c197cc17c63c64240af72b88f4d502cedd584a60f9e0bd2fd9e1d04",
            ),
            DownloadFile(
                "uniforge3d/shape1024/uniforge3d_stage2_shape1024_vggt_adapter.pt",
                CORE_BASE + "uniforge3d/shape1024/uniforge3d_stage2_shape1024_vggt_adapter.pt",
                256_108_930,
                "591eae22500e4e537c1e0f4d504efc61868847894406f9f41cca54cb38b41dde",
            ),
            DownloadFile(
                "uniforge3d/texture/uniforge3d_stage3_texture1024_denoiser.pt",
                CORE_BASE + "uniforge3d/texture/uniforge3d_stage3_texture1024_denoiser.pt",
                5_468_533_342,
                "302c5ac4ae5ebae31fee4cd78704953b4fe578594a66e93a5a655fb36b5b786e",
            ),
            DownloadFile(
                "uniforge3d/texture/uniforge3d_stage3_texture1024_vggt_adapter.pt",
                CORE_BASE + "uniforge3d/texture/uniforge3d_stage3_texture1024_vggt_adapter.pt",
                256_109_038,
                "30b4946d8a6cc22dac68695e078ab198cb6380e337674f414e7e70638e9ebbbb",
            ),
        ),
    ),
    ModelGroup(
        "microsoft/TRELLIS.2-4B",
        "Hugging Face, open (revision af44b45f)",
        (
            DownloadFile(
                "ckpts/shape_dec_next_dc_f16c32_fp16.json",
                TRELLIS2_BASE + "ckpts/shape_dec_next_dc_f16c32_fp16.json",
                678,
            ),
            DownloadFile(
                "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
                TRELLIS2_BASE + "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
                948_490_494,
                "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581",
            ),
            DownloadFile(
                "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
                TRELLIS2_BASE + "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
                2_584_426_920,
                "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6",
            ),
            DownloadFile(
                "ckpts/tex_dec_next_dc_f16c32_fp16.json",
                TRELLIS2_BASE + "ckpts/tex_dec_next_dc_f16c32_fp16.json",
                705,
            ),
            DownloadFile(
                "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
                TRELLIS2_BASE + "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
                948_458_812,
                "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c",
            ),
        ),
    ),
    ModelGroup(
        "microsoft/TRELLIS-image-large",
        "Hugging Face, open (revision 25e0d31f)",
        (
            DownloadFile(
                "ckpts/ss_dec_conv3d_16l8_fp16.json",
                TRELLIS_IMG_BASE + "ckpts/ss_dec_conv3d_16l8_fp16.json",
                245,
            ),
            DownloadFile(
                "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
                TRELLIS_IMG_BASE + "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
                147_591_972,
                "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a",
            ),
        ),
    ),
    ModelGroup(
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "ModelScope public mirror; original Hugging Face repo is gated",
        (
            DownloadFile(
                "config.json",
                DINO_MS + "f9eebae294f741f243dbf611a5bceee5b6c18825/config.json",
                745,
                "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5",
            ),
            DownloadFile(
                "model.safetensors",
                DINO_MS + "bab39f7f4023c45534a5326e9378edc5b2016e48/model.safetensors",
                1_212_559_808,
                "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
            ),
        ),
    ),
    ModelGroup(
        "briaai/RMBG-2.0",
        "ModelScope public mirror (AI-ModelScope/RMBG-2.0); original HF repo is gated",
        (
            DownloadFile(
                "config.json",
                RMBG_MS + "5e43c27f1c6c4dc2ecd891427a4735792055614a/config.json",
                405,
                "c97ea21569daf66b205491a4635147dd3bc42c7c168b89d7d75b53f67ef548ae",
            ),
            DownloadFile(
                "birefnet.py",
                RMBG_MS + "1f03884e5530e5bde3b5832eca78ce91207df804/birefnet.py",
                91_403,
                "e499d75224b8819e985e68fb78b7a8e8c99316840474e74e16b5529f03ca2860",
            ),
            DownloadFile(
                "BiRefNet_config.py",
                RMBG_MS + "e371aea0f81de58f97a44bc5e63ffba6645dea3e/BiRefNet_config.py",
                298,
                "e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52",
            ),
            DownloadFile(
                "model.safetensors",
                RMBG_MS + "e371aea0f81de58f97a44bc5e63ffba6645dea3e/model.safetensors",
                884_878_856,
                "566ed80c3d95f87ada6864d4cbe2290a1c5eb1c7bb0b123e984f60f76b02c3a7",
            ),
        ),
    ),
)

# LATO.2 is an optional group: V-Flow -> V-VAE -> T-Flow mesh generation.
# The six runtime checkpoints are the ones loaded by scripts/e2e_inference.py;
# vdf_encoder.pt is training-only and intentionally not downloaded.
# The DINOv2 weights are placed in the torch.hub cache layout so the encoder
# can run offline once the facebookresearch/dinov2 code is present.
LATO2_GROUPS: tuple[ModelGroup, ...] = (
    ModelGroup(
        "0x4c48/LATO.2",
        "Hugging Face, open (revision a91090e8) + Meta DINOv2-L/14-reg",
        (
            DownloadFile(
                "vflow.pt",
                LATO2_BASE + "vflow.pt",
                649_153_594,
                "a5c660601af9ce9909feb5157cfd456d81b0498835d7ba4129a7d4267a8927d3",
            ),
            DownloadFile(
                "vvae.pt",
                LATO2_BASE + "vvae.pt",
                1_261_420_852,
                "b1192e3e6301dee5ee3dac37d411f82141ee1a8764707f0ea5b7746052342d44",
            ),
            DownloadFile(
                "offset_head.pt",
                LATO2_BASE + "offset_head.pt",
                4_229_424,
                "b9ab60321ffa1026813b4cb79eba7a9d224ccd77e01e6b2c84fd29cad301c9df",
            ),
            DownloadFile(
                "tflow.pt",
                LATO2_BASE + "tflow.pt",
                882_821_662,
                "5ac0033544b813ec046f00d5f12d3b488a91a98c1bfef0560948fb4c6b52c6ad",
            ),
            DownloadFile(
                "tvae.pt",
                LATO2_BASE + "tvae.pt",
                713_397_214,
                "b109bcf2bc42abea89a3563e3fefa9509f9cd523501d5c62d78108380a3db17c",
            ),
            DownloadFile(
                "voxel_encoder.pt",
                LATO2_BASE + "voxel_encoder.pt",
                61_373_424,
                "c304e36651f2a8c63d5b30d0716a4a8ab4ef505fe39311d060e04e5ab253d4ce",
            ),
            DownloadFile(
                "dinov2/checkpoints/dinov2_vitl14_reg4_pretrain.pth",
                DINOV2_REG4_URL,
                1_217_607_321,
            ),
        ),
    ),
)


def fmt_bytes(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} GB"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} MB"
    if value >= 1_000:
        return f"{value / 1_000:.0f} KB"
    return f"{int(value)} B"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressBar:
    """One-line progress bar with speed and ETA."""

    def __init__(self, total: int, initial: int = 0) -> None:
        self.total = max(int(total), 1)
        self.done = min(max(int(initial), 0), self.total)
        self._initial = self.done
        self._started = time.monotonic()
        self._last_render = 0.0
        self._tty = sys.stdout.isatty()
        self._line_width = max(shutil.get_terminal_size((90, 20)).columns - 1, 40)
        self._bar_width = max(8, min(26, self._line_width - 56))

    def _render_line(self) -> str:
        fraction = min(self.done / self.total, 1.0)
        filled = int(self._bar_width * fraction)
        bar = "#" * filled + "-" * (self._bar_width - filled)
        elapsed = max(time.monotonic() - self._started, 0.001)
        rate = max(self.done - self._initial, 0) / elapsed
        eta = (self.total - self.done) / rate if rate > 1 else None
        return (
            f"  {fraction * 100:5.1f}% [{bar}] "
            f"{fmt_bytes(self.done)}/{fmt_bytes(self.total)} "
            f"{fmt_bytes(rate)}/s ETA {fmt_duration(eta)}"
        )

    def render(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_render < 0.2:
            return
        self._last_render = now
        line = self._render_line()
        if self._tty:
            sys.stdout.write("\r" + line.ljust(self._line_width))
        else:
            pct = int(self.done * 100 / self.total)
            line = f"  {pct:3d}% {fmt_bytes(self.done)}/{fmt_bytes(self.total)}"
            sys.stdout.write("\r" + line)
        sys.stdout.flush()

    def update(self, amount: int) -> None:
        self.done = min(self.done + amount, self.total)
        self.render()

    def finish(self) -> None:
        self.done = self.total
        self.render(force=True)
        if self._tty:
            sys.stdout.write("\n")
        else:
            sys.stdout.write("\n")
        sys.stdout.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    bar = ProgressBar(size) if size >= 16 * 1024 * 1024 else None
    if bar is not None:
        print("  verifying sha256 ...")
        bar.render(force=True)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if bar is not None:
                bar.update(len(chunk))
    if bar is not None:
        bar.finish()
    return digest.hexdigest()


def finalize_part(part: Path, dest: Path, spec: DownloadFile) -> None:
    """Verify a completed .part file and atomically move it into place."""
    actual_size = part.stat().st_size
    if actual_size != spec.size:
        raise RuntimeError(f"size mismatch: got {actual_size}, expected {spec.size}")
    if spec.sha256:
        actual = sha256_file(part)
        if actual != spec.sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"sha256 mismatch: got {actual}, expected {spec.sha256}"
            )
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(part, dest)


def download_file(spec: DownloadFile, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")

    if dest.exists():
        if dest.stat().st_size == spec.size:
            return
        print(f"  removing wrong-sized existing file ({fmt_bytes(dest.stat().st_size)})")
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    if part.exists() and part.stat().st_size > spec.size:
        print("  removing oversized .part file")
        part.unlink()
    if part.exists() and part.stat().st_size == spec.size:
        print("  completing previously downloaded .part file")
        finalize_part(part, dest, spec)
        return

    last_error: Exception | None = None
    for attempt in range(1, 6):
        offset = part.stat().st_size if part.exists() else 0
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Accept-Encoding": "identity",
        }
        parsed = urllib.parse.urlsplit(spec.url)
        headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(spec.url, headers=headers)
        bar: ProgressBar | None = None
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status == 206:
                    mode = "ab"
                    start = offset
                    print(f"  resuming at {start * 100 / spec.size:.1f}%")
                else:
                    if offset:
                        print("  server did not honor Range; restarting download")
                    mode = "wb"
                    start = 0
                bar = ProgressBar(spec.size, start)
                bar.render(force=True)
                with part.open(mode) as handle:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        bar.update(len(chunk))
            downloaded = part.stat().st_size if part.exists() else 0
            if downloaded == spec.size:
                bar.finish()
                finalize_part(part, dest, spec)
                return
            if bar is not None:
                sys.stdout.write("\n")
                sys.stdout.flush()
            last_error = RuntimeError(
                f"incomplete download ({downloaded} of {spec.size} bytes)"
            )
        except urllib.error.HTTPError as exc:
            if bar is not None:
                sys.stdout.write("\n")
                sys.stdout.flush()
            if exc.code == 416 and offset == spec.size:
                finalize_part(part, dest, spec)
                return
            last_error = exc
            if exc.code == 416:
                part.unlink(missing_ok=True)
            exc.close()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if bar is not None:
                sys.stdout.write("\n")
                sys.stdout.flush()
            last_error = exc
        if attempt < 5:
            delay = min(2 ** attempt, 20)
            print(f"  download error: {last_error}")
            print(f"  retrying in {delay}s ({attempt}/5) ...")
            time.sleep(delay)
    raise RuntimeError(f"failed to download {spec.url}: {last_error}")


def _config_ref(path: Path, config_dir: Path) -> str:
    """Point at *path* relative to the config directory, with forward slashes.

    The pipeline resolves every local checkpoint against the model directory
    that holds ``pipeline.json``, so relative references keep the models tree
    portable and safe to commit.
    """
    return Path(os.path.relpath(path, config_dir)).as_posix()


def build_local_pipeline(dest_root: Path) -> Path:
    """Write a pipeline config whose references are relative to the config."""
    source = ROOT / "submodules" / "uniforge3d" / "configs" / "pipeline.json"
    config = json.loads(source.read_text(encoding="utf-8"))

    core_root = dest_root / "xibushu" / "UniForge3D"
    trellis2 = dest_root / "microsoft" / "TRELLIS.2-4B"
    trellis_image = dest_root / "microsoft" / "TRELLIS-image-large"
    dino = dest_root / "facebook" / "dinov3-vitl16-pretrain-lvd1689m"
    rmbg = dest_root / "briaai" / "RMBG-2.0"

    for stage in ("sparse_structure", "structured_latent_512", "structured_latent_1024", "texture"):
        weights = config[stage]["weights"]
        for key in ("checkpoint", "adapter_checkpoint", "vggt_base", "base_checkpoint"):
            value = weights.get(key)
            if isinstance(value, str):
                path = Path(value)
                target = path if path.is_absolute() else core_root / path
                weights[key] = _config_ref(target, core_root)

    config["sparse_structure"]["weights"]["dit_base"] = _config_ref(
        trellis2 / "ckpts" / "ss_flow_img_dit_1_3B_64_bf16.safetensors", core_root,
    )
    config["sparse_structure"]["decoder"]["checkpoint"] = _config_ref(
        trellis_image / "ckpts" / "ss_dec_conv3d_16l8_fp16", core_root,
    )
    for stage in ("structured_latent_512", "structured_latent_1024"):
        config[stage]["decoder"]["checkpoint"] = _config_ref(
            trellis2 / "ckpts" / "shape_dec_next_dc_f16c32_fp16", core_root,
        )
    config["texture"]["shape_decoder_checkpoint"] = _config_ref(
        trellis2 / "ckpts" / "shape_dec_next_dc_f16c32_fp16", core_root,
    )
    config["texture"]["texture_decoder_checkpoint"] = _config_ref(
        trellis2 / "ckpts" / "tex_dec_next_dc_f16c32_fp16", core_root,
    )
    config["conditioner"]["vggt_feature_extractor"]["checkpoint_path"] = _config_ref(
        core_root / "uniforge3d" / "vggt" / "uniforge3d_shape_texture_vggt_conditioner.pt",
        core_root,
    )

    external = config.setdefault("external_models", {})
    external["dino"] = _config_ref(dino, core_root)
    external["rembg"] = _config_ref(rmbg, core_root)
    config.setdefault("image_preprocess", {})["rembg_model"] = _config_ref(rmbg, core_root)

    config_path = core_root / "pipeline.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return config_path


def required_paths(
    dest_root: Path,
    groups: tuple[ModelGroup, ...] = GROUPS,
    repo_groups: tuple[RepoGroup, ...] = REPO_GROUPS,
    include_main: bool = True,
) -> list[Path]:
    paths = [dest_root / group.name / spec.path for group in groups for spec in group.files]
    paths.extend(
        dest_root / group.name / relative
        for group in repo_groups
        for relative in group.required_files
    )
    if include_main:
        core_root = dest_root / "xibushu" / "UniForge3D"
        for decoder in (
            dest_root / "microsoft" / "TRELLIS-image-large" / "ckpts" / "ss_dec_conv3d_16l8_fp16",
            dest_root / "microsoft" / "TRELLIS.2-4B" / "ckpts" / "shape_dec_next_dc_f16c32_fp16",
            dest_root / "microsoft" / "TRELLIS.2-4B" / "ckpts" / "tex_dec_next_dc_f16c32_fp16",
        ):
            paths.extend([Path(f"{decoder}.json"), Path(f"{decoder}.safetensors")])
        paths.extend([
            dest_root / "facebook" / "dinov3-vitl16-pretrain-lvd1689m" / "config.json",
            dest_root / "briaai" / "RMBG-2.0" / "config.json",
        ])
        paths.append(core_root / "pipeline.json")
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download UniForge3D weights to models/<owner>/<repo>/ "
                    "without creating an HF cache.",
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
        help="destination root (default: models/)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="only check existing files; do not download or write a config",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="also re-hash files that already exist (new downloads are always verified)",
    )
    parser.add_argument(
        "--groups", default="all",
        help="comma-separated model sets: 'main' (UniForge3D + TRELLIS + "
             "preprocessors), 'hi3d' (Hi3Dv1.0 snapshot + encoders), 'lato2' "
             "(LATO.2 checkpoints + DINOv2-L/14-reg), or 'all' (default). "
             "Combine with commas, e.g. main,lato2.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dest = args.dest.expanduser().resolve()
    requested = [g.strip().lower() for g in args.groups.split(",") if g.strip()]
    valid_groups = {"main", "hi3d", "lato2", "all"}
    unknown = [g for g in requested if g not in valid_groups]
    if unknown or not requested:
        print(f"[get_models] ERROR: unknown --groups value(s): {', '.join(unknown) or '(empty)'}")
        print("[get_models] Valid values: all, main, hi3d, lato2 (comma-separated).")
        return 2
    if "all" in requested:
        include_main = include_hi3d = include_lato2 = True
    else:
        include_main = "main" in requested
        include_hi3d = "hi3d" in requested
        include_lato2 = "lato2" in requested
    groups: tuple[ModelGroup, ...] = (
        (GROUPS if include_main else ())
        + (LATO2_GROUPS if include_lato2 else ())
    )
    repo_groups: tuple[RepoGroup, ...] = REPO_GROUPS if include_hi3d else ()
    all_files = [(group, spec) for group in groups for spec in group.files]
    total_bytes = sum(spec.size for _, spec in all_files)
    total_files = len(all_files)
    repo_files = [
        (group, relative)
        for group in repo_groups
        for relative in group.required_files
    ]

    print("[get_models] UniForge3D + Hi3D + LATO.2 model downloader")
    print(f"[get_models] Destination: {dest}")
    print("[get_models] Layout: <owner>/<repo>/... (model files stored directly)")
    if not include_main and not include_hi3d and not include_lato2:
        print("[get_models] No model set selected; nothing to do.")
        return 0
    print(
        "[get_models] Selected: "
        + " + ".join(
            name for name, on in (
                ("Main models", include_main),
                ("Hi3Dv1.0", include_hi3d),
                ("LATO.2", include_lato2),
            ) if on
        )
    )
    print("\n[get_models] Sources:")
    for group in repo_groups:
        print(f"  {group.name:<48} {group.source} [snapshot]")
    for group in groups:
        print(f"  {group.name:<48} {group.source}")
    if repo_groups:
        print(
            f"\n[get_models] Required: {total_files + len(repo_files)} files "
            f"({fmt_bytes(total_bytes)} explicit + ~10.7 GB Hi3D snapshot)"
        )
    else:
        print(
            f"\n[get_models] Required: {total_files} files "
            f"({fmt_bytes(total_bytes)} explicit)"
        )

    present: list[tuple[ModelGroup, DownloadFile, Path]] = []
    missing: list[tuple[ModelGroup, DownloadFile, Path]] = []
    for group in groups:
        for spec in group.files:
            path = dest / group.name / spec.path
            okay = path.is_file() and path.stat().st_size == spec.size
            if okay and args.verify and spec.sha256:
                print(f"\n[verify] {group.name}/{spec.path}")
                actual = sha256_file(path)
                if actual != spec.sha256:
                    print(f"  checksum mismatch; will replace (got {actual})")
                    path.unlink()
                    okay = False
            if okay:
                present.append((group, spec, path))
            else:
                if path.exists():
                    print(
                        f"[check] wrong size: {group.name}/{spec.path} "
                        f"({fmt_bytes(path.stat().st_size)} vs {fmt_bytes(spec.size)})"
                    )
                part = path.with_name(path.name + ".part")
                if part.exists() and part.stat().st_size < spec.size:
                    print(
                        f"[check] resumable: {group.name}/{spec.path} "
                        f"at {part.stat().st_size * 100 / spec.size:.1f}%"
                    )
                missing.append((group, spec, path))

    repo_present = [
        (group, relative)
        for group, relative in repo_files
        if (dest / group.name / relative).is_file()
    ]
    repo_missing = [
        (group, relative)
        for group, relative in repo_files
        if not (dest / group.name / relative).is_file()
    ]

    present_bytes = sum(spec.size for _, spec, _ in present)
    missing_bytes = sum(spec.size for _, spec, _ in missing)
    print(
        f"\n[check] Existing: {len(present)}/{total_files} explicit files "
        f"({fmt_bytes(present_bytes)})"
    )
    print(
        f"[check] To download: {len(missing)} explicit files "
        f"({fmt_bytes(missing_bytes)} expected)"
    )
    print(
        f"[check] Hi3D snapshot repos: {len(repo_present)}/{len(repo_files)} "
        "runtime files present"
    )

    if args.check:
        if missing or repo_missing:
            if missing:
                print("\n[check] Missing explicit files:")
                for group, spec, _ in missing:
                    print(f"  {group.name}/{spec.path}")
            if repo_missing:
                print("\n[check] Missing Hi3D snapshot files:")
                for group, relative in repo_missing:
                    print(f"  {group.name}/{relative}")
            print("\n[check] Nothing was downloaded.")
            return 1
        print("\n[check] All required model files are present.")
        return 0

    current_group = None
    try:
        if repo_groups:
            from huggingface_hub import snapshot_download

        # Hi3D first: its weights are the large new download.
        for group in repo_groups:
            print(f"\n[{group.name}]")
            print(f"  {group.source}")
            snapshot_download(
                repo_id=group.repo_id,
                revision=group.revision,
                allow_patterns=list(group.allow_patterns),
                local_dir=dest / group.name,
            )
            group_missing = [
                relative
                for relative in group.required_files
                if not (dest / group.name / relative).is_file()
            ]
            if group_missing:
                raise RuntimeError(
                    f"{group.name} is missing required files after download: "
                    + ", ".join(group_missing)
                )

        for index, (group, spec, path) in enumerate(missing, 1):
            if group.name != current_group:
                print(f"\n[{group.name}]")
                current_group = group.name
            print(f"\n[{index}/{len(missing)}] {group.name}/{spec.path}  ({fmt_bytes(spec.size)})")
            download_file(spec, path)
    except KeyboardInterrupt:
        print("\n[get_models] Interrupted. Partial files are kept; rerun to resume.")
        return 130
    except Exception as exc:
        print(f"\n[get_models] ERROR: {exc}")
        print("[get_models] Partial downloads are kept; run again to resume.")
        return 1

    bad = [
        f"{group.name}/{spec.path}"
        for group, spec in all_files
        if not (dest / group.name / spec.path).is_file()
        or (dest / group.name / spec.path).stat().st_size != spec.size
    ]
    bad.extend(
        f"{group.name}/{relative}"
        for group, relative in repo_files
        if not (dest / group.name / relative).is_file()
    )
    if bad:
        print("\n[get_models] ERROR: these files are still incomplete:")
        for name in bad:
            print(f"  {name}")
        return 1

    try:
        config_path = build_local_pipeline(dest) if include_main else None
    except Exception as exc:
        print(f"\n[get_models] ERROR creating local pipeline config: {exc}")
        return 1

    missing_after = [
        path for path in required_paths(dest, groups, repo_groups, include_main)
        if not path.is_file()
    ]
    if missing_after:
        print("\n[get_models] ERROR: required files are missing:")
        for path in missing_after:
            print(f"  {path}")
        return 1

    python_cmd = r".venv\Scripts\python.exe" if (ROOT / ".venv" / "Scripts" / "python.exe").is_file() else "python"

    print(
        f"\n[get_models] Complete: {total_files} explicit files "
        f"({fmt_bytes(total_bytes)})"
        + (" plus Hi3D snapshot repos." if repo_groups else ".")
    )
    if config_path is not None:
        try:
            config_display = config_path.relative_to(ROOT)
        except ValueError:
            config_display = config_path
        print(f"[get_models] Local pipeline config: {config_display}")
        print("[get_models] Model files are stored directly under models/.")
        print("\nRun with:")
        print(f'  {python_cmd} infer.py --config "{config_display}"')
        print("\nThe PS6 UI (start_ps6.bat) and infer.py use this models/ tree automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
