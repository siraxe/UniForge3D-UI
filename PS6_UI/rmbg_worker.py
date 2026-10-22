"""Standalone background-removal worker (RMBG-2.0 only).

Spawned by flet_UI/main.py to pre-cut imported images WITHOUT loading the
full 3D pipeline (which pulls ~26 GB of weights into RAM). It loads just
RMBG-2.0, processes every requested image, saves RGBA cutouts as PNG files,
then exits -- which also frees the model's RAM automatically.

Images that already carry a real alpha channel are reported as processed
without touching the model, mirroring the pipeline's own skip rule.

stdout carries the JSON protocol, one object per line; every other print
(including transformers chatter) is redirected to stderr:

  {"type": "log", "msg": ...}
  {"type": "image", "source": "...", "processed": "..."}
  {"type": "done"}
  {"type": "error", "msg": "..."}
"""

import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_paths import (  # noqa: E402
    find_model_dir,
    find_rmbg_dir,
    hf_cache_dir,
)

# Same cache as worker.py so the model never downloads twice.
MODEL_DIR = find_model_dir()
HF_CACHE = hf_cache_dir()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from transformers import AutoModelForImageSegmentation  # noqa: E402

_real_stdout = sys.stdout
sys.stdout = sys.stderr


def emit(obj) -> None:
    """Write one protocol line; if the UI is gone (broken pipe), just quit."""
    try:
        _real_stdout.write(json.dumps(obj) + "\n")
        _real_stdout.flush()
    except OSError:
        os._exit(1)


# Identical transform to uniforge3d.utils.preprocessing.BackgroundRemover,
# so cutouts produced here match what the pipeline would make itself.
_TRANSFORM = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def has_real_alpha(path: Path) -> bool:
    """True when the image is RGBA with a non-trivial alpha channel."""
    try:
        with Image.open(path) as image:
            if image.mode != "RGBA":
                return False
            return not bool(np.all(np.asarray(image.getchannel("A")) == 255))
    except Exception:
        return False


def cutout_path_for(src: Path, out_dir: Path) -> Path:
    """Deterministic output name; the path hash avoids name collisions."""
    tag = hashlib.sha1(str(src.resolve()).encode("utf-8")).hexdigest()[:8]
    return out_dir / f"{src.stem}_{tag}_cutout.png"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the local models/ copy, fall back to the Hub.
    local = find_rmbg_dir(MODEL_DIR)
    source = str(local) if local is not None else "briaai/RMBG-2.0"

    try:
        emit({"type": "log", "msg": f"Loading RMBG-2.0 from {source} ..."})
        device = (torch.device("cuda", 0) if torch.cuda.is_available()
                  else torch.device("cpu"))
        # RMBG-2.0's remote code imports timm through deprecated shims.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"Importing from timm\.", category=FutureWarning
            )
            model = AutoModelForImageSegmentation.from_pretrained(
                source, trust_remote_code=True,
            ).eval().to(device)
    except Exception as ex:
        emit({"type": "error", "msg": f"Failed to load RMBG-2.0: {ex}"})
        return 1

    with torch.no_grad():
        for raw in args.images:
            src = Path(raw)
            try:
                if has_real_alpha(src):
                    emit({"type": "image", "source": str(src),
                          "processed": str(src)})
                    continue
                with Image.open(src) as image:
                    size = image.size
                rgb = Image.open(src).convert("RGB")
                inputs = _TRANSFORM(rgb).unsqueeze(0).to(device)
                prediction = model(inputs)[-1].sigmoid().cpu()
                mask = transforms.ToPILImage()(
                    prediction[0].squeeze()).resize(size)
                rgb.putalpha(mask)
                dst = cutout_path_for(src, out_dir)
                rgb.save(dst)
                del rgb
                emit({"type": "image", "source": str(src), "processed": str(dst)})
            except Exception as ex:
                emit({"type": "log", "msg": f"Failed on {src.name}: {ex}"})

    del model
    emit({"type": "done"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
