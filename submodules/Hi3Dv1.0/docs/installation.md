# Installation

Hi3D v1.0 is installed from this repository. Its Python package and API retain the
names `sparc3d` and `Sparc3DPipeline`. No PyPI release or prebuilt project wheel is
supplied. GPU mesh operations use Triton kernels compiled and cached on first use;
`flash-attn` may additionally require a CUDA source build during installation.

## Requirements

| Component | Requirement |
| --- | --- |
| OS | Linux |
| Python | 3.10 (`>=3.10,<3.11` in the package metadata) |
| GPU | NVIDIA, compute capability 8.0 or newer (Ampere or later) |
| CUDA toolkit | 12.0–12.5, matching the installed PyTorch build |

CPU inference and CPU-only installation are not supported. Use the measured CUDA
12.4 stack below as the starting point.

The 1536³ refinement ladder has been validated only on a **96 GB NVIDIA H20**.
Peak memory and feasibility on smaller cards have not been established. Each
transformer has approximately 1.8 billion parameters, with its main layers in
fp16. Sequential CPU offload reduces GPU model residency, but decoded sparse
features still require GPU memory; it does not establish a minimum supported
VRAM size. Python versions other than 3.10 have not been validated.

## The verified stack

The following environment was used for inference validation:

| Component | Version |
| --- | --- |
| Python | 3.10.12 |
| torch | 2.5.0+cu124 |
| Triton | 3.1.0 |
| torchvision | 0.20.0+cu124 |
| CUDA toolkit | 12.4.131 |
| spconv-cu120 | 2.3.6 |
| flash-attn | 2.7.4.post1 |
| transformers | 4.46.3 |
| numpy | 1.26.4 |
| GPU | 96 GB NVIDIA H20 (`sm_90`) |

`spconv-cu120` and `flash-attn` are pinned in the `inference` extra. The
[`docker/constraints.txt`](../docker/constraints.txt) file also pins the measured
versions of the remaining core dependencies. Alternative stacks have not been
measured.

## Install

Clone this repository and run the following from its root in a Python 3.10
environment:

```bash
# Install the CUDA 12.4 PyTorch build first.
pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Install flash-attn against the existing torch environment.
pip install packaging ninja psutil
pip install --no-build-isolation flash-attn==2.7.4.post1

# Install Hi3D v1.0 with the measured GPU dependency versions.
pip install -c docker/constraints.txt -e ".[inference]"
```

The `+cu124` wheels are supplied by the PyTorch CUDA 12.4 index. Install
`flash-attn` separately after PyTorch with build isolation disabled: its build
metadata imports torch, and a source build also requires the CUDA toolkit and
`nvcc`. See [alternative kernel builds](#deviating-from-the-pinned-kernels) below.

PyTorch's measured Linux wheel includes Triton 3.1.0. The first mesh operation in
a fresh environment includes Triton compilation time; later calls use the kernel
cache.

For optional background removal in the demo scripts:

```bash
pip install -c docker/constraints.txt -e ".[demo]"
```

This installs `rembg`. It does not install a coarse generator, its implementation,
or its weights.

### Ambient pip configuration

Use `pip config list` to inspect configured package indexes. Environment variables
and `extra-index-url` entries can add indexes to resolution even when a command
specifies its own primary index. To ignore inherited pip index configuration for
an install, prefix the command as follows:

```bash
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null \
    pip install -c docker/constraints.txt -e ".[inference]"
```

Apply the same prefix to earlier installation commands if needed, retaining their
explicit PyTorch wheel index.

## Docker

The GPU image in [`docker/Dockerfile`](../docker/Dockerfile) installs the measured
stack and copies this repository into the container:

```bash
docker build -f docker/Dockerfile -t hi3d-v1 .
docker run --gpus all --rm -it hi3d-v1
```

The image requires an NVIDIA GPU. It builds `flash-attn` from source by default
for the architectures in `TORCH_CUDA_ARCH_LIST`. Set this argument to the target
GPU architecture to reduce build time; for the H20:

```bash
docker build -f docker/Dockerfile -t hi3d-v1 . \
    --build-arg TORCH_CUDA_ARCH_LIST=9.0
```

Source builds avoid a prebuilt-wheel lookup on GitHub. If that lookup is desired
and GitHub is reachable, pass `--build-arg FLASH_ATTENTION_FORCE_BUILD=FALSE`.

The image sets its own pip index configuration and exports
[`docker/constraints.txt`](../docker/constraints.txt) as `PIP_CONSTRAINT`. This
constraint remains active for subsequent pip installs inside the container.

### Building behind a mirror

The Docker build accepts `PIP_INDEX_URL` for general packages and
`TORCH_INDEX_URL` for a PyTorch cu124 wheel index. Configure both when using
mirrors: the latter must supply the `+cu124` wheels, while the former supplies
other dependencies.

## The conditioning encoders

The refinement artifact does not contain the image encoders. `ImageConditioner`
loads `facebook/dinov2-large` and `openai/clip-vit-large-patch14` from Hugging Face
on first construction. This requires network access even if the refinement
weights are already local, unless the encoder files are cached or provided as
local directories.

For offline refinement, provide local copies of all three model components:

```bash
python demo/refine_mesh.py \
    --model /path/to/hi3d-v1.0-model \
    --image assets/example.png \
    --coarse-mesh coarse.glb \
    --output refined.glb \
    --local-files-only \
    --dino-model /path/to/dinov2-large \
    --clip-model /path/to/clip-vit-large-patch14
```

The encoder directories may use any layout that `transformers` can load; no Hub
cache structure is required. `Sparc3DPipeline.from_pretrained` accepts the same
overrides as `dino_model_id` and `clip_model_id`.

`--cache-dir` points to the directory holding the `models--*` trees, normally
`$HF_HOME/hub`. `--revision` selects the refinement artifact version and does not
pin the separate encoder revisions.

## The coarse stage

`demo/coarse_stage.py` and `demo/image_to_mesh.py` optionally use **Hunyuan3D-2
mini**, repository `tencent/Hunyuan3D-2mini`, subfolder
`hunyuan3d-dit-v2-mini-turbo`, to generate an initial coarse mesh. The Hunyuan integration is an
optional example, not a mandatory dependency of the core library or a required
product integration. You may replace the adapter or provide a triangle mesh
from another source. `demo/refine_mesh.py` requires no Hunyuan implementation or
weights.

Hunyuan code and weights are third-party releases, not Hi3D releases, and are not
bundled or relicensed here. Follow the [upstream source
license](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE), the
[Hunyuan3D-2 mini model
license](https://huggingface.co/tencent/Hunyuan3D-2mini/blob/main/LICENSE), and the
[external-model license index](../THIRD_PARTY_LICENSES.md#external-models-and-optional-examples).

Install and expose `hy3dgen` following the official
[Hunyuan3D-2 instructions](https://github.com/Tencent-Hunyuan/Hunyuan3D-2), and obtain
its weights separately. Its shape-generation modules also require these
packages; install them under this repository's constraints to retain NumPy 1.x:

```bash
PIP_CONSTRAINT="$PWD/docker/constraints.txt" pip install \
    einops diffusers opencv-python-headless pymeshlab scikit-image tqdm pyyaml
```

Neither this command nor the `[demo]` extra installs Hunyuan itself or its
weights. In the Docker image, `PIP_CONSTRAINT` is already set. MeshLab GPU plugins
may additionally need `libopengl0`; without it, `pymeshlab` can use its CPU path.

In the end-to-end demo, `--local-files-only` and `--cache-dir` configure the
refinement model and its image encoders; `--revision` configures only the
refinement artifact. None configures the Hunyuan loader. For a fully local run,
also pass `--coarse-model` pointing to a local Hunyuan model directory and the
correct `--coarse-subfolder`.

## Deviating from the pinned kernels

Alternative kernel builds are outside the measured environment. Install the
remaining required dependencies separately before using `--no-deps`.

**spconv.** The `spconv-cu120` build covers CUDA 12.0–12.5. A CUDA 12.6 stack
requires a matching build rather than the pinned one:

```bash
pip install spconv-cu126
pip install -e . --no-deps
```

The unsuffixed `spconv` PyPI package is CPU-only and cannot run this pipeline.
The sparse convolution weights were validated with `spconv-cu120==2.3.6`; other
builds have not been measured.

**flash-attn.** A matching wheel may not exist for every torch, Python, and CUDA
combination. To force a source build, set the target GPU architecture and choose
build parallelism appropriate for the host's CPU and RAM:

```bash
pip install packaging ninja psutil
FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=8 \
    pip install --no-build-isolation flash-attn==2.7.4.post1
```

The `9.0` architecture above targets the H20; change it for other GPUs.
`--no-build-isolation` allows the build to import the installed torch. The
transformer calls flash attention's variable-length kernel directly, with no
fallback through `torch.nn.functional.scaled_dot_product_attention`.

## Verifying the install

```bash
python -c "import sparc3d; from sparc3d import Sparc3DPipeline; print(sparc3d.__version__)"
```

This checks package import only. GPU dependencies are imported when a model is
constructed, so a successful import does not validate inference. Run a compatible
checkpoint with an image and coarse mesh to verify the complete setup; see
[reproduction and validation](reproduction.md).

Adapted from SPARC3D documentation for Hi3D v1.0.
