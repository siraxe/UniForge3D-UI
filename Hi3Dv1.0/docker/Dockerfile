# SPARC3D inference image.
#
# Ubuntu 22.04 supplies Python 3.10. The CUDA 12.4 devel toolkit is present so
# flash-attn can compile from source when no wheel matches. Compute capability
# 8.0 and newer, matching the documented floor.
FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

# Pip must resolve against these indexes and nothing else. A base image or a
# builder that exports PIP_INDEX_URL, PIP_EXTRA_INDEX_URL or a pip.conf would
# otherwise join resolution silently, and a command-line --index-url overrides
# the first of those but not the second. Point them at mirrors with --build-arg;
# see docs/installation.md.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

# Every architecture at or above the documented floor, so the image is portable.
# Narrow it to one to compile flash-attn several times faster.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# flash-attn's setup.py otherwise reaches for a prebuilt wheel on GitHub first,
# and on a host that cannot reach it the request hangs with no output rather than
# falling through to the compile. Building from source is what the devel toolkit
# in this image is for. Set FALSE to take the wheel when GitHub is reachable.
ARG FLASH_ATTENTION_FORCE_BUILD=TRUE

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} \
    PIP_CONFIG_FILE=/dev/null \
    PIP_CONSTRAINT=/etc/sparc3d-constraints.txt \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=64

# Copied before the first install so every pip run in the image is constrained,
# including ones a user makes later inside a running container.
COPY docker/constraints.txt /etc/sparc3d-constraints.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        ca-certificates \
        build-essential \
        ninja-build \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && pip3 install --upgrade pip setuptools wheel

WORKDIR /opt/sparc3d

# Install the PyTorch build for this CUDA toolkit first; pip cannot infer the
# CUDA minor from an unsuffixed pin. The +cu124 local version tag is carried
# only by a PyTorch index, so it decides which index each wheel comes from and
# no CPU build can satisfy the pin.
RUN pip3 install --extra-index-url "${TORCH_INDEX_URL}" \
        torch==2.5.0+cu124 torchvision==0.20.0+cu124

COPY . /opt/sparc3d

# flash-attn publishes an sdist rather than a wheel, and its setup.py imports
# torch, which pip's build isolation hides -- resolving it as part of
# ".[inference]" fails with ModuleNotFoundError: No module named 'torch'. Install
# it against this interpreter's torch first, so the extra below finds it already
# satisfied. packaging, ninja and psutil are its build-time imports. Compiling
# every architecture in TORCH_CUDA_ARCH_LIST takes a while; narrow that build
# argument to just your own to cut it down.
RUN pip3 install packaging ninja psutil \
    && FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD}" \
       pip3 install --no-build-isolation flash-attn==2.7.4.post1 \
    && pip3 install ".[inference]"

CMD ["bash"]
