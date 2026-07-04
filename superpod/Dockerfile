# Container for running LeWorldModel on HKUST SuperPOD
# Build locally:   docker build -f superpod/Dockerfile -t lewm:superpod .
# Convert:         enroot import -o lewm.sqsh dockerd://lewm:superpod
# Upload:          rsync -avP lewm.sqsh <USER>@superpod.ust.hk:~/containers/

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV UV_LINK_MODE=copy

# Add deadsnakes PPA for Python 3.10 on Ubuntu 24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && rm -rf /var/lib/apt/lists/*

# Minimal runtime and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgtk-3-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    swig \
    tcl \
    tclsh \
    zstd \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

# If the repo root has pyproject.toml / uv.lock, use them. Otherwise fall back
# to a pinned package list. The COPY is split so missing files do not abort the
# build; the dependency installation happens in a single RUN step.
COPY pyproject.toml* uv.lock* ./
RUN if [ -f pyproject.toml ] && [ -f uv.lock ]; then \
      echo "Installing from uv.lock..." && \
      uv venv --python=3.10 /workspace/.venv && \
      VIRTUAL_ENV=/workspace/.venv uv pip install --no-cache -r pyproject.toml; \
    elif [ -f pyproject.toml ]; then \
      echo "Installing from pyproject.toml..." && \
      uv venv --python=3.10 /workspace/.venv && \
      grep -qi 'stable-worldmodel' pyproject.toml || \
        VIRTUAL_ENV=/workspace/.venv uv pip install --no-cache \
          --index-url https://download.pytorch.org/whl/cu126 torch torchvision && \
      VIRTUAL_ENV=/workspace/.venv uv pip install --no-cache -r pyproject.toml; \
    else \
      echo "No pyproject.toml found; installing pinned fallback packages..." && \
      uv venv --python=3.10 /workspace/.venv && \
      VIRTUAL_ENV=/workspace/.venv uv pip install --no-cache \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch torchvision && \
      VIRTUAL_ENV=/workspace/.venv uv pip install --no-cache \
        stable-worldmodel[train] \
        datasets \
        imageio imageio-ffmpeg moviepy \
        gymnasium[mujoco] \
        pygame pymunk shapely hdf5plugin; \
    fi

# The repo code is mounted at runtime via --container-mounts, so we do not
# copy the source into the image. This keeps the image reusable.
# On SuperPOD we mount the host's PROJECT_DIR/.stable-wm to /workspace/.stable-wm.

ENV STABLEWM_HOME=/workspace/.stable-wm
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="/workspace/.venv/bin:${PATH}"

CMD ["/bin/bash"]
