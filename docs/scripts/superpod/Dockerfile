# Container for running LeWorldModel on HKUST SuperPOD
# Build locally:   docker build -f docs/scripts/superpod/Dockerfile -t lewm:superpod .
# Convert:         docker save lewm:superpod | enroot import -o lewm.sqsh -
# Upload:          rsync -avP lewm.sqsh <USER>@superpod.ust.hk:~/containers/

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV UV_LINK_MODE=copy

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
    zstd \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

# Copy dependency file and install Python packages first for layer caching
COPY pyproject.toml uv.lock* ./
RUN if [ -f uv.lock ]; then \
      uv venv --python=3.10 /workspace/.venv && \
      VIRTUAL_ENV=/workspace/.venv uv pip install --no-cache -r pyproject.toml; \
    else \
      uv venv --python=3.10 /workspace/.venv && \
      VIRTUAL_ENV=/workspace/.venv uv pip install --no-cache \
        torch torchvision \
        stable-worldmodel[train] \
        datasets \
        imageio imageio-ffmpeg moviepy \
        gymnasium[mujoco] \
        pygame pymunk shapely hdf5plugin; \
    fi

# The repo code is mounted at runtime via --container-mounts, so we do not
# copy the source into the image. This keeps the image reusable.

ENV STABLEWM_HOME=/workspace/.stable-wm
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="/workspace/.venv/bin:${PATH}"

CMD ["/bin/bash"]
