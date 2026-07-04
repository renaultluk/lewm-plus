#!/bin/bash
# GPU hello-world job for SuperPOD.
# It runs inside the LeWM container and verifies GPU access, PyTorch, and
# the superpod.env setup from a GPU compute node.
#
# Usage:
#   bash superpod/hello_superpod_gpu.sh

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=superpod/_common.sh
source "${SUBMIT_DIR}/superpod/_common.sh"

# If not already inside a Slurm job, generate a batch script and submit it.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$PROJECT_DIR"
    mkdir -p outputs

    SBATCH_SCRIPT=$(mktemp)
    cat > "$SBATCH_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=hello-superpod-gpu
#SBATCH --output=outputs/hello-gpu-%j.out
#SBATCH --error=outputs/hello-gpu-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --account=${SUPERPOD_ACCOUNT}
#SBATCH --partition=${SUPERPOD_PARTITION_GPU}
#SBATCH --time=00:05:00
exec ${BASH_SOURCE[0]} "\$@"
EOF
    sbatch "$SBATCH_SCRIPT" "$@"
    rm -f "$SBATCH_SCRIPT"
    exit 0
fi

# Job body starts here.
echo "========================================"
echo "Hello from SuperPOD GPU node!"
echo "========================================"
echo ""
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "PWD: $(pwd)"
echo ""
echo "--- Slurm environment ---"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-<unset>}"
echo "SLURM_SUBMIT_HOST: ${SLURM_SUBMIT_HOST:-<unset>}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-<unset>}"
echo "SLURM_GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE:-<unset>}"
echo ""
echo "--- Host GPU info (before container) ---"
nvidia-smi || echo "nvidia-smi not available on host"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo ""

echo "--- SuperPOD env loaded from superpod.env ---"
echo "SUPERPOD_ACCOUNT: ${SUPERPOD_ACCOUNT}"
echo "SUPERPOD_PARTITION_GPU: ${SUPERPOD_PARTITION_GPU}"
echo "SUPERPOD_GROUP: ${SUPERPOD_GROUP}"
echo "SUPERPOD_USER: ${SUPERPOD_USER}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "CONTAINER_PATH: ${CONTAINER_PATH}"
echo ""

echo "--- Container / GPU check ---"
if [[ ! -f "$CONTAINER_PATH" ]]; then
    echo "ERROR: container not found at $CONTAINER_PATH" >&2
    echo "Build and upload it first. See docs/SUPERPOD_GUIDE.md section 4." >&2
    exit 1
fi

echo "Container: $CONTAINER_PATH"

# Pyxis should automatically bind NVIDIA devices for GPU jobs. If it does not,
# run 'scontrol show config | grep -i gres' and check that gres/gpu is configured.
srun --container-image "$CONTAINER_PATH" \
     --container-mounts "/project:/project,/home:/home" \
     --container-writable \
     --container-workdir "$PROJECT_DIR" \
     bash -c '
        echo "Inside container:"
        echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
        nvidia-smi || echo "nvidia-smi not found in container"
        python --version
        python - <<"PY"
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
PY
     '

echo ""
echo "========================================"
echo "GPU hello-world job completed successfully."
echo "========================================"
