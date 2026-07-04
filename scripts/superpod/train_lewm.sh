#!/bin/bash
#SBATCH --job-name=lewm-train
#SBATCH --output=outputs/train-%j.out
#SBATCH --error=outputs/train-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node=1
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION_GPU}}
#SBATCH --time=12:00:00

# LeWorldModel training batch script for HKUST SuperPOD.
#
# Usage:
#   sbatch scripts/superpod/train_lewm.sh \
#       data=pusht_h5 trainer.max_epochs=100 output_model_name=pusht_replicate wandb.enabled=false
#
# All command-line arguments are forwarded to python train.py.

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=scripts/superpod/_common.sh
source "${SUBMIT_DIR}/scripts/superpod/_common.sh"

cd "$PROJECT_DIR"
mkdir -p outputs

# Combined list of extra mounts. Add /scratch/<GROUP> if you keep raw data there.
MOUNTS="/project:/project,/home:/home"

# Forward all remaining arguments to train.py. Use $@ directly instead of
# wrapping in bash -c so that Hydra overrides are preserved.
srun --container-image "$CONTAINER_PATH" \
     --container-mounts "$MOUNTS" \
     --container-writable \
     --container-workdir "$PROJECT_DIR" \
     --environment="STABLEWM_HOME=${STABLEWM_HOME}" \
     --environment="WANDB_DISABLED=${WANDB_DISABLED:-false}" \
     --environment="PATH=/workspace/.venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
     --environment="VIRTUAL_ENV=/workspace/.venv" \
     /workspace/.venv/bin/python train.py "$@"
