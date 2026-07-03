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
# The script sources superpod.env, changes to $PROJECT_DIR, activates the
# container, and passes all command-line arguments to python train.py.

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=scripts/superpod/_common.sh
source "${SUBMIT_DIR}/scripts/superpod/_common.sh"

cd "$PROJECT_DIR"
mkdir -p outputs

# Combined list of extra mounts. Add /scratch/<GROUP> if you keep raw data there.
MOUNTS="/project:/project,/home:/home"

srun --container-image "$CONTAINER_PATH" \
     --container-mounts "$MOUNTS" \
     --container-writable \
     --container-workdir "$PROJECT_DIR" \
     bash -c "
        export STABLEWM_HOME=${STABLEWM_HOME}
        export WANDB_DISABLED=${WANDB_DISABLED:-false}
        # The container already ships its venv at /workspace/.venv.
        export PATH=/workspace/.venv/bin:/root/.local/bin:\$PATH
        export VIRTUAL_ENV=/workspace/.venv
        python train.py \$@
     "
