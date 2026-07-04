#!/bin/bash
# LeWorldModel training batch script for HKUST SuperPOD.
#
# Usage:
#   bash superpod/train_lewm.sh data=pusht_h5 trainer.max_epochs=100 output_model_name=pusht_replicate wandb.enabled=false
#
# This script reads superpod/superpod.env and self-submits to Slurm.
# No separate configure step is required after superpod.env exists.

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
#SBATCH --job-name=lewm-train
#SBATCH --output=outputs/train-%j.out
#SBATCH --error=outputs/train-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=1
#SBATCH --account=${SUPERPOD_ACCOUNT}
#SBATCH --partition=${SUPERPOD_PARTITION_GPU}
#SBATCH --time=12:00:00
exec ${BASH_SOURCE[0]} "\$@"
EOF
    sbatch "$SBATCH_SCRIPT" "$@"
    rm -f "$SBATCH_SCRIPT"
    exit 0
fi

# Job body starts here.
cd "$PROJECT_DIR"
mkdir -p outputs

# Combined list of extra mounts. Add /scratch/<GROUP> if you keep raw data there.
MOUNTS="/project:/project,/home:/home"

# Set environment variables that need to be visible inside the container.
# --container-env preserves them from the host Slurm step.
export STABLEWM_HOME="${STABLEWM_HOME}"
export WANDB_DISABLED="${WANDB_DISABLED:-false}"
export PATH=/workspace/.venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export VIRTUAL_ENV=/workspace/.venv

srun --container-image "$CONTAINER_PATH" \
     --container-mounts "$MOUNTS" \
     --container-writable \
     --container-workdir "$PROJECT_DIR" \
     --container-env=STABLEWM_HOME,WANDB_DISABLED,PATH,VIRTUAL_ENV \
     /workspace/.venv/bin/python train.py "$@"
