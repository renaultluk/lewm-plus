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

    # SuperPOD login nodes require the slurm module to be loaded before
    # sbatch/srun can be used.
    module load slurm 2>/dev/null || true

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
#SBATCH --time=${TRAIN_TIME}
exec ${BASH_SOURCE[0]} "\$@"
EOF
    sbatch "$SBATCH_SCRIPT" "$@"
    rm -f "$SBATCH_SCRIPT"
    exit 0
fi

# Job body starts here.
cd "$PROJECT_DIR"
mkdir -p outputs

# Quick preflight for the official PushT HDF5 path. The dataset resolver expects
# the file at $STABLEWM_HOME/datasets/pusht_expert_train.h5 (not nested).
USE_PUSHT_H5=false
HAS_DATASET_NAME_OVERRIDE=false
for arg in "$@"; do
    if [[ "$arg" == "data=pusht_h5" ]]; then
        USE_PUSHT_H5=true
    fi
    if [[ "$arg" == "experiment=pusht_h5_replicate" ]] || [[ "$arg" == "+experiment=pusht_h5_replicate" ]]; then
        USE_PUSHT_H5=true
    fi
    if [[ "$arg" == data.dataset.name=* ]]; then
        HAS_DATASET_NAME_OVERRIDE=true
    fi
done

DATASET_NAME_OVERRIDE=""
if [[ "$USE_PUSHT_H5" == true ]]; then
    EXPECTED_H5="${STABLEWM_HOME}/datasets/pusht_expert_train.h5"
    NESTED_H5="${STABLEWM_HOME}/datasets/pusht/pusht_expert_train.h5"
    NESTED_ZST="${STABLEWM_HOME}/datasets/pusht/pusht_expert_train.h5.zst"
    CONTAINER_NESTED_H5="/workspace/.stable-wm/datasets/pusht/pusht_expert_train.h5"

    if [[ "$HAS_DATASET_NAME_OVERRIDE" == true ]]; then
        :
    elif [[ -f "$EXPECTED_H5" ]]; then
        :
    elif [[ -f "$NESTED_H5" ]]; then
        echo "INFO: using nested PushT HDF5 path: $NESTED_H5" >&2
        DATASET_NAME_OVERRIDE="data.dataset.name=${CONTAINER_NESTED_H5}"
    else
        echo "ERROR: expected dataset not found: $EXPECTED_H5" >&2

        if [[ -f "$NESTED_ZST" ]]; then
            echo "Found compressed archive at: $NESTED_ZST" >&2
            echo "Extract and move with:" >&2
            echo "  mkdir -p \"${STABLEWM_HOME}/datasets\"" >&2
            echo "  zstd -d \"$NESTED_ZST\" -o \"$EXPECTED_H5\"" >&2
        fi

        exit 1
    fi
fi

EXTRA_ARGS=()
if [[ -n "$DATASET_NAME_OVERRIDE" ]]; then
    EXTRA_ARGS+=("$DATASET_NAME_OVERRIDE")
fi

# Combined list of extra mounts. We also map the host's .stable-wm directory
# to /workspace/.stable-wm inside the container, which matches the Dockerfile.
MOUNTS="/project:/project,/home:/home,${STABLEWM_HOME}:/workspace/.stable-wm"

# On some compute nodes srun is a wrapper that requires the slurm module.
module load slurm 2>/dev/null || true

srun --container-image "$CONTAINER_PATH" \
     --container-mounts "$MOUNTS" \
     --container-writable \
     --container-workdir "$PROJECT_DIR" \
     /workspace/.venv/bin/python train.py "${EXTRA_ARGS[@]}" "$@"
