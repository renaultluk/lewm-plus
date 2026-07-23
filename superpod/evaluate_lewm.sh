#!/bin/bash
# LeWorldModel evaluation batch script for HKUST SuperPOD.
#
# Usage:
#   bash superpod/evaluate_lewm.sh \
#       --config-name=pusht.yaml policy=pusht/lewm eval.num_eval=50
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
#SBATCH --job-name=lewm-eval
#SBATCH --output=outputs/eval-%j.out
#SBATCH --error=outputs/eval-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=1
#SBATCH --account=${SUPERPOD_ACCOUNT}
#SBATCH --partition=${SUPERPOD_PARTITION_GPU}
#SBATCH --time=06:00:00
exec ${BASH_SOURCE[0]} "\$@"
EOF
    sbatch "$SBATCH_SCRIPT" "$@"
    rm -f "$SBATCH_SCRIPT"
    exit 0
fi

# Job body starts here.
cd "$PROJECT_DIR"
mkdir -p outputs

# Keep evaluation dataset/cache resolution under the mounted STABLEWM_HOME
# inside the container, regardless of host defaults.
CONTAINER_STABLEWM_HOME="/workspace/.stable-wm"

HAS_CACHE_DIR_OVERRIDE=false
HAS_DATASET_NAME_OVERRIDE=false
USE_PUSHT_EVAL=false
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == cache_dir=* ]]; then
        HAS_CACHE_DIR_OVERRIDE=true
    fi
    if [[ "$arg" == eval.dataset_name=* ]]; then
        HAS_DATASET_NAME_OVERRIDE=true
    fi
    if [[ "$arg" == "--config-name=pusht.yaml" ]] || [[ "$arg" == "--config-name=pusht" ]] || [[ "$arg" == "experiment=pusht_h5_replicate" ]] || [[ "$arg" == "+experiment=pusht_h5_replicate" ]]; then
        USE_PUSHT_EVAL=true
    fi
done

if [[ "$HAS_CACHE_DIR_OVERRIDE" == false ]]; then
    EXTRA_ARGS+=("cache_dir=${CONTAINER_STABLEWM_HOME}")
fi

if [[ "$USE_PUSHT_EVAL" == true ]] && [[ "$HAS_DATASET_NAME_OVERRIDE" == false ]]; then
    EXPECTED_H5="${STABLEWM_HOME}/datasets/pusht_expert_train.h5"
    NESTED_H5="${STABLEWM_HOME}/datasets/pusht/pusht_expert_train.h5"
    CONTAINER_NESTED_H5="${CONTAINER_STABLEWM_HOME}/datasets/pusht/pusht_expert_train.h5"

    if [[ -f "$EXPECTED_H5" ]]; then
        :
    elif [[ -f "$NESTED_H5" ]]; then
        echo "INFO: using nested PushT HDF5 path for eval: $NESTED_H5" >&2
        EXTRA_ARGS+=("eval.dataset_name=${CONTAINER_NESTED_H5}")
    fi
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
     /usr/bin/env STABLEWM_HOME="${CONTAINER_STABLEWM_HOME}" \
     /bin/bash -c 'mkdir -p /tmp/dm_pkgs && /root/.local/bin/uv pip install -q --target /tmp/dm_pkgs dm_control 2>/dev/null && PYTHONPATH=/tmp/dm_pkgs exec /workspace/.venv/bin/python eval.py "$@"' _ "${EXTRA_ARGS[@]}" "$@"
