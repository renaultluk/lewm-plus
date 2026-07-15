#!/bin/bash
# Convert a training .ckpt into an eval-ready policy folder on SuperPOD.
#
# Usage:
#   bash superpod/convert_ckpt_to_eval_pt.sh \
#       --src-ckpt /project/<GROUP>/lewm-plus/.stable-wm/checkpoints/<run_id>/<output_model_name>_weights.ckpt \
#       --run-name <eval_policy_name>

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=superpod/_common.sh
source "${SUBMIT_DIR}/superpod/_common.sh"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$PROJECT_DIR"
    mkdir -p outputs

    module load slurm 2>/dev/null || true

    SBATCH_SCRIPT=$(mktemp)
    cat > "$SBATCH_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=lewm-convert
#SBATCH --output=outputs/convert-%j.out
#SBATCH --error=outputs/convert-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=1
#SBATCH --account=${SUPERPOD_ACCOUNT}
#SBATCH --partition=${SUPERPOD_PARTITION_GPU}
#SBATCH --time=00:20:00
exec ${BASH_SOURCE[0]} "\$@"
EOF
    sbatch "$SBATCH_SCRIPT" "$@"
    rm -f "$SBATCH_SCRIPT"
    exit 0
fi

cd "$PROJECT_DIR"
mkdir -p outputs

HAS_CACHE_DIR=false
for arg in "$@"; do
    if [[ "$arg" == "--cache-dir" ]] || [[ "$arg" == --cache-dir=* ]]; then
        HAS_CACHE_DIR=true
    fi
done

EXTRA_ARGS=()
if [[ "$HAS_CACHE_DIR" == false ]]; then
    EXTRA_ARGS+=("--cache-dir" "/workspace/.stable-wm")
fi

MOUNTS="/project:/project,/home:/home,${STABLEWM_HOME}:/workspace/.stable-wm"

module load slurm 2>/dev/null || true

srun --container-image "$CONTAINER_PATH" \
     --container-mounts "$MOUNTS" \
     --container-writable \
     --container-workdir "$PROJECT_DIR" \
     /usr/bin/env STABLEWM_HOME=/workspace/.stable-wm \
     /workspace/.venv/bin/python scripts/convert_train_ckpt_to_eval_pt.py "${EXTRA_ARGS[@]}" "$@"
