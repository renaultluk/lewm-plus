#!/bin/bash
#SBATCH --job-name=lewm-eval
#SBATCH --output=outputs/eval-%j.out
#SBATCH --error=outputs/eval-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node=1
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION_GPU}}
#SBATCH --time=06:00:00

# LeWorldModel evaluation batch script for HKUST SuperPOD.
#
# Usage:
#   sbatch scripts/superpod/evaluate_lewm.sh \
#       --config-name=pusht.yaml policy=pusht/lewm eval.num_eval=50
#
# All command-line arguments are forwarded to python eval.py.

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=scripts/superpod/_common.sh
source "${SUBMIT_DIR}/scripts/superpod/_common.sh"

cd "$PROJECT_DIR"
mkdir -p outputs

# Combined list of extra mounts. Add /scratch/<GROUP> if you keep raw data there.
MOUNTS="/project:/project,/home:/home"

# Forward all remaining arguments to eval.py. Use $@ directly instead of
# wrapping in bash -c so that Hydra overrides are preserved.
srun --container-image "$CONTAINER_PATH" \
     --container-mounts "$MOUNTS" \
     --container-writable \
     --container-workdir "$PROJECT_DIR" \
     --environment="STABLEWM_HOME=${STABLEWM_HOME}" \
     --environment="PATH=/workspace/.venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
     --environment="VIRTUAL_ENV=/workspace/.venv" \
     /workspace/.venv/bin/python eval.py "$@"
