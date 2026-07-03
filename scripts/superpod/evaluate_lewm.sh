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
# All arguments are forwarded to python eval.py.

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
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
        source .venv/bin/activate
        python eval.py \$@
     "
