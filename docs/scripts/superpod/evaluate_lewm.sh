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
#SBATCH --partition=normal
#SBATCH --time=06:00:00

# LeWorldModel evaluation batch script for HKUST SuperPOD.
#
# Usage:
#   sbatch docs/scripts/superpod/evaluate_lewm.sh --config-name=pusht.yaml policy=pusht/lewm eval.num_eval=50
#
# All arguments are forwarded to python eval.py.

set -euo pipefail

PROJECT_DIR="/project/{{GROUP}}/lewm-plus"
CONTAINER="$HOME/containers/lewm.sqsh"

cd "$PROJECT_DIR"
mkdir -p outputs

MOUNTS="/project:/project,/home:/home"

srun --container-image "$CONTAINER" \
     --container-mounts "$MOUNTS" \
     --container-writable \
     --container-workdir "$PROJECT_DIR" \
     bash -c "
        export STABLEWM_HOME=${PROJECT_DIR}/.stable-wm
        source .venv/bin/activate
        python eval.py $@
     "
