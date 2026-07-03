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
#SBATCH --partition=normal
#SBATCH --time=12:00:00

# LeWorldModel training batch script for HKUST SuperPOD.
#
# Usage:
#   sbatch docs/scripts/superpod/train_lewm.sh data=pusht_h5 trainer.max_epochs=100 output_model_name=pusht_replicate wandb.enabled=false
#
# The script changes to /project/<GROUP>/lewm-plus, activates the container,
# and passes all command-line arguments to python train.py.

set -euo pipefail

PROJECT_DIR="/project/{{GROUP}}/lewm-plus"
CONTAINER="$HOME/containers/lewm.sqsh"

cd "$PROJECT_DIR"
mkdir -p outputs

# Combined list of extra mounts. Add /scratch/<GROUP> if you keep raw data there.
MOUNTS="/project:/project,/home:/home"

srun --container-image "$CONTAINER" \
     --container-mounts "$MOUNTS" \
     --container-writable \
     --container-workdir "$PROJECT_DIR" \
     bash -c "
        export STABLEWM_HOME=${PROJECT_DIR}/.stable-wm
        export WANDB_DISABLED=${WANDB_DISABLED:-false}
        source .venv/bin/activate
        python train.py $@
     "
