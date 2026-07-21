#!/bin/bash
# Generic LeWM training sbatch template for HKUST SuperPOD.
#
# Usage:
#   sbatch superpod/train_job.sbatch data=reacher_reach_random experiment=pretrain \
#     trainer.max_epochs=50 output_model_name=my_run wandb.enabled=false
#
# All arguments are passed through to train.py.

#SBATCH --job-name=lewm-train
#SBATCH --output=outputs/train-%j.out
#SBATCH --error=outputs/train-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=4
#SBATCH --account=mscbdt2024
#SBATCH --partition=normal
#SBATCH --time=12:00:00

cd /project/mscbdt2024/lewm-plus
mkdir -p outputs

source superpod/superpod.env

MOUNTS="/project:/project,/home:/home,${STABLEWM_HOME}:/workspace/.stable-wm"

srun --ntasks=4 \
     --container-image "$HOME/containers/lewm.sqsh" \
     --container-mounts "$MOUNTS" \
     --container-writable \
     --container-workdir /project/mscbdt2024/lewm-plus \
     /usr/bin/env STABLEWM_HOME=/workspace/.stable-wm SPT_CACHE_DIR=/workspace/.stable-wm \
     /workspace/.venv/bin/python train.py "$@"
