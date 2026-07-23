#!/bin/bash
#SBATCH --job-name=lewm-eval
#SBATCH --output=outputs/eval-%j.out
#SBATCH --error=outputs/eval-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --account=mscbdt2024
#SBATCH --partition=normal
#SBATCH --time=06:00:00

cd /project/mscbdt2024/lewm-plus
mkdir -p outputs

source superpod/superpod.env

MOUNTS="/project:/project,/home:/home,${STABLEWM_HOME}:/workspace/.stable-wm"

srun --ntasks=1 \
     --container-image "$HOME/containers/lewm.sqsh" \
     --container-mounts "$MOUNTS" \
     --container-writable \
     --container-workdir /project/mscbdt2024/lewm-plus \
     /usr/bin/env STABLEWM_HOME=/workspace/.stable-wm \
     /bin/bash -c 'mkdir -p /tmp/dm_pkgs && /root/.local/bin/uv pip install -q --target /tmp/dm_pkgs dm_control 2>/dev/null && PYTHONPATH=/tmp/dm_pkgs exec /workspace/.venv/bin/python eval.py "$@"' _ "$@"
