#!/bin/bash
# Request an interactive GPU session on SuperPOD.
#
# Usage:
#   bash docs/scripts/superpod/interactive_gpu.sh [hours]
#
# Default duration is 1 hour. The session lands on a compute node where you
# can debug commands, inspect GPUs, or run short tests.

set -euo pipefail

HOURS="${1:-1}"
# Format as HH:MM:SS
TIME=$(printf "%02d:00:00" "$HOURS")

srun --account={{ACCOUNT}} \
     --partition=normal \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=28 \
     --gpus-per-node=1 \
     --time="$TIME" \
     --container-image "$HOME/containers/lewm.sqsh" \
     --container-mounts "/project:/project,/home:/home" \
     --container-writable \
     --container-workdir "/project/{{GROUP}}/lewm-plus" \
     --pty bash
