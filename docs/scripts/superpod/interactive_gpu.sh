#!/bin/bash
# Request an interactive GPU session on SuperPOD.
#
# Usage:
#   bash docs/scripts/superpod/interactive_gpu.sh [hours]
#
# Default duration is 1 hour. The session lands on a compute node where you
# can debug commands, inspect GPUs, or run short tests.

set -euo pipefail

# shellcheck source=docs/scripts/superpod/_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

HOURS="${1:-1}"
# Format as HH:MM:SS
TIME=$(printf "%02d:00:00" "$HOURS")

srun --account="${SUPERPOD_ACCOUNT}" \
     --partition="${SUPERPOD_PARTITION_GPU}" \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task="${CPUS_PER_TASK}" \
     --gpus-per-node=1 \
     --time="$TIME" \
     --container-image "$CONTAINER_PATH" \
     --container-mounts "/project:/project,/home:/home" \
     --container-writable \
     --container-workdir "$PROJECT_DIR" \
     --pty bash
