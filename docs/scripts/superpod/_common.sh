#!/bin/bash
# Common helpers for all SuperPOD scripts.
#
# Usage in other scripts:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/superpod.env"

if [[ -f "$ENV_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$ENV_FILE"
else
    echo "ERROR: SuperPOD config not found at $ENV_FILE" >&2
    echo "Copy docs/scripts/superpod/superpod.env.example to superpod.env and fill it in." >&2
    exit 1
fi

# Default values for optional variables
: "${SUPERPOD_ACCOUNT:?SUPERPOD_ACCOUNT must be set in superpod.env}"
: "${SUPERPOD_GROUP:?SUPERPOD_GROUP must be set in superpod.env}"
: "${SUPERPOD_USER:?SUPERPOD_USER must be set in superpod.env}"
: "${SUPERPOD_PARTITION_GPU:=normal}"
: "${SUPERPOD_PARTITION_CPU:=cpu}"
: "${PROJECT_DIR:=/project/${SUPERPOD_GROUP}/lewm-plus}"
: "${CONTAINER_PATH:=$HOME/containers/lewm.sqsh}"
: "${CPUS_PER_TASK:=28}"
: "${CPUS_PER_TASK_CPU:=8}"
: "${STABLEWM_HOME:=${PROJECT_DIR}/.stable-wm}"
: "${SCRATCH_DATA:=/scratch/${SUPERPOD_GROUP}/datasets}"
