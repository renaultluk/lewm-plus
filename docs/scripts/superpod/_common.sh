#!/bin/bash
# Common helpers for all SuperPOD scripts.
#
# Usage in other scripts:
#   source "${SUBMIT_DIR}/docs/scripts/superpod/_common.sh"
#
# SUBMIT_DIR should be set by the caller to the directory from which the job
# was submitted (the repo root). For batch scripts this is typically
# $SLURM_SUBMIT_DIR; for interactive/local scripts it is the repo root.

set -euo pipefail

# Allow callers to override the config directory (useful for tests).
SUPERPOD_CONFIG_DIR="${SUPERPOD_CONFIG_DIR:-${SUBMIT_DIR}/docs/scripts/superpod}"
ENV_FILE="${SUPERPOD_CONFIG_DIR}/superpod.env"

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
