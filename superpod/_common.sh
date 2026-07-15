#!/bin/bash
# Common helpers for all SuperPOD scripts.
#
# Usage in other scripts:
#   source "${SUBMIT_DIR}/superpod/_common.sh"
#
# SUBMIT_DIR should be set by the caller to the directory from which the job
# was submitted (the repo root). For batch scripts this is typically
# $SLURM_SUBMIT_DIR; for interactive/local scripts it is the repo root.

set -euo pipefail

# If SUBMIT_DIR is unset, fall back to the repo root derived from this file.
if [[ -z "${SUBMIT_DIR:-}" ]]; then
    SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# Allow callers to override the config directory (useful for tests).
SUPERPOD_CONFIG_DIR="${SUPERPOD_CONFIG_DIR:-${SUBMIT_DIR}/superpod}"
ENV_FILE="${SUPERPOD_CONFIG_DIR}/superpod.env"

if [[ -f "$ENV_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$ENV_FILE"
else
    echo "ERROR: SuperPOD config not found at $ENV_FILE" >&2
    echo "Run: bash superpod/configure_superpod.sh" >&2
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
: "${TRAIN_NODES:=1}"
: "${TRAIN_NTASKS:=1}"
: "${TRAIN_GPUS_PER_NODE:=1}"
: "${CPUS_PER_TASK:=28}"
: "${CPUS_PER_TASK_CPU:=8}"
: "${TRAIN_TIME:=12:00:00}"
: "${STABLEWM_HOME:=${PROJECT_DIR}/.stable-wm}"
: "${SCRATCH_DATA:=/scratch/${SUPERPOD_GROUP}/datasets}"
