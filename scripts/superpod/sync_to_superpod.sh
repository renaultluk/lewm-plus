#!/bin/bash
# Sync the local lewm-plus project to SuperPOD /project storage.
#
# Usage:
#   bash scripts/superpod/sync_to_superpod.sh
#
# Excludes heavy transient directories (.stable-wm/datasets, .venv, outputs,
# .git, etc.) so only code, configs, and helper scripts are copied.

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# shellcheck source=scripts/superpod/_common.sh
source "${SUBMIT_DIR}/scripts/superpod/_common.sh"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REMOTE="${SUPERPOD_USER}@superpod.ust.hk:${PROJECT_DIR}"

echo "Syncing $LOCAL_DIR -> $REMOTE"

rsync -avP \
    --exclude='.git' \
    --exclude='.stable-wm' \
    --exclude='.venv' \
    --exclude='outputs' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.egg-info' \
    --exclude='.pytest_cache' \
    --exclude='*.ckpt' \
    --exclude='*.pt' \
    --exclude='*.h5' \
    --exclude='*.zst' \
    --exclude='*.lance' \
    "$LOCAL_DIR/" "$REMOTE/"

echo "Done. Datasets, checkpoints, and .venv are excluded; sync them separately if needed."
