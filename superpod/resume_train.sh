#!/bin/bash
# Resume-friendly training helper for long runs on SuperPOD walltime limits.
#
# Usage:
#   bash superpod/resume_train.sh [run_id] [first_target_epoch] [final_target_epoch]
#
# Example:
#   bash superpod/resume_train.sh pusht_h5_replicate_run 70 100

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=superpod/_common.sh
source "${SUBMIT_DIR}/superpod/_common.sh"

RUN_ID="${1:-pusht_h5_replicate_run}"
FIRST_TARGET_EPOCH="${2:-70}"
FINAL_TARGET_EPOCH="${3:-100}"

if ! [[ "$FIRST_TARGET_EPOCH" =~ ^[0-9]+$ ]] || ! [[ "$FINAL_TARGET_EPOCH" =~ ^[0-9]+$ ]]; then
    echo "ERROR: epoch targets must be integers" >&2
    exit 1
fi

if (( FIRST_TARGET_EPOCH <= 0 || FINAL_TARGET_EPOCH <= 0 )); then
    echo "ERROR: epoch targets must be > 0" >&2
    exit 1
fi

if (( FINAL_TARGET_EPOCH < FIRST_TARGET_EPOCH )); then
    echo "ERROR: final_target_epoch must be >= first_target_epoch" >&2
    exit 1
fi

echo "Submitting stage 1 to epoch ${FIRST_TARGET_EPOCH} for run_id=${RUN_ID}"
bash "${SUBMIT_DIR}/superpod/train_lewm.sh" \
    experiment=pusht_h5_replicate \
    trainer.max_epochs="${FIRST_TARGET_EPOCH}" \
    subdir="${RUN_ID}"

echo ""
echo "After stage 1 reaches epoch ${FIRST_TARGET_EPOCH}, resume with:"
echo "bash superpod/train_lewm.sh experiment=pusht_h5_replicate trainer.max_epochs=${FINAL_TARGET_EPOCH} subdir=${RUN_ID}"
