#!/bin/bash
# Migrate old LeWM checkpoint layouts into the unified run directory layout.
#
# Old layouts this script consolidates:
#  - ${STABLEWM_HOME}/checkpoints/<output_model_name>/weights_epoch_*.pt
#  - ${STABLEWM_HOME}/checkpoints/<output_model_name>_weights.ckpt
#  - last.ckpt under stable-pretraining run cache locations
#
# New unified layout target:
#  - ${STABLEWM_HOME}/checkpoints/<run_id>/
#      - weights_epoch_*.pt
#      - last.ckpt
#      - <output_model_name>_weights.ckpt
#
# Usage:
#   bash superpod/migrate_checkpoints.sh <run_id> <output_model_name>
#
# Example:
#   bash superpod/migrate_checkpoints.sh pusht_h5_replicate_run pusht_h5_replicate

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=superpod/_common.sh
source "${SUBMIT_DIR}/superpod/_common.sh"

RUN_ID="${1:-}"
OUTPUT_MODEL_NAME="${2:-}"

if [[ -z "$RUN_ID" || -z "$OUTPUT_MODEL_NAME" ]]; then
    echo "Usage: bash superpod/migrate_checkpoints.sh <run_id> <output_model_name>" >&2
    exit 1
fi

CHECKPOINTS_ROOT="${STABLEWM_HOME}/checkpoints"
TARGET_DIR="${CHECKPOINTS_ROOT}/${RUN_ID}"
OLD_PT_DIR="${CHECKPOINTS_ROOT}/${OUTPUT_MODEL_NAME}"
OLD_FLAT_CKPT="${CHECKPOINTS_ROOT}/${OUTPUT_MODEL_NAME}_weights.ckpt"
TARGET_ALIAS_CKPT="${TARGET_DIR}/${OUTPUT_MODEL_NAME}_weights.ckpt"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "ERROR: python3/python not found on PATH; required for checkpoint discovery." >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"

echo "Checkpoint root: $CHECKPOINTS_ROOT"
echo "Target run dir : $TARGET_DIR"
echo ""

copy_if_exists() {
    local src="$1"
    local dst="$2"
    if [[ -f "$src" ]]; then
        cp -f "$src" "$dst"
        echo "Copied: $src -> $dst"
    fi
}

# 1) Migrate old .pt export directory if present.
if [[ -d "$OLD_PT_DIR" ]]; then
    shopt -s nullglob
    PT_FILES=("${OLD_PT_DIR}"/weights_epoch_*.pt)
    if (( ${#PT_FILES[@]} > 0 )); then
        cp -f "${OLD_PT_DIR}"/weights_epoch_*.pt "$TARGET_DIR"/
        echo "Copied ${#PT_FILES[@]} weights_epoch_*.pt file(s) from ${OLD_PT_DIR}"
    fi
    shopt -u nullglob

    copy_if_exists "${OLD_PT_DIR}/config.json" "${TARGET_DIR}/config.json"
fi

# 2) Migrate flat alias ckpt if present.
copy_if_exists "$OLD_FLAT_CKPT" "$TARGET_ALIAS_CKPT"

# 3) If last.ckpt is missing in target, backfill from likely stable-pretraining caches.
if [[ ! -f "${TARGET_DIR}/last.ckpt" ]]; then
    LAST_CANDIDATE=$(STABLEWM_HOME="$STABLEWM_HOME" $PYTHON_BIN - <<'PY'
from pathlib import Path
import os

cands = []
stablewm = Path(os.environ["STABLEWM_HOME"])
home = Path.home()

for root in [stablewm / "runs", home / ".cache" / "stable-pretraining" / "runs"]:
    if root.exists():
        cands.extend(root.rglob("checkpoints/last.ckpt"))

if not cands:
    print("")
else:
    cands.sort(key=lambda p: p.stat().st_mtime)
    print(cands[-1])
PY
)

    if [[ -n "$LAST_CANDIDATE" && -f "$LAST_CANDIDATE" ]]; then
        cp -f "$LAST_CANDIDATE" "${TARGET_DIR}/last.ckpt"
        echo "Copied latest last.ckpt: $LAST_CANDIDATE -> ${TARGET_DIR}/last.ckpt"
    fi
fi

# 4) Ensure resume alias exists; if not, derive it from last.ckpt.
if [[ ! -f "$TARGET_ALIAS_CKPT" && -f "${TARGET_DIR}/last.ckpt" ]]; then
    cp -f "${TARGET_DIR}/last.ckpt" "$TARGET_ALIAS_CKPT"
    echo "Created alias from last.ckpt: $TARGET_ALIAS_CKPT"
fi

echo ""
echo "Final files in target run dir:"
ls -lah "$TARGET_DIR"

LATEST_PT=$(TARGET_DIR="$TARGET_DIR" $PYTHON_BIN - <<'PY'
from pathlib import Path
import os

target = Path(os.environ["TARGET_DIR"])
pts = sorted(target.glob("weights_epoch_*.pt"), key=lambda p: p.stat().st_mtime)
print(pts[-1].name if pts else "")
PY
)

echo ""
if [[ -n "$LATEST_PT" ]]; then
    echo "Suggested eval policy override:"
    echo "  policy=${RUN_ID}/${LATEST_PT}"
else
    echo "No weights_epoch_*.pt found in ${TARGET_DIR}."
    echo "If only .ckpt exists, convert/export to .pt before eval with:"
    echo "  python scripts/convert_train_ckpt_to_eval_pt.py --src-ckpt \"${TARGET_ALIAS_CKPT}\" --run-name \"${RUN_ID}_eval\" --cache-dir \"${STABLEWM_HOME}\""
    echo "Then evaluate with policy=${RUN_ID}_eval"
fi
