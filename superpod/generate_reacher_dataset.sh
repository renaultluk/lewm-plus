#!/bin/bash
# Full-scale Reacher task-agnostic dataset generation for HKUST SuperPOD.
#
# Usage:
#   bash superpod/generate_reacher_dataset.sh
#
# Optional overrides (passed as KEY=VALUE):
#   EPISODES=10000            # default: 10000 (about 1M transitions at 100 steps)
#   MAX_STEPS=100             # default: 100
#   IMAGE_SIZE=224            # default: 224
#   OUTPUT_NAME=reacher_task_agnostic_train_224
#   DATASET_TIME=24:00:00     # Slurm walltime for this job
#   RUN_VERIFY=true           # run push-quality verifier after generation (default true)
#   MUJOCO_GL_BACKEND=egl     # headless renderer backend: egl|osmesa
#
# Examples:
#   bash superpod/generate_reacher_dataset.sh EPISODES=20000 MAX_STEPS=100
#   bash superpod/generate_reacher_dataset.sh OUTPUT_NAME=reacher_task_agnostic_train_v2 DATASET_TIME=36:00:00

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=superpod/_common.sh
source "${SUBMIT_DIR}/superpod/_common.sh"

# Defaults (can be overridden by KEY=VALUE args).
EPISODES="${EPISODES:-10000}"
MAX_STEPS="${MAX_STEPS:-100}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
OUTPUT_NAME="${OUTPUT_NAME:-reacher_task_agnostic_train_224}"
DATASET_TIME="${DATASET_TIME:-24:00:00}"
RUN_VERIFY="${RUN_VERIFY:-true}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-egl}"

for arg in "$@"; do
    case "$arg" in
        EPISODES=*) EPISODES="${arg#*=}" ;;
        MAX_STEPS=*) MAX_STEPS="${arg#*=}" ;;
        IMAGE_SIZE=*) IMAGE_SIZE="${arg#*=}" ;;
        OUTPUT_NAME=*) OUTPUT_NAME="${arg#*=}" ;;
        DATASET_TIME=*) DATASET_TIME="${arg#*=}" ;;
        RUN_VERIFY=*) RUN_VERIFY="${arg#*=}" ;;
        MUJOCO_GL_BACKEND=*) MUJOCO_GL_BACKEND="${arg#*=}" ;;
        *)
            echo "ERROR: unknown argument '$arg'" >&2
            echo "Supported: EPISODES=..., MAX_STEPS=..., IMAGE_SIZE=..., OUTPUT_NAME=..., DATASET_TIME=..., RUN_VERIFY=..., MUJOCO_GL_BACKEND=..." >&2
            exit 2
            ;;
    esac
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$PROJECT_DIR"
    mkdir -p outputs

    module load slurm 2>/dev/null || true

    SBATCH_SCRIPT=$(mktemp)
    cat > "$SBATCH_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=reacher-data
#SBATCH --output=outputs/reacher-data-%j.out
#SBATCH --error=outputs/reacher-data-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=1
#SBATCH --account=${SUPERPOD_ACCOUNT}
#SBATCH --partition=${SUPERPOD_PARTITION_GPU}
#SBATCH --time=${DATASET_TIME}
exec ${BASH_SOURCE[0]} "\$@"
EOF
    sbatch "$SBATCH_SCRIPT" \
        "EPISODES=${EPISODES}" \
        "MAX_STEPS=${MAX_STEPS}" \
        "IMAGE_SIZE=${IMAGE_SIZE}" \
        "OUTPUT_NAME=${OUTPUT_NAME}" \
        "DATASET_TIME=${DATASET_TIME}" \
        "RUN_VERIFY=${RUN_VERIFY}" \
        "MUJOCO_GL_BACKEND=${MUJOCO_GL_BACKEND}"
    rm -f "$SBATCH_SCRIPT"
    exit 0
fi

cd "$PROJECT_DIR"
mkdir -p outputs "${STABLEWM_HOME}/datasets"

OUT_DATASET="/workspace/.stable-wm/datasets/${OUTPUT_NAME}.lance"
MOUNTS="/project:/project,/home:/home,${STABLEWM_HOME}:/workspace/.stable-wm"

module load slurm 2>/dev/null || true

echo "Generating dataset: ${OUT_DATASET}" >&2
echo "episodes=${EPISODES} max_steps=${MAX_STEPS} image_size=${IMAGE_SIZE}" >&2
echo "headless renderer: MUJOCO_GL=${MUJOCO_GL_BACKEND}" >&2

srun --container-image "$CONTAINER_PATH" \
     --container-mounts "$MOUNTS" \
     --container-writable \
     --container-workdir "$PROJECT_DIR" \
     /usr/bin/env MUJOCO_GL="${MUJOCO_GL_BACKEND}" PYOPENGL_PLATFORM="${MUJOCO_GL_BACKEND}" DISPLAY= \
     /workspace/.venv/bin/python scripts/generate_mujoco_dataset.py \
     --env ReacherTaskAgnostic-v0 \
     --policy reacher_multitask \
     --episodes "$EPISODES" \
     --max_steps "$MAX_STEPS" \
     --image_size "$IMAGE_SIZE" \
     --output "$OUT_DATASET"

if [[ "${RUN_VERIFY}" == "true" ]]; then
    echo "Running push-quality verification..." >&2
    srun --container-image "$CONTAINER_PATH" \
         --container-mounts "$MOUNTS" \
         --container-writable \
         --container-workdir "$PROJECT_DIR" \
         /usr/bin/env MUJOCO_GL="${MUJOCO_GL_BACKEND}" PYOPENGL_PLATFORM="${MUJOCO_GL_BACKEND}" DISPLAY= \
         /workspace/.venv/bin/python scripts/verify_reacher_push_quality.py \
         --dataset "$OUT_DATASET" \
         --min_push_episodes 200 \
         --min_improved_frac 0.7 \
         --min_contact_frac 0.7
fi

echo "Done: ${STABLEWM_HOME}/datasets/${OUTPUT_NAME}.lance" >&2
