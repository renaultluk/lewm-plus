#!/bin/bash
set -e
export STABLEWM_HOME=/workspace/.stable-wm
export PYTHONPATH=/project/mscbdt2024/lewm-plus
PYTHON=/workspace/.venv/bin/python
WORKDIR=/project/mscbdt2024/lewm-plus

echo "STABLEWM_HOME=$STABLEWM_HOME"
CACHE_DIR=$($PYTHON -c "import stable_worldmodel as swm; print(swm.data.utils.get_cache_dir())")
echo "Cache dir: $CACHE_DIR"
DATASETS_DIR=$CACHE_DIR/datasets
mkdir -p "$DATASETS_DIR"

##########################################
# Eval datasets (HDF5, 50 episodes, 100 steps)
##########################################
echo "=== EVAL: reach ==="
$PYTHON $WORKDIR/scripts/generate_reacher_custom_eval_dataset.py \
  --task reach --episodes 50 --max-steps 100
echo "DONE"

echo "=== EVAL: push ==="
$PYTHON $WORKDIR/scripts/generate_reacher_custom_eval_dataset.py \
  --task push --episodes 50 --max-steps 100
echo "DONE"

##########################################
# Training datasets (Lance)
##########################################

# Reach random (pretrain)
echo "=== TRAIN: reach random (500 ep, 1000 steps) ==="
$PYTHON $WORKDIR/scripts/generate_mujoco_dataset.py \
  --env ReacherTaskAgnostic-v0 --episodes 500 --max_steps 1000 --policy random \
  --output "$DATASETS_DIR/reacher_reach_random.lance"
echo "DONE"

# Reach expert (finetune)
echo "=== TRAIN: reach expert (200 ep, 100 steps) ==="
$PYTHON $WORKDIR/scripts/generate_mujoco_dataset.py \
  --env ReacherTaskAgnostic-v0 --episodes 200 --max_steps 100 \
  --policy reacher_multitask --reacher_tasks reach_red_spot \
  --output "$DATASETS_DIR/reacher_reach_expert.lance"
echo "DONE"

# Push random (pretrain)
echo "=== TRAIN: push random (500 ep, 1000 steps) ==="
$PYTHON $WORKDIR/scripts/generate_mujoco_dataset.py \
  --env ReacherTaskAgnostic-v0 --episodes 500 --max_steps 1000 --policy random \
  --output "$DATASETS_DIR/reacher_push_random.lance"
echo "DONE"

# Push expert (finetune)
echo "=== TRAIN: push expert (200 ep, 100 steps) ==="
$PYTHON $WORKDIR/scripts/generate_mujoco_dataset.py \
  --env ReacherTaskAgnostic-v0 --episodes 200 --max_steps 100 \
  --policy reacher_multitask --reacher_tasks push_blue_object_to_blue_spot \
  --output "$DATASETS_DIR/reacher_push_expert.lance"
echo "DONE"

echo "=== ALL DATASETS GENERATED ==="
