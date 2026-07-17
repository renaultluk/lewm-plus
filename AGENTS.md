# AGENTS Guide

## Project Overview
- This repo contains LeWorldModel (LeWM): a JEPA-style world model trained end-to-end from pixels.
- Core entry points are `train.py` (training) and `eval.py` (planning/evaluation).
- Hydra drives configuration from `config/train/` and `config/eval/`.
- The project relies on `stable-worldmodel` (data/env/planning) and `stable-pretraining` (training loop/module wrappers).

## Key Layout
- `train.py`, `eval.py`: main scripts.
- `jepa.py`, `module.py`, `utils.py`: model/loss/support code.
- `config/train/lewm.yaml`: base train config; data profiles in `config/train/data/`.
- `config/eval/*.yaml`: eval task configs (`pusht.yaml`, `cube.yaml`, etc.).
- `superpod/`: HKUST SuperPOD helpers (self-submitting Slurm scripts + container tooling).
- `docs/SUPERPOD_GUIDE.md`: SuperPOD runbook.

## Dev Environment Tips

### Local (recommended)
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

### Runtime data location
- `STABLEWM_HOME` controls where datasets/checkpoints are resolved.
- Default is `~/.stable-wm` unless overridden.
- In SuperPOD jobs, scripts mount host `${STABLEWM_HOME}` into container at `/workspace/.stable-wm`.
- `superpod/train_lewm.sh` also sets `SPT_CACHE_DIR=${STABLEWM_HOME}` so Lightning/stable-pretraining checkpoints stay under the same root.

### Data expectations
- For `data=pusht_h5`, canonical dataset path is `${STABLEWM_HOME}/datasets/pusht_expert_train.h5`.
- Current `superpod/train_lewm.sh` also auto-detects nested `${STABLEWM_HOME}/datasets/pusht/pusht_expert_train.h5` and injects a Hydra override.

## Common Commands

### Local training/eval
```bash
python train.py data=pusht_h5 trainer.max_epochs=1 output_model_name=dev_smoke wandb.enabled=false
python eval.py --config-name=pusht.yaml policy=random eval.num_eval=2
```

### Reacher task-agnostic dataset + export
```bash
# Generate a 224x224 smoke dataset from the custom XML reacher scene
.venv/bin/python scripts/generate_mujoco_dataset.py \
  --env ReacherTaskAgnostic-v0 \
  --policy reacher_multitask \
  --episodes 3 \
  --max_steps 60 \
  --image_size 224 \
  --output .stable-wm/datasets/reacher_task_agnostic_xml_smoke_224.lance

# Export episode 0 to MP4 for quick visual verification
.venv/bin/python scripts/view_lance_episode.py \
  --dataset .stable-wm/datasets/reacher_task_agnostic_xml_smoke_224.lance \
  --episode 0 \
  --output .stable-wm/datasets/reacher_task_agnostic_xml_smoke_224_ep0.mp4

# SuperPOD full-scale generation (comparable training scale)
bash superpod/generate_reacher_dataset.sh \
  EPISODES=10000 \
  MAX_STEPS=100 \
  IMAGE_SIZE=224 \
  OUTPUT_NAME=reacher_task_agnostic_train_224

# If render init fails on a node, force headless backend explicitly
bash superpod/generate_reacher_dataset.sh MUJOCO_GL_BACKEND=osmesa
```

### SuperPOD workflow
```bash
bash superpod/configure_superpod.sh
bash superpod/sync_to_superpod.sh
bash superpod/hello_superpod.sh
bash superpod/hello_superpod_gpu.sh
bash superpod/migrate_checkpoints.sh pusht_h5_replicate_run pusht_h5_replicate
bash superpod/convert_ckpt_to_eval_pt.sh --src-ckpt /project/<GROUP>/lewm-plus/.stable-wm/checkpoints/pusht_h5_replicate_run/pusht_h5_replicate_weights.ckpt --run-name pusht_h5_replicate_eval
bash superpod/resume_train.sh pusht_h5_replicate_run 70 100
bash superpod/train_lewm.sh data=pusht_h5 trainer.max_epochs=100 output_model_name=pusht_h5_replicate wandb.enabled=false
bash superpod/evaluate_lewm.sh --config-name=pusht.yaml policy=pusht_h5_replicate_run/weights_epoch_70.pt eval.num_eval=50 cache_dir=/workspace/.stable-wm eval.dataset_name=/workspace/.stable-wm/datasets/pusht/pusht_expert_train
```

## Two-Phase Training (Pretrain → Finetune)

The training script supports two-phase training via config overrides.

### Phase 1 — Pretrain with SIGReg on task-agnostic data
```bash
python train.py data=reacher_agnostic \
  loss.sigreg.weight=0.09 \
  sigreg_enabled=true \
  freeze_encoder=false \
  trainer.max_epochs=50 \
  output_model_name=reacher_pretrain
```

### Phase 2 — Finetune on task data (SIGReg off, encoder frozen)
```bash
python train.py data=pusht_h5 \
  sigreg_enabled=false \
  freeze_encoder=true \
  trainer.max_epochs=100 \
  output_model_name=pusht_finetuned \
  ckpt_path=/absolute/path/to/reacher_pretrain_weights.ckpt
```

Key flags:
- `sigreg_enabled=false` — disables SIGReg loss computation (only prediction loss remains).
- `freeze_encoder=true` — sets `requires_grad=False` on all encoder parameters.
- `ckpt_path=/path/to/model.ckpt` — loads pretrained weights via `spt.Manager(weights_only=True)`.

## Testing and Verification
- There is currently no formal `pytest` suite in this repo.
- Minimum checks before pushing:
  - Shell syntax for helper scripts:
    ```bash
    bash -n superpod/*.sh
    ```
  - Python import smoke test:
    ```bash
    python -c "import stable_worldmodel, stable_pretraining"
    ```
  - If touching configs/scripts, run a short train smoke (`max_epochs=1`) or a random-policy eval (`eval.num_eval=2`).
- For SuperPOD changes, always validate both:
  - CPU hello job: `bash superpod/hello_superpod.sh`
  - GPU/container hello job: `bash superpod/hello_superpod_gpu.sh`

## Contribution Notes
- Keep `superpod/superpod.env` out of git (contains user/group specifics).
- Large artifacts are intentionally ignored (`.stable-wm/`, `*.h5`, `*.ckpt`, `outputs/`, etc.).
- Prefer Hydra overrides for experiment changes instead of hard-coding values.
- Keep SuperPOD scripts self-submitting and source `superpod/_common.sh` for shared env handling.

## Troubleshooting Quick Hits
- `Cannot resolve 'pusht_expert_train'`: check dataset path under `${STABLEWM_HOME}/datasets/` and file extension.
- `sbatch/srun` module warning on SuperPOD: ensure scripts load `slurm` module before submission/runtime.
- Pyxis/container issues: verify `CONTAINER_PATH` in `superpod/superpod.env` and run `hello_superpod_gpu.sh` first.

## Current SuperPOD Status
- Repo: `renaultluk/lewm-plus` with SuperPOD helpers under repo-root `superpod/`.
- Container expectation: `superpod/Dockerfile` sets `STABLEWM_HOME=/workspace/.stable-wm` and uses `/workspace/.venv/bin/python`.
- Mount strategy: host `${STABLEWM_HOME}` is mounted to container `/workspace/.stable-wm` (no `--container-env` requirement).
- Walltime control: set `TRAIN_TIME` in `superpod/superpod.env` (e.g. `71:00:00` under a 72h cap).
- Training resource controls: set `TRAIN_NODES`, `TRAIN_NTASKS`, and `TRAIN_GPUS_PER_NODE` in `superpod/superpod.env`.
- Checkpoint layout: both `.pt` and `.ckpt` files for a run now live under `${STABLEWM_HOME}/checkpoints/<subdir>/`.
- For old runs, use `bash superpod/migrate_checkpoints.sh <run_id> <output_model_name>` to consolidate mixed legacy paths into `${STABLEWM_HOME}/checkpoints/<subdir>/`.
- Preferred training command:
  ```bash
  bash superpod/train_lewm.sh data=pusht_h5 trainer.max_epochs=100 output_model_name=pusht_h5_replicate wandb.enabled=false
  ```
- Preferred resumed training pattern for long runs:
  ```bash
  bash superpod/resume_train.sh pusht_h5_replicate_run 70 100
  ```
- Preferred eval command:
  ```bash
  bash superpod/evaluate_lewm.sh --config-name=pusht.yaml policy=pusht_h5_replicate_run/weights_epoch_70.pt eval.num_eval=50 cache_dir=/workspace/.stable-wm eval.dataset_name=/workspace/.stable-wm/datasets/pusht/pusht_expert_train
  ```
- Dataset resolution for `data=pusht_h5`:
  - Canonical: `${STABLEWM_HOME}/datasets/pusht_expert_train.h5`
  - Also supported by `train_lewm.sh`: nested `${STABLEWM_HOME}/datasets/pusht/pusht_expert_train.h5` via auto override.
