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

### Data expectations
- For `data=pusht_h5`, canonical dataset path is `${STABLEWM_HOME}/datasets/pusht_expert_train.h5`.
- Current `superpod/train_lewm.sh` also auto-detects nested `${STABLEWM_HOME}/datasets/pusht/pusht_expert_train.h5` and injects a Hydra override.

## Common Commands

### Local training/eval
```bash
python train.py data=pusht_h5 trainer.max_epochs=1 output_model_name=dev_smoke wandb.enabled=false
python eval.py --config-name=pusht.yaml policy=random eval.num_eval=2
```

### SuperPOD workflow
```bash
bash superpod/configure_superpod.sh
bash superpod/sync_to_superpod.sh
bash superpod/hello_superpod.sh
bash superpod/hello_superpod_gpu.sh
bash superpod/resume_train.sh pusht_h5_replicate_run 70 100
bash superpod/train_lewm.sh data=pusht_h5 trainer.max_epochs=100 output_model_name=pusht_h5_replicate wandb.enabled=false
bash superpod/evaluate_lewm.sh --config-name=pusht.yaml policy=pusht_h5_replicate/pusht_h5_replicate eval.num_eval=50
```

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
  bash superpod/evaluate_lewm.sh --config-name=pusht.yaml policy=pusht_h5_replicate/pusht_h5_replicate eval.num_eval=50
  ```
- Dataset resolution for `data=pusht_h5`:
  - Canonical: `${STABLEWM_HOME}/datasets/pusht_expert_train.h5`
  - Also supported by `train_lewm.sh`: nested `${STABLEWM_HOME}/datasets/pusht/pusht_expert_train.h5` via auto override.
