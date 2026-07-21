# LeWorldModel Architecture

## System Diagram
```text
                         Hydra configs (config/train, config/eval)
                                       |
                                       v
                              +------------------+
                              | train.py / eval.py|
                              +------------------+
                                  |          |
                    training path |          | evaluation/planning path
                                  v          v
                        +----------------+  +----------------------+
                        | Dataset loader |  | swm.World + policy   |
                        | + transforms   |  | (random or model)    |
                        +----------------+  +----------------------+
                                  |
                                  v
                          +---------------+
                          | JEPA model    |
                          | (jepa.py)     |
                          +---------------+
                                  |
          +-----------------------+-----------------------+
          |                                               |
          v                                               v
 +---------------------+                       +----------------------+
 | ViT encoder         |                       | ARPredictor          |
 | + projector         |                       | + action Embedder    |
 | -> context emb      |                       | -> pred emb sequence |
 +---------------------+                       +----------------------+
          |                                               |
          +-----------------------+-----------------------+
                                  v
                      +-----------------------------+
                      | Loss = pred MSE + SIGReg    |
                      | (module.py / train.py)      |
                      +-----------------------------+
                                  |
                                  v
                 Checkpoints / outputs under STABLEWM_HOME + outputs/

             SuperPOD wrapper layer (superpod/*.sh) orchestrates Slurm + container
```

## Purpose
LeWorldModel (LeWM) is a JEPA-style world model that learns from pixel trajectories and predicts future latent embeddings conditioned on actions. The repository focuses on:
- model definition (`jepa.py`, `module.py`)
- train/eval orchestration (`train.py`, `eval.py`)
- Hydra configuration (`config/`)
- custom Reacher environment and dataset generation (`scripts/`)
- SuperPOD execution wrappers (`superpod/`)

## High-Level Data Flow
1. Load a trajectory dataset (`stable_worldmodel.data.load_dataset`).
2. Apply transforms:
   - image preprocessing (ImageNet normalization + resize to 224)
   - z-score normalization for non-image columns (`action`)
3. Build LeWM model from Hydra config.
4. Encode context frames into latent embeddings.
5. Predict next latent embeddings autoregressively from context embeddings + action embeddings.
6. Optimize prediction loss + SIGReg regularizer.
7. Save checkpoints under `${STABLEWM_HOME}/checkpoints/...`.

Two-stage pipeline:
- **Pretrain**: random exploration data, SIGReg enabled, encoder trainable.
- **Finetune**: expert demonstration data, SIGReg disabled, encoder frozen.

## Training Step Tensor Shapes
```text
Input batch (from DataLoader)
  pixels: (B, T, C, H, W)
  action: (B, T, A)

Encode path (JEPA.encode)
  pixels -> flatten time -> (B*T, C, H, W)
  ViT CLS + projector -> (B*T, D)
  reshape -> emb: (B, T, D)
  action_encoder(action) -> act_emb: (B, T, D_a)

Predict path (train.py lejepa_forward)
  ctx_emb = emb[:, :history_size]                -> (B, Hs, D)
  ctx_act = act_emb[:, :history_size]            -> (B, Hs, D_a)
  pred_emb = predictor(ctx_emb, ctx_act)         -> (B, T-Hs, D) or aligned next-step slice
  tgt_emb = emb[:, num_preds:]                   -> target latent sequence

Losses
  pred_loss   = mean((pred_emb - tgt_emb)^2)
  sigreg_loss = SIGReg(emb.transpose(0, 1))      # expects (T, B, D)
  total_loss  = pred_loss + lambda * sigreg_loss
```

## Model Components

### `JEPA` (`jepa.py`)
- `encode(info)`:
  - flattens `(B,T,C,H,W)` pixels to `(B*T,...)`
  - runs ViT encoder (`stable_pretraining.backbone.utils.vit_hf`)
  - takes CLS token and projects to latent embedding
  - returns `emb` shaped `(B,T,D)` and optional `act_emb`
- `predict(emb, act_emb)`:
  - runs autoregressive predictor over sequence
  - outputs predicted latent sequence `(B,T,D)`
- Inference/planning helpers:
  - `rollout(...)` for candidate action rollout in latent space
  - `criterion(...)` and `get_cost(...)` for planning-time scoring

### Predictor Stack (`module.py`)
- `ARPredictor`: Transformer-based temporal predictor with learned positional embeddings.
- `ConditionalBlock`: AdaLN-zero conditioned Transformer block (conditioned on action embeddings).
- `Transformer`, `Attention`, `FeedForward`: core sequence modeling blocks.
- `Embedder`: action encoder (1D conv + MLP).
- `MLP`: projector heads for encoder output and predictor output.

### Regularization
- `SIGReg` in `module.py` implements a Sketch Isotropic Gaussian regularizer.
- Training objective in `train.py`:
  - `pred_loss = MSE(pred_emb, tgt_emb)`
  - `sigreg_loss = SIGReg(emb)`
  - `loss = pred_loss + lambda * sigreg_loss`

## Training Runtime (`train.py`)
- Hydra entrypoint: `config/train/lewm.yaml`.
- Data config selected via `data=...` (e.g. `data=reacher_reach_random`).
- `cfg.model.action_encoder.input_dim` is set dynamically from dataset action dim and frameskip.
- Uses `stable_pretraining.Module` + `stable_pretraining.Manager` around a Lightning trainer.
- `SaveCkptCallback` writes pretrained-style weights every epoch.
- Two-phase support: `experiment=pretrain` / `experiment=finetune` set `sigreg_enabled`/`freeze_encoder`.
- `ckpt_path` loads pretrained weights for finetuning phase.

## Evaluation Runtime (`eval.py`)
- Hydra entrypoint: `config/eval/<task>.yaml` (e.g. `reacher_reach.yaml`).
- Builds a `stable_worldmodel.World` environment and preprocessing pipelines.
- Custom env `swm/ReacherCustom-v0` (registered by `scripts/reacher_custom_env.py`) supports three task modes:
  - `qpos_match`: arm joints match target qpos
  - `reach`: fingertip reaches target body position
  - `push`: blue object reaches blue goal position
- Policy options:
  - `random`
  - model-based policy loaded from checkpoint (`swm.wm.utils.load_pretrained`)
- Uses `callables` mechanism to set env state from dataset: `set_state(qpos, qvel)` positions the simulation, `set_target_qpos(target_qpos)` extracts the task-relevant goal from a future qpos vector.
- Runs planning/evaluation episodes and writes metrics/results text output.

## Custom Environment (`scripts/reacher_custom_env.py`)
- `ReacherCustomEvalEnv` wraps `gymnasium.envs.mujoco.reacher_v5.ReacherEnv` with the custom XML `assets/reacher_task_agnostic.xml`.
- The XML has 10 DOF: 2 arm joints + 8 slider joints for objects and goals.
- Registered as `swm/ReacherCustom-v0`.
- Supports callables `set_state(qpos, qvel)` and `set_target_qpos(target_qpos)`.
- Task-specific success termination in `step()`.

## Dataset Generators (`scripts/`)

| Script | Format | Purpose |
|--------|--------|---------|
| `generate_reacher_custom_eval_dataset.py` | HDF5 | Eval datasets with `--task` (reach/push/qpos_match) |
| `generate_mujoco_dataset.py` | Lance | Training datasets (random or multitask expert) |

Both render at native 224×224 (width/height passed to env constructor).

### Dataset naming convention
- Eval: `<STABLEWM_HOME>/datasets/reacher_<task>_eval.h5`
- Train random: `<STABLEWM_HOME>/datasets/reacher_<task>_random.lance`
- Train expert: `<STABLEWM_HOME>/datasets/reacher_<task>_expert.lance`

## Configuration Layout
- `config/train/lewm.yaml`: global train defaults (trainer, loader, optimizer, loss).
- `config/train/model/lewm.yaml`: architecture wiring for JEPA/encoder/predictor.
- `config/train/data/*.yaml`: dataset profiles (`reacher_reach_random.yaml`, `reacher_push_expert.yaml`, etc.).
- `config/train/experiment/*.yaml`: phase profiles (`pretrain.yaml`, `finetune.yaml`).
- `config/eval/*.yaml`: task-specific evaluation config (`reacher_reach.yaml`, `reacher_push.yaml`, `reacher_custom.yaml`).
- `config/*/launcher/local.yaml`: local launcher defaults.

## Task-Specific Data Profiles (`config/train/data/`)

| Profile | Dataset | Use |
|---------|---------|-----|
| `reacher_reach_random` | `reacher_reach_random.lance` | Reach pretrain |
| `reacher_reach_expert` | `reacher_reach_expert.lance` | Reach finetune |
| `reacher_push_random` | `reacher_push_random.lance` | Push pretrain |
| `reacher_push_expert` | `reacher_push_expert.lance` | Push finetune |

Each profile specifies `num_steps`, `frameskip: 5`, `keys_to_load: [pixels, action]`, `keys_to_cache: [action]`.

## Storage and Paths
- Runtime root is `STABLEWM_HOME` (default `~/.stable-wm`).
- Dataset paths: `${STABLEWM_HOME}/datasets/<name>.{h5,lance}`.
- Checkpoint paths: `${STABLEWM_HOME}/checkpoints/<output_model_name>/`.
- Eval results: `outputs/eval-<jobid>.out` on cluster; text results in `${STABLEWM_HOME}/checkpoints/<policy_path>/dmc_results.txt`.

## SuperPOD Execution Layer
- `superpod/*.sh` are self-submitting helpers:
  - source `superpod/superpod.env`
  - generate temporary Slurm script
  - run job body with `srun` + Pyxis container
- Container contract:
  - code mounted at `${PROJECT_DIR}`
  - `${STABLEWM_HOME}` mounted to `/workspace/.stable-wm`
  - Python executable inside container: `/workspace/.venv/bin/python`
  - `STABLEWM_HOME` env var must be explicitly passed with `--container-env`.
- GPU partition (`normal`) works with containers; CPU partition (`cpu`) does not reliably support containers.

## Extension Points
- New task environment: add a new env class in `scripts/`, register it, add eval config.
- New task mode in existing env: add condition in `set_target_qpos()` and `step()` in `scripts/reacher_custom_env.py`, add eval config.
- New dataset: generate with existing scripts, add `config/train/data/<name>.yaml`.
- New architecture blocks: extend `module.py`, then wire into `config/train/model/lewm.yaml`.
- New cluster behavior: update `superpod/_common.sh` and script wrappers; keep all user-specific values in `superpod/superpod.env`.
