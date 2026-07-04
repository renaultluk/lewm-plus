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
- SuperPOD execution wrappers (`superpod/`)

## High-Level Data Flow
1. Load a trajectory dataset (`stable_worldmodel.data.load_dataset`).
2. Apply transforms:
   - image preprocessing (ImageNet normalization + resize)
   - z-score normalization for non-image columns (`action`, `proprio`, `state`)
3. Build LeWM model from Hydra config.
4. Encode context frames into latent embeddings.
5. Predict next latent embeddings autoregressively from context embeddings + action embeddings.
6. Optimize prediction loss + SIGReg regularizer.
7. Save checkpoints under `${STABLEWM_HOME}/checkpoints/...`.

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
- Data config selected via `data=...` (for PushT HDF5: `data=pusht_h5`).
- `cfg.model.action_encoder.input_dim` is set dynamically from dataset action dim and frameskip.
- Uses `stable_pretraining.Module` + `stable_pretraining.Manager` around a Lightning trainer.
- `SaveCkptCallback` writes pretrained-style weights every epoch.

## Evaluation Runtime (`eval.py`)
- Hydra entrypoint: `config/eval/<task>.yaml` (e.g. `pusht.yaml`).
- Builds a `stable_worldmodel.World` environment and preprocessing pipelines.
- Policy options:
  - `random`
  - model-based policy loaded from checkpoint (`swm.wm.utils.load_pretrained`)
- Runs planning/evaluation episodes and writes metrics/results text output.

## Configuration Layout
- `config/train/lewm.yaml`: global train defaults (trainer, loader, optimizer, loss).
- `config/train/model/lewm.yaml`: architecture wiring for JEPA/encoder/predictor.
- `config/train/data/*.yaml`: dataset profiles (`pusht.yaml`, `pusht_h5.yaml`, etc.).
- `config/eval/*.yaml`: task-specific evaluation config.
- `config/*/launcher/local.yaml`: local launcher defaults.

## Storage and Paths
- Runtime root is `STABLEWM_HOME` (default `~/.stable-wm`).
- Expected train dataset name resolution for `pusht_h5` is `pusht_expert_train`.
- Canonical path: `${STABLEWM_HOME}/datasets/pusht_expert_train.h5`.
- SuperPOD scripts also support nested `${STABLEWM_HOME}/datasets/pusht/pusht_expert_train.h5` for convenience.

## SuperPOD Execution Layer
- `superpod/*.sh` are self-submitting helpers:
  - source `superpod/superpod.env`
  - generate temporary Slurm script
  - run job body with `srun` + Pyxis container
- Container contract:
  - code mounted at `${PROJECT_DIR}`
  - `${STABLEWM_HOME}` mounted to `/workspace/.stable-wm`
  - Python executable inside container: `/workspace/.venv/bin/python`

## Checkpoint and Artifact Outputs
- Train logs: `outputs/train-<jobid>.out|err`.
- Eval logs: `outputs/eval-<jobid>.out|err`.
- Model checkpoints: `${STABLEWM_HOME}/checkpoints/<run>/...`.
- Eval results text/video paths are controlled by eval config and selected policy path.

## Extension Points
- New dataset: add `config/train/data/<name>.yaml` and matching eval config.
- New architecture blocks: extend `module.py`, then wire into `config/train/model/lewm.yaml`.
- New task eval: add `config/eval/<task>.yaml`.
- New cluster behavior: update `superpod/_common.sh` and script wrappers; keep all user-specific values in `superpod/superpod.env`.
