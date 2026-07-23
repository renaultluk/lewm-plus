# Two-Stage vs Single-Stage Training: Reacher Push — Test Report

## 1. Executive Summary

We compare two training strategies for the LeWorldModel (LeWM) JEPA-style world model on the
reacher **push-blue-object-to-blue-spot** task (ReacherCustom-v0, `task=push`):

| Strategy | Description | Task Success Rate | Total Train Time |
|---|---|---|---|
| **Random baseline** | Random policy + CEM planning | **6.0%** (3/50) | — |
| **Single-stage** | Train from scratch on expert data (100 epochs) | **10.0%** (5/50) | ~23 min |
| **Two-stage** | Pretrain on random data (50 epochs) → finetune on expert data (100 epochs) | **8.0%** (4/50) | ~3h11m |

**Key result**: The push task is significantly harder than reach — the best learned policy
achieves only 10% success vs 76% for reach. Counterintuitively, **single-stage outperforms
two-stage** (10% vs 8%), replicating the pattern from the earlier small-scale push experiment.
The frozen encoder in two-stage may prevent adaptation to the contact dynamics required for
pushing.

---

## 2. Environmental Setup

### 2.1 Hardware

| Component | Specification |
|---|---|
| **Node** | DGX node (various) |
| **GPU** | 4× NVIDIA H800 |
| **CPU** | 28 cores per node |
| **Driver** | NVIDIA 570.158.01 |
| **Interconnect** | Slurm-managed cluster (HKUST SuperPOD) |

### 2.2 Software

| Component | Version |
|---|---|
| **OS** | Linux 5.15.0-1082-nvidia (x86_64) |
| **Python** | 3.10.20 |
| **PyTorch** | (CUDA-enabled, bf16 support) |
| **Lightning** | stable-pretraining wrapper (Lightning v2.x based) |
| **Precision** | bf16-mixed (via `Trainer(precision=bf16)`) |
| **CUDA** | Available, Tensor Cores detected |

### 2.3 Cluster Configuration (HKUST SuperPOD)

- Slurm job scheduler
- Container runtime: Pyxis (srun --container-image)
- Container Python: `/workspace/.venv/bin/python`
- Shared storage: `STABLEWM_HOME=/workspace/.stable-wm` (mounted from host)

---

## 3. Experimental Setup

### 3.1 Model Architecture

LeWM JEPA with ViT-tiny backbone (18M params). Identical to the reach report architecture.

### 3.2 Hyperparameters (shared across all runs)

| Parameter | Value | Notes |
|---|---|---|
| `batch_size` | 128 | Per GPU; global 512 over 4 GPUs |
| `max_epochs` | 50 (pretrain) / 100 (finetune + single-stage) | |
| `optimizer` | AdamW, lr=5e-5, weight_decay=1e-3 | |
| `precision` | bf16-mixed | |
| `history_size` | 3 | context frames |
| `num_preds` | 1 | predict 1 step ahead |
| `frameskip` | 5 | action repeat |
| `img_size` | 224 | |
| `embed_dim` | 192 | |
| `seed` | 3072 | |
| `train_split` | 0.9 | 90/10 train/val |

### 3.3 Two-Phase Training Configuration

Identical to the reach experiment — see the reach report for details.

**Phase 1 (Pretrain)** — `experiment=pretrain`: SIGReg on, encoder unfrozen, scratch init.
**Phase 2 (Finetune)** — `experiment=finetune`: SIGReg off, encoder frozen, pretrained ckpt.
**Single-stage** — Default flags: SIGReg on, encoder unfrozen, scratch init.

### 3.4 Datasets

| Dataset | Task | Episodes | Steps/ep | Total frames |
|---|---|---|---|---|
| **reacher_push_random** | Random exploration (Phase 1 pretrain) | 500 | 1000 | 500,000 |
| **reacher_push_expert** | Push blue object to blue goal (Phase 2 + single-stage) | 200 | 100 | 20,000 |

Both datasets use the same MuJoCo scene with a blue pushable object and blue goal target.
The `task=push` success condition requires the blue object center to be within 0.025 of the
blue goal center.

### 3.5 Training Commands

**Phase 1 pretrain** (job 482444 — 4-GPU, dgx-29, 3h01m):

```bash
python train.py data=reacher_push_random experiment=pretrain \
  trainer.max_epochs=50 output_model_name=reacher_push_pretrain wandb.enabled=false
```

**Phase 2 finetune** (job 482607 — 4-GPU, 10m):

```bash
python train.py data=reacher_push_expert experiment=finetune \
  trainer.max_epochs=100 output_model_name=reacher_push_finetuned \
  ckpt_path=/workspace/.stable-wm/checkpoints/reacher_push_pretrain/weights_epoch_50.ckpt \
  wandb.enabled=false
```

**Single-stage** (job 482608 — 4-GPU, 23m):

```bash
python train.py data=reacher_push_expert \
  sigreg_enabled=true freeze_encoder=false \
  trainer.max_epochs=100 output_model_name=reacher_push_single_stage wandb.enabled=false
```

---

## 4. Results

### 4.1 Phase 1 — Pretrain on Random Data (job 482444)

50 epochs on `reacher_push_random` (500 episodes × 1000 steps, random actions).
Total wall time: **3h01m** on 4 GPUs. Final val pred_loss: **0.00410**.

### 4.2 Single-Stage — Train from Scratch on Expert Data (job 482608)

100 epochs on `reacher_push_expert` (200 episodes × 100 steps). Total wall time: **~23 min**.

### 4.3 Two-Stage Finetune — Frozen Encoder on Expert Data (job 482607)

100 epochs from push-pretrain checkpoint. Total wall time: **~10 min**.
Val pred_loss stabilizes around 0.044 (similar to the reach finetune behavior).

### 4.4 Task Performance Evaluation

Models evaluated on the **ReacherCustom-v0 push** task using CEM planning:

- **Eval config**: `config/eval/reacher_push.yaml`
- **Dataset**: `reacher_push_eval.h5` — 50 episodes
- **Evaluator**: CEM planner (horizon=5, action_block=5)
- **Goal**: Push blue object within 0.025 of blue goal within 25 steps
- **Metric**: `success_rate`

| Model | Success Rate | Δ vs Random | Eval Time |
|---|---|---|---|
| **Random baseline** | **6.0%** (3/50) | — | — |
| **Single-stage** (100 epochs) | **10.0%** (5/50) | **+4.0 pp** | ~3m |
| **Two-stage** (pretrain + finetune) | **8.0%** (4/50) | +2.0 pp | ~3m |

---

## 5. Analysis

### 5.1 Push Task Is Fundamentally Harder

All models perform poorly on push compared to reach:

| Metric | Reach | Push |
|---|---|---|
| Random baseline | 54.0% | 6.0% |
| Best learned policy | 76.0% | 10.0% |

The push task requires contact dynamics — the arm must make and maintain contact with the
blue object, then apply forces in the correct direction. This is inherently more challenging
than the reach task, which only requires the arm's fingertip to approach a target.

### 5.2 Single-Stage Beats Two-Stage for Push

Single-stage (10%) outperforms two-stage (8%), replicating the earlier small-scale push
experiment. This is the opposite of the reach finding. Two hypotheses:

1. **Frozen encoder limits contact-dynamics adaptation**: The pretrained encoder learned
   features from random arm movements that occasionally bump objects, but the frozen
   encoder cannot refine its representations for the precise contact dynamics needed for
   pushing.

2. **Expert data diversity**: At only 200 episodes, the expert data is limited. The
   single-stage model can adapt all parameters to this small dataset, while the frozen
   encoder in two-stage restricts the effective model capacity.

### 5.3 Comparison with Reach

| Metric | Push | Reach |
|---|---|---|
| **Random baseline** | **6.0%** | **54.0%** |
| **Single-stage** | **10.0%** | **70.0%** |
| **Two-stage** | **8.0%** | **76.0%** |
| **Winner** | Single-stage (+4 pp) | Two-stage (+6 pp) |

The optimal training strategy is **task-dependent**: two-stage helps for spatial tasks
(reach), while single-stage is better for contact-dynamics tasks (push).

---

## 6. Caveats & Future Work

- **Small expert dataset**: 200 episodes may be insufficient for the push task. Increasing
  expert data may narrow the gap between single-stage and two-stage.
- **CEM planner**: Default planner hyperparameters may not be optimal for push. Planner
  tuning could change rankings.
- **No replicates**: Single run per condition. Results are indicative.
- **Task difficulty**: The 10% ceiling suggests the model, planner, or both are
  ill-suited to the push task in the current configuration.

### Suggested next steps:
1. **Increase expert data** — generate more push expert episodes (e.g., 1000+)
2. **Planner tuning** — optimize CEM hyperparameters (horizon, samples, steps) for push
3. **Unfrozen finetune** — test two-stage without freezing the encoder for push
4. **Reach-to-push transfer** — test whether a reach-pretrained model finetunes better on push
   (the reverse of the cross-task experiment done for reach)
5. **Alternative planners** — test MPPI or gradient-based planners

---

## 7. Appendix: Constructing the Test Dataset

The eval dataset was generated on SuperPOD:

```bash
python scripts/generate_reacher_custom_eval_dataset.py \
  --task push --episodes 50 --max-steps 100 --image-size 224 \
  --output /workspace/.stable-wm/datasets/reacher_push_eval.h5
```

Training datasets generated via:

```bash
python scripts/generate_mujoco_dataset.py \
  --env ReacherTaskAgnostic-v0 --policy random \
  --episodes 500 --max-steps 1000 --image-size 224 \
  --output /workspace/.stable-wm/datasets/reacher_push_random.lance

python scripts/generate_mujoco_dataset.py \
  --env ReacherTaskAgnostic-v0 --policy reacher_multitask \
  --episodes 200 --max-steps 100 --image-size 224 \
  --output /workspace/.stable-wm/datasets/reacher_push_expert.lance
```
