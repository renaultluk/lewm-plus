---
name: superpod
description: |
  Workflows for the HKUST SuperPOD GPU cluster: SSH access, Slurm job management,
  container-based LeWM training/evaluation, quota tracking, and sbatch development.
license: MIT
metadata:
  cluster: superpod.ust.hk
  group: mscbdt2024
  container: lewm.sqsh
---

# SuperPOD Skill (HKUST GPU Cluster)

## 1. SSH & Connection

```bash
ssh wlluk@superpod.ust.hk          # off-campus: VPN first
# Login node: DO NOT run training here. Use sbatch/srun.
```

## 2. Slurm Basics

`squeue --me`                        # list your jobs + states (PD=PENDING, R=RUNNING, CG=COMPLETING)
`squota`                             # check remaining GPU/CPU hours
`sacct -j <JOBID> --format=JobID,JobName,Partition,State,ExitCode,Elapsed`
`scancel <JOBID>`                    # kill a job

**Watch out:** `sbatch` is a symlink to `/usr/bin/tutorial-slurm` (blocks with `less` pager).
Workaround — run `bash -lic "sbatch ..."` or pre-load module:
```bash
module load slurm 2>/dev/null
sbatch <args>
```
The scripts `train_lewm.sh` and `evaluate_lewm.sh` handle this automatically.

**Key cluster facts:**
- GPU partition: `normal` (allocates whole node, supports Pyxis containers)
- CPU partition: `cpu` (no reliable container support — use GPU even for dataset gen)
- Account: `mscbdt2024`
- User: `wlluk`
- Project dir: `/project/mscbdt2024/lewm-plus/`
- Container: `~/containers/lewm.sqsh`
- `STABLEWM_HOME` host: `/project/mscbdt2024/lewm-plus/.stable-wm` → container `/workspace/.stable-wm`
- GPU quota: 160h/month across partitions (check with `squota`)

## 3. Container & Mounts

The container (`lewm.sqsh`) sets STABLEWM_HOME in its Dockerfile, but `srun --container-image` does not reliably propagate Dockerfile ENV. Always pass explicitly:

```bash
srun --container-image "$CONTAINER_PATH" \
     --container-mounts "/project:/project,/home:/home,${STABLEWM_HOME}:/workspace/.stable-wm" \
     --container-writable \
     /usr/bin/env STABLEWM_HOME=/workspace/.stable-wm \
     /workspace/.venv/bin/python train.py ...
```

## 4. Training Jobs

Submit via the convenience scripts. They self-submit to Slurm and read `superpod/superpod.env`:

```bash
# Pretrain (50 epochs, SIGReg enabled)
bash train_lewm.sh data=reacher_reach_random experiment=pretrain \
  trainer.max_epochs=50 output_model_name=reacher_reach_pretrain wandb.enabled=false

# Finetune (100 epochs, SIGReg off, encoder frozen)
bash train_lewm.sh data=reacher_reach_expert experiment=finetune \
  trainer.max_epochs=100 output_model_name=reacher_reach_finetuned \
  ckpt_path=/workspace/.stable-wm/checkpoints/reacher_reach_pretrain/reacher_reach_pretrain_weights.ckpt \
  wandb.enabled=false
```

### Resource Tuning (edit `superpod/superpod.env`)
| Variable | Optimal (4-GPU) | Meaning |
|---|---|---|
| TRAIN_GPUS_PER_NODE | 4 | GPUs per node |
| TRAIN_NTASKS | 4 | total tasks (= GPUs) |
| CPUS_PER_TASK | 7 | CPU cores per GPU |
| TRAIN_TIME | 3:00:00 | walltime |

**Efficiency (reacher_reach_random, bs=512):**
- 1-GPU: 1152 samples/GPU-s (100% efficiency) — 6.4h for 50ep
- 4-GPU: 696 samples/GPU-s (60% efficiency) — 2.5h for 50ep
- 8-GPU: 630 samples/GPU-s (55% efficiency) — 1.7h for 50ep
- 4-GPU is the quota-efficiency sweet spot (2.4× wall-clock speedup at 60% per-GPU cost)

### Outputs
- Logs: `outputs/train-<JOBID>.out` / `.err`
- Checkpoints: `$STABLEWM_HOME/checkpoints/<output_model_name>/`
- Both `.ckpt` (Lightning) and `.pt` (state-dict) files saved

### Direct sbatch (when not using the convenience script)

```bash
module load slurm 2>/dev/null
sbatch <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=lewm-train
#SBATCH --output=outputs/train-%j.out
#SBATCH --nodes=1 --ntasks=4 --ntasks-per-node=4
#SBATCH --cpus-per-task=7 --gpus-per-node=4
#SBATCH --account=mscbdt2024 --partition=normal --time=3:00:00
module load slurm 2>/dev/null
MOUNTS="/project:/project,/home:/home,${HOME}/.stable-wm:/workspace/.stable-wm"
srun --ntasks=4 --container-image ~/containers/lewm.sqsh \
  --container-mounts "$MOUNTS" --container-writable \
  --container-workdir /project/mscbdt2024/lewm-plus \
  /usr/bin/env STABLEWM_HOME=/workspace/.stable-wm \
  /workspace/.venv/bin/python train.py data=reacher_reach_random \
  trainer.max_epochs=50 output_model_name=custom_run wandb.enabled=false
SBATCH
```

## 5. Evaluation Jobs

```bash
bash evaluate_lewm.sh --config-name=reacher_reach.yaml \
  policy=reacher_reach_finetuned/weights_epoch_100.pt eval.num_eval=50 \
  cache_dir=/workspace/.stable-wm
```

Key args:
- `policy=<run_name>/weights_epoch_N.pt` — model to evaluate (looks in `$STABLEWM_HOME/checkpoints/<run_name>/`)
- `eval.num_eval=50` — number of episodes
- `--config-name=reacher_reach.yaml` — Hydra config from `config/eval/`
- `policy=random` — random policy baseline

Results: printed to stdout and captured in `outputs/eval-<JOBID>.out`.

## 6. sbatch Development Cycle

When developing new sbatch scripts:

1. **Write** the script in the repo root on SuperPOD (e.g. `my_job.sbatch`).
2. **Validate shell syntax:** `bash -n my_job.sbatch`
3. **Test with short walltime:** set `--time=00:10:00`.
4. **Submit:** `bash -lic "sbatch my_job.sbatch"` (bypasses the sbatch symlink issue).
5. **Check immediately:** `squeue --me` to see if it goes to R or stays PD.
6. **If PD with reason `AssocGrpGRESMinutes`:** you've exceeded GPU quota. Reduce time/GPUs or wait for reset.
7. **When it runs:** tail the output: `tail -f outputs/my_job-<JOBID>.out`.
8. **If it fails:** check the err file: `cat outputs/my_job-<JOBID>.err`.

## 7. Dataset Generation

All 6 datasets already exist on SuperPOD. To regenerate:

```bash
# On SuperPOD, via sbatch (use GPU partition — cpu doesn't support containers):
sbatch --time=12:00:00 --ntasks=1 --gpus-per-node=1 \
  --account=mscbdt2024 --partition=normal --job-name=gen-all \
  --wrap="srun --container-image=$HOME/containers/lewm.sqsh \
    --container-mounts=/project:/project,/home:/home,\${STABLEWM_HOME}:/workspace/.stable-wm \
    --container-writable \
    /usr/bin/env STABLEWM_HOME=/workspace/.stable-wm \
    bash /project/mscbdt2024/lewm-plus/scripts/generate_all_datasets.sh"
```

## 8. Datasets & Checkpoints

**`$STABLEWM_HOME/datasets/`** (6 files):
- `reacher_reach_eval.h5` (719 MB) — 50 episodes, 100 steps
- `reacher_push_eval.h5` (719 MB) — 50 episodes, 100 steps
- `reacher_reach_random.lance/` — 500 ep × 1000 steps (pretrain)
- `reacher_reach_expert.lance/` — 200 ep × 100 steps (finetune)
- `reacher_push_random.lance/` — 500 ep × 1000 steps (pretrain)
- `reacher_push_expert.lance/` — 200 ep × 100 steps (finetune)

**`$STABLEWM_HOME/checkpoints/`** per-run subdirectories with `.ckpt` + `.pt` files.

## 9. Troubleshooting Common Issues

| Symptom | Cause & Fix |
|---|---|
| `sbatch` opens a pager (`less`) | `sbatch` is a symlink to tutorial-slurm. Run `module load slurm` first, or use `bash -lic "sbatch ..."`, or use the convenience scripts. |
| `ModuleNotFoundError: No module named 'scripts'` | Repo root not on sys.path. The scripts in `scripts/` add it automatically; if running manually, use `PYTHONPATH=/project/mscbdt2024/lewm-plus`. |
| Job stays PD, reason `AssocGrpGRESMinutes` | GPU quota exhausted (160h/month). Check `squota`. Reduce walltime, use fewer GPUs, or wait for reset. |
| Container can't find dataset | `STABLEWM_HOME` not set correctly inside container. Use `/usr/bin/env STABLEWM_HOME=/workspace/.stable-wm` (Dockerfile ENV is unreliable). |
| CPU partition fails with container | `cpu` partition doesn't support Pyxis reliably. Use `--partition=normal` even for non-GPU work. |
| EGL / headless rendering fails | EGL context must be created at module top before any MuJoCo import. All dataset generators handle this. |
| Large output file | Check `outputs/train-<JOBID>.out` / `eval-<JOBID>.out`. |

## 10. Reference Files

| File | Purpose |
|---|---|
| `superpod/superpod.env` | Cluster config (account, paths, resource defaults) |
| `superpod/_common.sh` | Shared helpers sourced by all scripts |
| `superpod/train_lewm.sh` | Self-submitting training launcher |
| `superpod/evaluate_lewm.sh` | Self-submitting evaluation launcher |
| `superpod/interactive_gpu.sh` | Request interactive GPU shell |
| `superpod/convert_ckpt_to_eval_pt.sh` | Convert .ckpt → .pt for eval |
| `superpod/migrate_checkpoints.sh` | Consolidate legacy checkpoint paths |
| `config/train/data/reacher_{reach,push}_{random,expert}.yaml` | Data profiles |
| `config/eval/reacher_{reach,push}.yaml` | Eval configs |
| `scripts/reacher_custom_env.py` | Custom env with task modes |
