# SuperPOD Guide for LeWorldModel Training

This guide covers running the LeWorldModel (LeWM) codebase on the HKUST
SuperPOD cluster (`superpod.ust.hk`). It is based on the official HPC docs
last updated 2026-06-04.

Official references:

- [Quick Start](https://hkust-hpc-docs.readthedocs.io/latest/quick-start/index.html)
- [First Job Template](https://hkust-hpc-docs.readthedocs.io/latest/quick-start/first-job-template.html)
- [Job Submission](https://hkust-hpc-docs.readthedocs.io/latest/quick-start/job-submission.html)
- [Software Environment](https://hkust-hpc-docs.readthedocs.io/latest/quick-start/software-environment.html)
- [Data and Storage](https://hkust-hpc-docs.readthedocs.io/latest/quick-start/data-and-storage.html)
- [Enroot/Pyxis Container Runtime](https://hkust-hpc-docs.readthedocs.io/latest/kb/enroot/index.html)

## 1. What is SuperPOD?

SuperPOD is a GPU cluster managed by **Slurm**. You do **not** run workloads
on the login node. Instead you write a batch script, submit it with `sbatch`,
and Slurm schedules it on a compute node.

Key differences from a laptop:

|                   | Laptop              | SuperPOD                                 |
| ----------------- | ------------------- | ---------------------------------------- |
| Login/work        | local shell         | `ssh` to login node, `sbatch` to compute |
| Software          | install anything    | containers + modules + `uv`              |
| GPUs              | 0–1                 | many GPU nodes via scheduler             |
| Storage           | local SSD           | `/home`, `/scratch/<group>`, `/project`  |

Important SuperPOD specifics:

- **Login host:** `superpod.ust.hk` (off campus: use VPN first).
- **GPU partition:** `normal` (allocates a whole GPU node).
- **CPU partition:** `cpu` (for preprocessing, testing, and small jobs).
- **Recommended environment:** container-based (`enroot`/`pyxis`). The shared
  Spack instance is at `/scratch/spack/2025` but containers are preferred.
- **Scratch:** `/scratch/<groupname>` is temporary, **auto-purged after 30 days
  of inactivity**, and shared by your group.
- **Home:** `/home/<username>` is persistent but limited in quota.
- **Project:** `/project/<groupname>` for long-term shared data.

## 2. First-Time Access Checklist

Run these once after you receive your credentials.

```bash
# 1. SSH into the cluster (replace <USER> with your HKUST username)
ssh <USER>@superpod.ust.hk

# 2. Confirm your Slurm associations and available partitions
sacctmgr show user $USER withassoc
```

Look for an `Account` (e.g. `cse`, `itsc`, ...) and partition (`normal`/`cpu`).
You will use those values in `#SBATCH --account=...` and `#SBATCH --partition=...`.

```bash
# 3. Confirm that the container runtime is available
srun --container-image nvcr.io#nvidia/cuda:12.8.0-base-ubuntu24.04 \
     --account=<your-account> --partition=cpu \
     --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:05:00 \
     nvidia-smi
```

If you see `nvidia-smi` output listing GPUs, your container path works.
(If the job lands on a CPU-only node, `nvidia-smi` will not be available.)

## 3. Project Layout on the Cluster

A sensible layout keeps code in `/project`, large datasets on `/scratch`, and a
container image in `/home`:

```text
/home/<USER>/
├── containers/
│   └── lewm.sqsh                 # pinned container image (large but reused)
└── .cache/
    └── huggingface/              # optional: shared HF cache

/project/<GROUP>/lewm-plus/
├── le-wm/                        # this repository
├── docs/scripts/superpod/        # helper scripts from this guide
├── .stable-wm/                   # stable-worldmodel home
│   ├── checkpoints/              # LeWM checkpoints
│   └── datasets/                 # extracted HDF5 / Lance datasets
│       └── pusht_expert_train.h5
└── outputs/                      # training logs + eval results
```

Use `/scratch/<GROUP>/` only for truly transient large files (e.g. raw
`tar.zst` downloads). Move extracted datasets to `/project` if they are reused
across experiments.

## 4. Build a Reproducible Container

The simplest robust workflow is to build a container on your local machine,
upload the resulting image, and run it on SuperPOD.

### Option A: Enroot `.sqsh` image (recommended)

We provide a Dockerfile in the repo:

```bash
# On your local machine (Linux with Docker + nvidia-container-toolkit)
docker build -f docs/scripts/superpod/Dockerfile -t lewm:superpod .
docker save lewm:superpod | enroot import -o lewm.sqsh -
```

Upload the image:

```bash
rsync -avP lewm.sqsh <USER>@superpod.ust.hk:~/containers/
```

### Option B: Python venv inside the job

If you prefer not to use containers, you can reproduce the local setup on a
compute node using the edge Spack Python module + `uv`:

```bash
module purge
source /scratch/spack/2025/share/spack/setup-env.sh || true
module load python/3.10
uv venv --python=3.10 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

This is fragile across node restarts and library changes, so the container
workflow is recommended for long experiments.

## 5. Helper Scripts

All scripts below live in `docs/scripts/superpod/` and should be copied to
`/project/<GROUP>/lewm-plus/` on SuperPOD.

### 5.1 Dashboard / quick commands

```bash
# Submit a LeWM training job
sbatch docs/scripts/superpod/train_lewm.sh --data=pusht_h5 --epochs=100

# Check queue
squeue --me

# Cancel a job
scancel <JOBID>

# Inspect finished job
sacct -j <JOBID> --format=JobID,JobName,Partition,State,ExitCode,Elapsed
```

### 5.2 Batch scripts

| Script                         | Purpose                                            |
| ------------------------------ | -------------------------------------------------- |
| `train_lewm.sh`                | Generic training job with container                |
| `evaluate_lewm.sh`             | Generic evaluation job with container              |
| `interactive_gpu.sh`           | Request a 1-hour GPU shell for debugging           |
| `download_datasets.sh`         | Use `hf` to download official datasets to scratch  |
| `sync_to_superpod.sh`          | One-command rsync from local machine               |

Each script is templated with `{{USER}}`, `{{GROUP}}`, and `{{ACCOUNT}}`
placeholders. Replace them by editing the script once, or use the
`configure_superpod.sh` helper described below.

## 6. Step-by-Step: Run an Official Training Job on SuperPOD

These steps assume you have already:

1. Cloned the repo locally and pushed it to the cluster (or copied it).
2. Built and uploaded the `lewm.sqsh` container.
3. Obtained your SuperPOD account and partition names.

### Step 1 — Configure the helper scripts

On SuperPOD, run the configure helper once:

```bash
cd /project/<GROUP>/lewm-plus
bash docs/scripts/superpod/configure_superpod.sh
```

It asks for your `ACCOUNT`, `GROUP`, and `USER`, then writes the values into
all batch scripts under `docs/scripts/superpod/`.

### Step 2 — Download/verify datasets

If the official PushT dataset is not yet in `.stable-wm/datasets/`:

```bash
sbatch docs/scripts/superpod/download_datasets.sh
```

This downloads `pusht_expert_train.h5.zst` to `/scratch/<GROUP>/datasets/` and
extracts it. Move the final `.h5` to `.stable-wm/datasets/` when done.

### Step 3 — Submit training

```bash
sbatch docs/scripts/superpod/train_lewm.sh data=pusht_h5 trainer.max_epochs=100 output_model_name=pusht_lewm_replicate
```

Training output goes to `outputs/train-<JOBID>.out` and checkpoints are saved
to `.stable-wm/checkpoints/<output_model_name>/`.

### Step 4 — Monitor

```bash
squeue --me
sacct -j <JOBID> --format=JobID,Partition,State,ExitCode,Elapsed,ReqTRES
```

### Step 5 — Evaluate the checkpoint

```bash
sbatch docs/scripts/superpod/evaluate_lewm.sh \
    --config-name=pusht.yaml \
    policy=pusht_lewm_replicate \
    eval.num_eval=50
```

## 7. Step-by-Step: Interactive Debugging on a GPU Node

```bash
bash docs/scripts/superpod/interactive_gpu.sh
# wait until you are on a compute node, then:
cd /project/<GROUP>/lewm-plus
python - <<'PY'
import torch
print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
PY
```

## 8. Tips and Notes

- **Do not run training on the login node.** Always use `sbatch` or `srun`.
- **Start small.** Request `--time=01:00:00` for debug runs and scale up after
  confirming the script works.
- **Whole-node allocation.** The `normal` GPU partition allocates a full node,
  so multi-GPU training can use all available GPUs if the code supports it.
- **Memory.** You generally do **not** need to set `--mem`; the cluster
  allocates memory proportionally to CPUs/GPUs.
- **Backups.** `/scratch` is purged after 30 days. Keep checkpoints and final
  results in `/project`.
- **VPN.** Off-campus access requires the HKUST VPN.
- **Acknowledgement.** If you publish work using SuperPOD, include:
  "The computations in this work were performed on the High Performance
  Computing facilities, HKUST SuperPOD, provided by ITSO, The Hong Kong
  University of Science and Technology (HKUST)."
