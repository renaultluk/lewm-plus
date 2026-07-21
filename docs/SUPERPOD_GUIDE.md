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
  - **Important:** containers (Pyxis) do not work reliably on the CPU partition.
    Use the GPU partition even for dataset generation if the CPU partition fails.
- **Recommended environment:** container-based (`enroot`/`pyxis`). The shared
  Spack instance is at `/scratch/spack/2025` but containers are preferred.
- **Scratch:** `/scratch/<groupname>` is temporary, **auto-purged after 30 days
  of inactivity**, and shared by your group.
- **Home:** `/home/<username>` is persistent but limited in quota.
- **Project:** `/project/<groupname>` for long-term shared data.

### GPU Quota
Check remaining hours with `squota`. The `mscbdt2024` account has a total of
160 GPU-hours per user across all partitions. When exhausted, jobs get the
reason `AssocGrpGRESMinutes` and stay PENDING until quota resets. Reduce
requested wall time or use fewer GPUs to fit within remaining quota.

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
├── superpod/             # helper scripts from this guide
├── .stable-wm/                   # stable-worldmodel home
│   ├── checkpoints/              # LeWM checkpoints
│   └── datasets/                 # extracted HDF5 / Lance datasets
│       ├── reacher_reach_eval.h5
│       ├── reacher_push_eval.h5
│       ├── reacher_reach_random.lance/
│       ├── reacher_reach_expert.lance/
│       ├── reacher_push_random.lance/
│       ├── reacher_push_expert.lance/
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
# Make sure your Docker daemon socket is reachable from WSL/Linux,
# e.g. /var/run/docker.sock exists.
docker build -f superpod/Dockerfile -t lewm:superpod .
enroot import -o lewm.sqsh dockerd://lewm:superpod
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

All scripts below live in `superpod/` and should be copied to
`/project/<GROUP>/lewm-plus/` on SuperPOD.

### 5.1 Configuration file

All user- and group-specific values are centralized in a single file:

```text
superpod/superpod.env
```

A template is provided as `superpod.env.example`. To generate your own config:

```bash
cd /project/<GROUP>/lewm-plus
bash superpod/configure_superpod.sh
```

This interactively asks for your Slurm account, partitions, group, and username,
then writes `superpod.env`. `superpod.env` is ignored by Git so your credentials
stay local.

Variables you can tune in `superpod.env`:

| Variable                  | Meaning                                         | Example         |
| ------------------------- | ----------------------------------------------- | --------------- |
| `SUPERPOD_ACCOUNT`        | Slurm account                                   | `cse`           |
| `SUPERPOD_PARTITION_GPU`  | GPU partition                                   | `normal`        |
| `SUPERPOD_PARTITION_CPU`  | CPU partition                                   | `cpu`           |
| `SUPERPOD_GROUP`          | Project group (used in `/project`, `/scratch`)  | `cse`           |
| `SUPERPOD_USER`           | Your HKUST username                             | `renaultluk`    |
| `PROJECT_DIR`             | Repository root on SuperPOD                     | `/project/...`  |
| `CONTAINER_PATH`          | Path to the Enroot `.sqsh` container            | `~/containers/...` |
| `TRAIN_NODES`             | Number of nodes for training jobs               | `1`             |
| `TRAIN_NTASKS`            | Number of tasks for training jobs               | `1`             |
| `TRAIN_GPUS_PER_NODE`     | GPUs requested per training node                | `1`             |
| `CPUS_PER_TASK`           | CPU cores for GPU jobs                          | `28`            |
| `CPUS_PER_TASK_CPU`       | CPU cores for CPU jobs                          | `8`             |
| `TRAIN_TIME`              | Walltime for `train_lewm.sh` jobs              | `71:00:00`      |
| `STABLEWM_HOME`           | Dataset/checkpoint root                         | `$PROJECT_DIR/.stable-wm` |
| `SCRATCH_DATA`            | Scratch directory for raw downloads             | `/scratch/.../datasets`   |

After changing `superpod.env`, no reconfiguration step is needed. Job scripts
read it at submit/runtime and generate the correct Slurm directives automatically.

### 5.2 Dashboard / quick commands

```bash
# Check remaining GPU/CPU quota
squota

# Submit a LeWM training job
bash superpod/train_lewm.sh data=reacher_reach_random experiment=pretrain \
  trainer.max_epochs=50 output_model_name=reacher_reach_pretrain wandb.enabled=false

# Submit an evaluation job
bash superpod/evaluate_lewm.sh --config-name=reacher_reach.yaml \
  policy=reacher_reach_finetuned/weights_epoch_100.pt eval.num_eval=50 \
  cache_dir=/workspace/.stable-wm

# Check queue
squeue --me

# Cancel a job
scancel <JOBID>

# Inspect finished job
sacct -j <JOBID> --format=JobID,JobName,Partition,State,ExitCode,Elapsed
```

### 5.3 Batch scripts

| Script                         | Purpose                                            |
| ------------------------------ | -------------------------------------------------- |
| `train_lewm.sh`                | Generic training job with container                |
| `evaluate_lewm.sh`             | Generic evaluation job with container              |
| `migrate_checkpoints.sh`       | Consolidate old checkpoint layouts into new layout |
| `convert_ckpt_to_eval_pt.sh`   | Convert training `.ckpt` to eval-ready `.pt`      |
| `interactive_gpu.sh`           | Request a GPU shell for debugging                  |
| `download_datasets.sh`         | Use `hf` to download official datasets to scratch  |
| `sync_to_superpod.sh`          | One-command rsync from local machine               |
| `hello_superpod.sh`            | Hello-world job to verify Slurm + env setup        |
| `hello_superpod_gpu.sh`        | Hello-world job to verify Slurm + GPU + container  |

## 6. Quick Sanity Check: Submit a Hello-World Job

Before running a real training/eval job, submit the minimal CPU job to confirm
that Slurm accepts your config and the compute node can read `superpod.env`.

### Step 1 — Make sure `superpod.env` exists

```bash
cd /project/<GROUP>/lewm-plus
ls superpod/superpod.env
```

If missing, run:

```bash
bash superpod/configure_superpod.sh
```

### Step 2 — Submit the hello-world job

```bash
bash superpod/hello_superpod.sh
```

Expected output looks like:

```text
Submitted batch job 12345
```

### Step 3 — Monitor and inspect

```bash
squeue --me
# wait until the job reaches COMPLETED

sacct -j 12345 --format=JobID,JobName,Partition,State,ExitCode,Elapsed
cat outputs/hello-12345.out
```

A successful run prints the hostname, Slurm environment variables, the values
loaded from `superpod.env`, Python version, and a test file written under
`$STABLEWM_HOME`.

### Step 4 — If something fails

- **File not found / bad interpreter**: run the script with `bash superpod/hello_superpod.sh` (recommended).
- **Invalid account or partition**: run `sacctmgr show user $USER withassoc`
  and update `superpod.env`, then rerun `configure_superpod.sh`.
- **`superpod.env` not found**: confirm it was generated by `configure_superpod.sh`
  and is in the same directory as the helper scripts.

Once `hello_superpod.sh` completes successfully, test the GPU path too:

```bash
bash superpod/hello_superpod_gpu.sh
```

This runs a GPU node inside the LeWM container and prints PyTorch + CUDA info.
It requires `~/containers/lewm.sqsh` to exist on SuperPOD. If it fails with a
missing container error, build and upload the image first (see section 4).

## 7. Step-by-Step: Run Training on SuperPOD

These steps assume you have already:

1. Cloned the repo locally and pushed it to the cluster (or copied it).
2. Built and uploaded the `lewm.sqsh` container.
3. Obtained your SuperPOD account and partition names.

### Step 1 — Configure the helper scripts

On SuperPOD, run the configure helper once:

```bash
cd /project/<GROUP>/lewm-plus
bash superpod/configure_superpod.sh
```

It asks for your account, partitions, group, and username, then writes them
to `superpod/superpod.env`. After that, job scripts read this file
automatically and generate the correct Slurm directives at submit time. You
only need to rerun `configure_superpod.sh` if your account/partition values
change.

### Step 2 — Generate datasets

Use the convenience script to generate all datasets at once:

```bash
sbatch --time=12:00:00 --ntasks=1 --gpus-per-node=1 \
  --account=$SUPERPOD_ACCOUNT --partition=$SUPERPOD_PARTITION_GPU \
  --job-name=gen-all \
  --wrap="srun --container-image=$CONTAINER_PATH \
    --container-mounts=/project:/project,/home:/home,${STABLEWM_HOME}:/workspace/.stable-wm \
    --container-writable --container-env=STABLEWM_HOME=/workspace/.stable-wm \
    bash /project/mscbdt2024/lewm-plus/scripts/generate_all_datasets.sh"
```

This generates:
- 2 eval HDF5 datasets (`reacher_reach_eval.h5`, `reacher_push_eval.h5`)
- 4 training Lance datasets (`reacher_reach_random.lance`, `reacher_reach_expert.lance`, `reacher_push_random.lance`, `reacher_push_expert.lance`)

### Step 3 — Submit two-stage training

**Reach:**

```bash
# Phase 1 — Pretrain
bash superpod/train_lewm.sh data=reacher_reach_random experiment=pretrain \
  trainer.max_epochs=50 output_model_name=reacher_reach_pretrain wandb.enabled=false

# Phase 2 — Finetune (after Phase 1 completes)
bash superpod/train_lewm.sh data=reacher_reach_expert experiment=finetune \
  trainer.max_epochs=100 output_model_name=reacher_reach_finetuned \
  ckpt_path=/workspace/.stable-wm/checkpoints/reacher_reach_pretrain/reacher_reach_pretrain_weights.ckpt \
  wandb.enabled=false
```

**Push:**

```bash
# Phase 1 — Pretrain
bash superpod/train_lewm.sh data=reacher_push_random experiment=pretrain \
  trainer.max_epochs=50 output_model_name=reacher_push_pretrain wandb.enabled=false

# Phase 2 — Finetune
bash superpod/train_lewm.sh data=reacher_push_expert experiment=finetune \
  trainer.max_epochs=100 output_model_name=reacher_push_finetuned \
  ckpt_path=/workspace/.stable-wm/checkpoints/reacher_push_pretrain/reacher_push_pretrain_weights.ckpt \
  wandb.enabled=false
```

Training output goes to `outputs/train-<JOBID>.out` and checkpoints are saved
to `.stable-wm/checkpoints/<subdir>/`.

### Step 4 — Monitor

```bash
squeue --me
squota                                  # check remaining hours
sacct -j <JOBID> --format=JobID,Partition,State,ExitCode,Elapsed,ReqTRES
```

### Step 5 — Evaluate

```bash
bash superpod/evaluate_lewm.sh --config-name=reacher_reach.yaml \
  policy=reacher_reach_finetuned/weights_epoch_100.pt eval.num_eval=50 \
  cache_dir=/workspace/.stable-wm

bash superpod/evaluate_lewm.sh --config-name=reacher_push.yaml \
  policy=reacher_push_finetuned/weights_epoch_100.pt eval.num_eval=50 \
  cache_dir=/workspace/.stable-wm
```

## 7. Step-by-Step: Interactive Debugging on a GPU Node

```bash
bash superpod/interactive_gpu.sh
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
- **Container env.** The Dockerfile sets `STABLEWM_HOME=/workspace/.stable-wm`,
  but `srun --container-image` does not reliably propagate Dockerfile `ENV`.
  Always pass `--container-env STABLEWM_HOME=/workspace/.stable-wm`.
- **CPU partition limitations.** The `cpu` partition does not support Pyxis
  containers reliably. Use the GPU partition for all containerized workloads.
- **GPU quota.** Check `squota` before submitting long jobs. The group-level
  GPU-minute limit may cause jobs to stay PENDING with reason
  `AssocGrpGRESMinutes`. Short jobs under 1 hour are more likely to schedule.
- **Acknowledgement.** If you publish work using SuperPOD, include:
  "The computations in this work were performed on the High Performance
  Computing facilities, HKUST SuperPOD, provided by ITSO, The Hong Kong
  University of Science and Technology (HKUST)."
