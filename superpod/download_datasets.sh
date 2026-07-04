#!/bin/bash
# Download official LeWM datasets from HuggingFace to scratch storage.
#
# Usage:
#   bash superpod/download_datasets.sh
#
# Edit the REPOS list below to download only what you need.
# This script reads superpod/superpod.env and self-submits to Slurm.

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=superpod/_common.sh
source "${SUBMIT_DIR}/superpod/_common.sh"

# If not already inside a Slurm job, generate a batch script and submit it.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$PROJECT_DIR"
    mkdir -p outputs

    # SuperPOD login nodes require the slurm module to be loaded before
    # sbatch/srun can be used.
    module load slurm 2>/dev/null || true

    SBATCH_SCRIPT=$(mktemp)
    cat > "$SBATCH_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=lewm-download
#SBATCH --output=outputs/download-%j.out
#SBATCH --error=outputs/download-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK_CPU}
#SBATCH --account=${SUPERPOD_ACCOUNT}
#SBATCH --partition=${SUPERPOD_PARTITION_CPU}
#SBATCH --time=06:00:00
exec ${BASH_SOURCE[0]} "\$@"
EOF
    sbatch "$SBATCH_SCRIPT" "$@"
    rm -f "$SBATCH_SCRIPT"
    exit 0
fi

# Job body starts here.
# After the job finishes, move extracted .h5 files from $SCRATCH_DATA
# to $STABLEWM_HOME/datasets/ for long-term storage.

REPOS=(
    quentinll/lewm-pusht
)

mkdir -p "$SCRATCH_DATA"
cd "$SCRATCH_DATA"

source /scratch/spack/2025/share/spack/setup-env.sh || true
module load python/3.10 || true

if ! command -v hf &> /dev/null; then
    echo "Installing huggingface-cli (hf)..."
    pip install --user -U huggingface_hub hf-xet
fi

for repo in "${REPOS[@]}"; do
    echo "Downloading $repo ..."
    hf download "$repo" --repo-type dataset --local-dir "$SCRATCH_DATA/${repo##*/}"
done

echo "All downloads finished. Extract tar archives and move .h5 files with:"
echo "  cd $SCRATCH_DATA/<repo>/ && tar -xvf *.tar.zst"
echo "  mv $SCRATCH_DATA/<repo>/*.h5 $STABLEWM_HOME/datasets/"
