#!/bin/bash
#SBATCH --job-name=lewm-download
#SBATCH --output=outputs/download-%j.out
#SBATCH --error=outputs/download-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION_CPU}}
#SBATCH --time=06:00:00

# Download official LeWM datasets from HuggingFace to scratch storage.
#
# Usage:
#   sbatch scripts/superpod/download_datasets.sh
#
# Edit the REPOS list below to download only what you need.

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=scripts/superpod/_common.sh
source "${SUBMIT_DIR}/scripts/superpod/_common.sh"

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
