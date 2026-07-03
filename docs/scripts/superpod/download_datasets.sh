#!/bin/bash
#SBATCH --job-name=lewm-download
#SBATCH --output=outputs/download-%j.out
#SBATCH --error=outputs/download-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition=cpu
#SBATCH --time=06:00:00

# Download official LeWM datasets from HuggingFace to scratch storage.
# After the job finishes, move extracted .h5 files from /scratch/<GROUP>/datasets
# to /project/<GROUP>/lewm-plus/.stable-wm/datasets/ for long-term storage.
#
# Edit the REPOS list below to download only what you need.

set -euo pipefail

REPOS=(
    quentinll/lewm-pusht
    quentinll/lewm-reacher
    quentinll/lewm-cube
    quentinll/lewm-tworooms
)

SCRATCH_DATA="/scratch/{{GROUP}}/datasets"
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

echo "All downloads finished. Extract .tar.zst archives with:"
echo "  cd $SCRATCH_DATA/<repo>/ && tar --zstd -xvf *.tar.zst"
