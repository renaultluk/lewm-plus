#!/bin/bash
# Hello-world job for SuperPOD.
# It prints node info, Slurm environment variables, Python version, and
# verifies that the STABLEWM_HOME directory is writable from the job.
#
# Usage:
#   bash superpod/hello_superpod.sh

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=superpod/_common.sh
source "${SUBMIT_DIR}/superpod/_common.sh"

# If not already inside a Slurm job, generate a batch script and submit it.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$PROJECT_DIR"
    mkdir -p outputs

    SBATCH_SCRIPT=$(mktemp)
    cat > "$SBATCH_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=hello-superpod
#SBATCH --output=outputs/hello-%j.out
#SBATCH --error=outputs/hello-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --account=${SUPERPOD_ACCOUNT}
#SBATCH --partition=${SUPERPOD_PARTITION_CPU}
#SBATCH --time=00:05:00
exec ${BASH_SOURCE[0]} "\$@"
EOF
    sbatch "$SBATCH_SCRIPT" "$@"
    rm -f "$SBATCH_SCRIPT"
    exit 0
fi

# Job body starts here.
echo "========================================"
echo "Hello from SuperPOD!"
echo "========================================"
echo ""
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "PWD: $(pwd)"
echo ""
echo "--- Slurm environment ---"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-<unset>}"
echo "SLURM_SUBMIT_HOST: ${SLURM_SUBMIT_HOST:-<unset>}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-<unset>}"
echo ""
echo "--- SuperPOD env loaded from superpod.env ---"
echo "SUPERPOD_ACCOUNT: ${SUPERPOD_ACCOUNT}"
echo "SUPERPOD_PARTITION_CPU: ${SUPERPOD_PARTITION_CPU}"
echo "SUPERPOD_GROUP: ${SUPERPOD_GROUP}"
echo "SUPERPOD_USER: ${SUPERPOD_USER}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "STABLEWM_HOME: ${STABLEWM_HOME}"
echo "SCRATCH_DATA: ${SCRATCH_DATA}"
echo ""

if command -v python3 &> /dev/null; then
    echo "--- Python check ---"
    python3 --version
else
    echo "python3 not found in job environment"
fi

echo ""
echo "--- Filesystem check ---"
mkdir -p "$STABLEWM_HOME"
TEST_FILE="$STABLEWM_HOME/hello_superpod_$SLURM_JOB_ID.txt"
echo "Job $SLURM_JOB_ID ran at $(date)" > "$TEST_FILE"
echo "Wrote test file: $TEST_FILE"
ls -lh "$TEST_FILE"

echo ""
echo "========================================"
echo "Hello-world job completed successfully."
echo "========================================"
