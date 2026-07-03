#!/bin/bash
# Configure placeholders in all SuperPOD helper scripts.
#
# Usage:
#   bash docs/scripts/superpod/configure_superpod.sh
#
# The script interactively asks for ACCOUNT, GROUP, and USER, then replaces
# the placeholders in the batch scripts under docs/scripts/superpod/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== SuperPOD helper script configuration ==="
echo "Find your account/partition with: sacctmgr show user \$USER withassoc"
read -rp "SLURM account (e.g. cse): " ACCOUNT
read -rp "Project group name (e.g. cse): " GROUP
read -rp "Your HKUST username (no @ust.hk): " USER

for f in "$SCRIPT_DIR"/*.sh; do
    # Skip this script itself
    [[ "$f" == "$SCRIPT_DIR/configure_superpod.sh" ]] && continue
    sed -i \
        -e "s/{{ACCOUNT}}/${ACCOUNT}/g" \
        -e "s/{{GROUP}}/${GROUP}/g" \
        -e "s/{{USER}}/${USER}/g" \
        "$f"
    echo "Configured: $f"
done

echo "Done. Review the scripts before submitting jobs."
