#!/bin/bash
#SBATCH --job-name=benchmark-scaling
#SBATCH --partition=shared
#SBATCH --output=benchmark-scaling-%j.out
#SBATCH --error=benchmark-scaling-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --account=ees250129

# Exit on error
set -euo pipefail

# Get the directory where this script is located
# Use SLURM_SUBMIT_DIR if available (set by sbatch), otherwise use script location
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"

# Verify we're in the right directory and files exist
if [[ ! -f "benchmark_scaling.py" ]]; then
    echo "Error: benchmark_scaling.py not found in $SCRIPT_DIR" >&2
    echo "Current directory: $(pwd)" >&2
    echo "Files in directory:" >&2
    ls -la >&2
    exit 1
fi

# Activate conda environment if available
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cson-forge-v0 2>/dev/null || true
fi

clobber_inputs_flag=
#clobber_inputs_flag="--clobber-inputs"

# Loop over ensemble IDs
for ensemble_id in 2 3 4; do
    echo "=========================================="
    echo "Running benchmark scaling for ensemble_id=${ensemble_id}"
    echo "Current directory: $(pwd)"
    echo "=========================================="
    
    python "$SCRIPT_DIR/benchmark_scaling.py" \
        --ensemble-id "${ensemble_id}" \
        --domains-file "$SCRIPT_DIR/domains-bm-scaling.yml" \
        ${clobber_inputs_flag}
    
    echo ""
    echo "Completed ensemble_id=${ensemble_id}"
    echo ""
done

echo "=========================================="
echo "All ensemble runs completed"
echo "=========================================="
