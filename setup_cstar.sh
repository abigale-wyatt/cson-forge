#!/bin/bash
# Script to clone C-Star, switch to orchestration branch, and install in editable mode

set -e  # Exit on error

# Initialize conda if needed
if ! command -v conda &> /dev/null; then
    echo "Error: conda command not found. Please initialize conda first."
    exit 1
fi

REPO_URL="https://github.com/CWorthy-ocean/C-Star.git"
BRANCH="orchestration"
CODE_ROOT=$(python -c "import cson_forge.config as config; paths = config.get_data_paths(); print(paths.code_root)")
REPO_DIR="$CODE_ROOT/C-Star"
CONDA_ENV="cson-forge"

pushd "$CODE_ROOT" > /dev/null

# Clone or update the repository
if [ -d "$REPO_DIR" ]; then
    echo "Repository already exists. Updating..."
    cd "$REPO_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
    cd ..
else
    echo "Cloning C-Star repository..."
    git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi

# Activate conda environment and install
echo "Activating conda environment: $CONDA_ENV"
echo "Installing C-Star in editable mode..."

# Activate conda environment and run pip install
# Note: In CI, conda may be a wrapper around micromamba
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

cd "$REPO_DIR"
pip install -e .

echo "C-Star installation completed successfully!"

popd > /dev/null