#!/bin/bash
#SBATCH --job-name=lu_fno
#SBATCH -t 24:00:00
#SBATCH -p kisski-h100
#SBATCH --constraint=inet
#SBATCH -G H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/user/joanna.wybranska/u10867/.project/dir.project/MedImages.jl/experiment/sciml_dose_refinement/logs/fno/fno_%j.out
#SBATCH --error=/user/joanna.wybranska/u10867/.project/dir.project/MedImages.jl/experiment/sciml_dose_refinement/logs/fno/fno_%j.err

set -e

# --- Environment Setup ---
module load miniforge3
module load gcc/13.2.0

# Initialize Conda
source $(conda info --base)/etc/profile.d/conda.sh

# Define paths
REPO_ROOT="/user/joanna.wybranska/u10867/.project/dir.project/MedImages.jl"
EXP_DIR="$REPO_ROOT/experiment/sciml_dose_refinement"
CONDA_ENV_PATH="/user/joanna.wybranska/u10867/.conda/envs/julia_registration_env"

conda activate "$CONDA_ENV_PATH"

# Prevent precompilation race conditions and MKL_jll hangs
export JULIA_PKG_PRECOMPILE_AUTO=0

# Clean stale package caches that cause ENOTEMPTY errors
echo "Cleaning stale package caches..."
find "$CONDA_ENV_PATH/share/julia/packages" -name ".github" -type d -exec rm -rf {} + 2>/dev/null || true

# Pre-initialize: develop MedImages and instantiate before main script
echo "Pre-initializing Julia environment..."
cd "$EXP_DIR"
julia --project=. -e '
    using Pkg
    Pkg.develop(path=joinpath(@__DIR__, "..", ".."))
    Pkg.instantiate()
    println("Environment ready")
'

# Explicit precompilation (single process, no race conditions)
echo "Precompiling packages..."
julia --project=. -e 'using Pkg; Pkg.precompile()'

# Data Path on Slurm
export LU_DATA_DIR="/user/joanna.wybranska/u10867/.project/dir.project/ollama_data/dataset_Lu/home/jm/project_ssd/MedImages.jl/test_data/dataset_Lu"
export LU_EPOCHS=100
export LU_NUM_SAMPLES=100
export LU_BATCH_SIZE=1
export LU_TARGET_SIZE=256
export LU_LR=1e-3

# --- Execution ---
mkdir -p logs

echo "Starting FNO Training on Slurm..."
julia --project=. run_experiment_lu.jl fno
