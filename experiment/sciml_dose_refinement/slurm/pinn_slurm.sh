#!/bin/bash
#SBATCH --job-name=lu_pinn
#SBATCH -t 24:00:00
#SBATCH -p kisski-h100
#SBATCH --constraint=inet
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/pinn_%j.out
#SBATCH --error=logs/pinn_%j.err

set -e

# --- Environment Setup ---
module load miniforge3
module load gcc/13.2.0

# Initialize Conda
source $(conda info --base)/etc/profile.d/conda.sh

# Define paths
REPO_ROOT="/user/joanna.wybranska/u10867/.project/dir.project/MedImages.jl"
EXP_DIR="$REPO_ROOT/experiment/sciml_dose_refinement"
CONDA_ENV_PATH="/user/joanna.wybranska/u10867/.conda/envs/julia_registration_env" # Re-using existing env if compatible

conda activate "$CONDA_ENV_PATH"

# Data Path on Slurm
export LU_DATA_DIR="/user/joanna.wybranska/u10867/.project/dir.project/ollama_data/dataset_Lu/home/jm/project_ssd/MedImages.jl/test_data/dataset_Lu"
export LU_EPOCHS=100
export LU_NUM_SAMPLES=100
export LU_BATCH_SIZE=1
export LU_TARGET_SIZE=256
export LU_LR=1e-3

# --- Execution ---
cd "$EXP_DIR"
mkdir -p logs

echo "Starting PINN Training on Slurm..."
julia --project=. run_experiment_lu.jl pinn
