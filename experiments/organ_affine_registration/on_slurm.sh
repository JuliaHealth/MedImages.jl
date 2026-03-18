#!/bin/bash
# --- SLURM Job Submission Directives ---
#SBATCH --job-name=julia_register
#SBATCH -t 48:00:00
#SBATCH -p kisski-h100
#SBATCH --constraint=inet
#SBATCH -G H100:4
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=jakub.mitura14@gmail.com
#SBATCH --mail-type=all
#SBATCH --output=/user/joanna.wybranska/u10867/.project/dir.project/MedImages.jl/experiments/organ_affine_registration/logs/tb.out
#SBATCH --error=/user/joanna.wybranska/u10867/.project/dir.project/MedImages.jl/experiments/organ_affine_registration/logs/tb.err

set -e

# --- WandB Setup ---
export WANDB_API_KEY="b68288801637eb858426b1cd5ec62345a00879f3"

# --- Environment Setup ---
echo "--- Setting up the environment... ---"
module load miniforge3
module load gcc/13.2.0 openmpi/5.0.7

# Initialize Conda
echo "Verifying GPU presence..."
nvidia-smi
echo "Initializing Conda..."
source $(conda info --base)/etc/profile.d/conda.sh

# Define paths
REPO_ROOT="/user/joanna.wybranska/u10867/.project/dir.project/MedImages.jl"
EXP_DIR="$REPO_ROOT/experiments/organ_affine_registration"
ENV_YAML="$EXP_DIR/julia_env.yml"
CONDA_ENV_PATH="/user/joanna.wybranska/u10867/.conda/envs/julia_registration_env"

# Create Conda Environment if missing (skip update - env is stable)
if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "--- Creating Conda environment at $CONDA_ENV_PATH... ---"
    conda env create --prefix "$CONDA_ENV_PATH" --file "$ENV_YAML"
else
    echo "--- Conda environment found at $CONDA_ENV_PATH. ---"
fi

echo "Activating Conda environment..."
conda activate "$CONDA_ENV_PATH"

# --- Environment Stabilization (Fixes for Precompilation Races & Cache Errors) ---
# Prevent precompilation race conditions and MKL_jll hangs
export JULIA_PKG_PRECOMPILE_AUTO=0

# Clean stale package caches that cause ENOTEMPTY errors
echo "Cleaning stale package caches..."
find "$CONDA_ENV_PATH/share/julia/packages" -name ".github" -type d -exec rm -rf {} + 2>/dev/null || true

# --- Python Environment Stabilization ---
export PYTHON=$(which python)
export JULIA_PYTHONCALL_EXE="$PYTHON"
unset PYTHONHOME
unset PYTHONPATH
# Ensure Conda's lib stays in front to avoid system library leaks
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"

# --- Execution ---
cd "$REPO_ROOT"

# Set Julia threads
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Sequential Pre-initialization (Phase 1: Build & Instantiate)..."
# Force-remove any conflicting PyCall preferences before building
rm -f ~/.julia/environments/v1.10/LocalPreferences.toml
rm -f ~/.julia/prefs/PyCall*

# Run directly on the first node to handle build and instantiation safely
# We enforce PYTHONCOERCE to ensure it binds strongly to the conda env
PYTHONCOERCE=1 julia --project=. -e "
    using Pkg; 
    ENV[\"PYTHON\"]=\"$PYTHON\"; 
    Pkg.build(\"PyCall\"); 
    Pkg.instantiate(); 
    println(\"Phase 1: Instantiation complete\")
"

# Configure MPI.jl to use the system OpenMPI loaded via module
OMPI_LIB_DIR=$(dirname $(which mpiexec))/../lib
echo "Configuring MPIPreferences to use system OpenMPI at $OMPI_LIB_DIR..."
julia --project=. -e "
    using MPIPreferences
    MPIPreferences.use_system_binary(;
        library_names=[\"$OMPI_LIB_DIR/libmpi.so\"],
        mpiexec=\"$(which mpiexec)\")
    println(\"MPIPreferences configured for system OpenMPI\")
"

# --- Phase 2: Clean stale caches and re-precompile AFTER MPI config ---
# MPIPreferences.use_system_binary() invalidates all MPI-dependent compiled caches.
# We MUST re-precompile before srun, or 4 ranks will race to recompile simultaneously.
echo "Cleaning stale compiled caches after MPI reconfiguration..."
COMPILED_DIR="$CONDA_ENV_PATH/share/julia/compiled"
if [ -d "$COMPILED_DIR" ]; then
    echo "Removing stale compiled caches in $COMPILED_DIR..."
    rm -rf "$COMPILED_DIR"/v1.10/MPI*
    rm -rf "$COMPILED_DIR"/v1.10/HDF5*
    rm -rf "$COMPILED_DIR"/v1.10/JLD*
    rm -rf "$COMPILED_DIR"/v1.10/MedImages*
    rm -rf "$COMPILED_DIR"/v1.10/OpenMPI*
    rm -rf "$COMPILED_DIR"/v1.10/Wandb*
fi

echo "Sequential Pre-initialization (Phase 2: Full Precompile after MPI config)..."
julia --project=. -e '
    using Pkg;
    Pkg.precompile();
    println("Phase 2: Full precompilation complete")
'

echo "Verifying all packages load correctly (single process)..."
julia --project=. -e '
    using MPI
    println("MPI loaded: ", MPI.identify_implementation())
    using HDF5
    println("HDF5 loaded successfully")
    using MedImages
    println("MedImages loaded successfully")
    println("All packages verified!")
'

echo "Starting Multi-GPU Training on Slurm..."
srun --mpi=pmix julia --project=. experiments/organ_affine_registration/train.jl
