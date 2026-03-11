#!/bin/bash
# --- SLURM Job Submission Directives ---
#SBATCH --job-name=julia_register
#SBATCH -t 48:00:00
#SBATCH -p kisski-h100
#SBATCH --constraint=inet
#SBATCH -G H100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
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

echo "Sequential Pre-initialization (Building PyCall and Instantiating)..."
# Run directly on the first node to handle build, precompilation, and CondaPkg setup safely
julia --project=. -e 'using Pkg; ENV["PYTHON"]=ENV["PYTHON"]; Pkg.build("PyCall"); Pkg.instantiate(); println("Base packages ready")'

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

echo "Verifying MPI configuration..."
julia --project=. -e 'using MPI; println("MPI loaded successfully"); println("MPI implementation: ", MPI.identify_implementation())'

echo "Starting Multi-GPU Training on Slurm..."
srun --mpi=pmix julia --project=. experiments/organ_affine_registration/train.jl
