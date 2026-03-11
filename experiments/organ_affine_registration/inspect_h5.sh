#!/bin/bash
#SBATCH --job-name=inspect_h5
#SBATCH -t 00:05:00
#SBATCH -p kisski
#SBATCH --output=/user/joanna.wybranska/u10867/.project/dir.project/MedImages.jl/experiments/organ_affine_registration/logs/inspect_h5.out
#SBATCH --error=/user/joanna.wybranska/u10867/.project/dir.project/MedImages.jl/experiments/organ_affine_registration/logs/inspect_h5.err

module load miniforge3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /user/joanna.wybranska/u10867/.conda/envs/julia_registration_env

cd /user/joanna.wybranska/u10867/.project/dir.project/MedImages.jl
julia --project=. experiments/organ_affine_registration/inspect_h5.jl
