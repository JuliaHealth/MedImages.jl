#!/bin/bash
#SBATCH --account=yuni0
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --job-name=DblurDoseNet-Y90-test-1
#SBATCH --mail-user=jiayx@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=64g
module load python3.8-anaconda/2021.05

export PYTHONPATH=$PWD:$PYTHONPATH
python3.8 test.py