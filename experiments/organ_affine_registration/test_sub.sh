#!/bin/bash
#SBATCH --job-name=test_sub
#SBATCH -t 00:05:00
#SBATCH -p kisski-h100
#SBATCH --constraint=inet
#SBATCH -G H100:1
#SBATCH --output=/mnt/vast-kisski/projects/ovgu_medicine_llm/ollama_data/temp/test_sub-%j.out
#SBATCH --error=/mnt/vast-kisski/projects/ovgu_medicine_llm/ollama_data/temp/test_sub-%j.err

echo "Test submission successful"
squeue -j $SLURM_JOB_ID
