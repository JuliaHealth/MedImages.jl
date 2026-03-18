# Lu-177 Dose Refinement - Slurm Training Guide

This directory contains Slurm batch scripts for training the three different SciML models on the Lu-177 dataset.

## Scripts

1.  **`pinn_slurm.sh`**: Trains the PINN-style 3D CNN model.
2.  **`fno_slurm.sh`**: Trains the Fourier Neural Operator (FNO) model.
3.  **`ude_slurm.sh`**: Trains the Universal Differential Equation (UDE) model.

## How to Run

To submit a job to the Slurm queue, use the `sbatch` command from the cluster login node:

```bash
# Submit PINN training
sbatch experiment/sciml_dose_refinement/slurm/pinn_slurm.sh

# Submit FNO training
sbatch experiment/sciml_dose_refinement/slurm/fno_slurm.sh

# Submit UDE training
sbatch experiment/sciml_dose_refinement/slurm/ude_slurm.sh
```

## Configuration

You can adjust training hyperparameters by setting environment variables in the Slurm scripts:

- `LU_EPOCHS`: Number of training epochs (default: 50).
- `LU_NUM_SAMPLES`: Number of patient cases to load (default: 100).
- `LU_BATCH_SIZE`: Batch size for GPU training (default: 4).

The logs (standard output and error) will be stored in the `experiment/sciml_dose_refinement/logs/` directory.

## Environment

The scripts assume a pre-configured Conda environment at `/user/joanna.wybranska/u10867/.conda/envs/julia_registration_env`. 
The scripts also use the system-provided `miniforge3` and `gcc/13.2.0` modules.
Unlike the registration experiments, these scripts are optimized for **single-GPU** execution and do not require MPI.
