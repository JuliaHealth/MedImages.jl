# SciML Dose Refinement Experiments

This directory contains the source code for the dosimetry refinement experiments using Universal Differential Equations (UDEs) and various baseline models.

## Models and Execution Paths

### 1. UDE Models (Main Contributions)
- **UDE No-Approx (64x64x64):** Trained on larger patches for better spatial consistency.
  - Script: `train_competitors_64.jl` (Mode: `UDE_NO_APPROX`)
  - Checkpoint: `data/checkpoints/UDE_NO_APPROX_64/model_best_UDE_NO_APPROX_64.jls`
- **UDE No-Approx (Original):**
  - Script: `train_ude_no_approx.jl`
  - Checkpoint: `data/checkpoints/UDE_NO_APPROX/model_best.jls`
- **Standard UDE:**
  - Script: `train_ude.jl`
  - Checkpoint: `data/checkpoints/UDE_NO_APPROX/model_ude_no_approx.jls` (Legacy)

### 2. Deep Learning Competitors
- **Stabilized CNN (ResNet-32):**
  - Script: `train_competitors_64.jl` (Mode: `CNN_APPROX`)
  - Checkpoint: `data/checkpoints/CNN_APPROX_64/model_best_CNN_APPROX_64.jls`
- **Pure CNN:**
  - Script: `train_pure_cnn.jl`
  - Checkpoint: `data/checkpoints/pure_cnn/model_pure_cnn.jls`

### 3. External Baselines (Python)
Located in the `baselines/` subdirectory:
- **PBPK-ML (Extra Trees):** `baselines/PBPK_TDT/pbpk_ml_pipeline.py`
- **Spect0 (3D Residual):** `baselines/spect0/stageIII/train.py`
- **DblurDoseNet:** `baselines/DblurDoseNet/train.py`
- **SemiDose:** `baselines/SemiDose/src/train_SimRegMatch.py`

## How to Run

### Dataset Preparation
If the dosimetry dataset is missing, it can be regenerated from the source `/DATA/` directory:
```bash
julia --project=. experiments/sciml_dose_refinement/prepare_dataset.jl
```
This script processes raw SPECT/CT and Monte Carlo dosemaps into the format expected by the training and evaluation scripts, saving them to `data/dosimetry_data/`.

### Julia Scripts
To run training or evaluation (CPU fallback supported):
```bash
julia --project=. experiments/sciml_dose_refinement/eval_all_models_64.jl
```

### Evaluation
To evaluate all models and generate metrics:
- `eval_all_models.jl`: Evaluates models trained on original patches.
- `eval_all_models_64.jl`: Evaluates models trained on 64x64x64 patches.

## Directory Structure
- `baselines/`: Source code for external comparison models.
- `dosimetry/`: Utility scripts for dose processing.
- `logs/`: SLURM and local execution logs.
- `slurm/`: Job submission scripts for high-performance clusters.
