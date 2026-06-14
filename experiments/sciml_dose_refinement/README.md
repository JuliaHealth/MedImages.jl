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

## Recent Physics Refinements

Based on clinical physics review, the core `train_ude_no_approx.jl` was upgraded to strictly adhere to absolute mathematical modeling conventions for $^{177}$Lu-PSMA TRT:
1. **Absolute Image Quantification**: Implemented exact scaling via parameterized Camera Calibration Factors (`CF`) and DICOM `Rescale Slope`.
2. **Partial Volume Tracking**: Integrated explicit hooks for spatial Recovery Coefficients (`RC`). It is noted structurally that this primarily governs accurate tracking for per-ROI evaluation rather than per-voxel.
3. **Parametrized Hardware Calibration**: The Hounsfield Unit conversion mapping (`hu_to_density`) was parameterized to support tailorable `slope_air` and `slope_tissue` coefficients mapped individually for specific clinical CT hardware configurations.
4. **Spatial Density Gradients ($\nabla \rho$)**: The `build_no_approx_model` neural architecture was expanded from two to **three parallel input branches**. The instantaneous physical density gradient ($\nabla \rho$) is analytically computed via 3D finite differences in the data-pipe and fed continuously to the neural residual $\mathcal{N}_\theta$. This enables mathematically constrained learning of heterogeneous Compton scattering at organ boundaries.
