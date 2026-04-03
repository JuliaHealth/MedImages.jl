# MedImages.jl Execution Guide

This guide describes how to run the main library features, experiments, and benchmarks after the repository reorganization.

## 1. Main Library (MedImages.jl)

The core functionality of `MedImages.jl` is implemented in `src/`.

```julia
using Pkg
Pkg.activate(".")
using MedImages
```

## 2. Experiments

All experiments are located in the `experiments/` directory.

### SciML Dose Refinement (Dosimetry)
Located in `experiments/sciml_dose_refinement/`.

#### Main UDE Models
- **UDE Training (No Approximations):**
  ```bash
  julia --project=. experiments/sciml_dose_refinement/train_ude_no_approx.jl
  ```
- **UDE Training (Standard):**
  ```bash
  julia --project=. experiments/sciml_dose_refinement/train_ude.jl
  ```
- **Competitor Training (CNN+Approx, etc.):**
  ```bash
  julia --project=. experiments/sciml_dose_refinement/train_competitors_64.jl
  ```

#### Evaluation & Inference
- **Inference and Evaluation:**
  ```bash
  julia --project=. experiments/sciml_dose_refinement/run_inference_and_eval.jl
  ```
- **Comprehensive Evaluation:**
  ```bash
  julia --project=. experiments/sciml_dose_refinement/eval_all_models.jl
  ```

#### Baselines
Located in `experiments/sciml_dose_refinement/baselines/`.
- **PBPK ML Pipeline:**
  ```bash
  python3 experiments/sciml_dose_refinement/baselines/PBPK_TDT/pbpk_ml_pipeline.py
  ```
- **DblurDoseNet:**
  ```bash
  python3 experiments/sciml_dose_refinement/baselines/DblurDoseNet/train.py
  ```
- **SemiDose:**
  ```bash
  python3 experiments/sciml_dose_refinement/baselines/SemiDose/src/train_SimRegMatch.py
  ```

### SUV Consistency Experiment
Located in `experiments/suv_consistency/`.
To run the SUV metrics plotting:
```bash
julia --project=. experiments/suv_consistency/plot_suv_metrics.jl
```

## 3. Benchmarks

Benchmarks are located in `experiments/benchmark/`.
Results are saved to `data/benchmark_results/`.

To run the main benchmark suite:
```bash
julia --project=. experiments/benchmark/run_benchmarks.jl
```

## 4. Model Checkpoints

Model checkpoints are organized in `data/checkpoints/` with subfolders for each model type:
- `data/checkpoints/PBPK_ML/`: Baseline PBPK models
- `data/checkpoints/CNN_APPROX/`: CNN models with approximations
- `data/checkpoints/UDE_NO_APPROX/`: UDE models without approximations

## 5. Testing

Tests are organized as follows:
- Unit and integration tests: `test/`
- Experiment-specific tests: `test/experiments/`

To run all tests:
```bash
julia --project=. test/runtests.jl
```
