# Reproducing MedImages.jl Dosimetry Benchmarks

This guide provides step-by-step instructions to replicate the comparative dosimetry benchmarks described in the manuscript. The benchmarks cover three areas: Scientific Fidelity, Computational Performance (Cross-Language), and Traditional Deep Learning Baselines.

## 1. Prerequisites

### Julia Environment
*   **Version**: Julia 1.10+
*   **Setup**:
    ```bash
    julia --project=. -e 'using Pkg; Pkg.instantiate()'
    ```

### Python Environment
*   **Conda/Mamba Environment**: `merlin_anatomical`
*   **Dependencies**: `torch`, `torchdiffeq`, `jax`, `diffrax`, `equinox`, `scikit-learn`, `nibabel`
*   **Installation**:
    ```bash
    pip install torch torchdiffeq jax[cuda12] diffrax equinox scikit-learn nibabel
    ```

## 2. Scientific Fidelity Benchmarks (Julia vs. Baselines)

These benchmarks evaluate the Pearson correlation ($r$) and MAE against Monte Carlo ground truth.

### Run UDE Champion (Julia)
Evaluates the No-Approx UDE model on the validation set.
```bash
julia --project=. test/debug/eval_all_models_64.jl
```

### Run Python AI Baselines
Evaluates Spect0, SemiDose, and DblurDoseNet.
```bash
python3 test/debug/save_final_benchmarks.py
```

## 3. Cross-Language Performance Benchmarks

These benchmarks measure the forward pass latency for a $64^3$ patch (300h integration).

### Julia (DifferentialEquations.jl)
```bash
julia --project=. test/debug/benchmark_speed.jl
```

### PyTorch (torchdiffeq)
```bash
python3 test/debug/benchmark_torchdiffeq.py
```

### JAX (Diffrax)
```bash
python3 test/debug/benchmark_diffrax.py
```

## 4. Visual Comparison Generation

To generate the comprehensive 3x3 grid figure (`elsarticle/figures_new/dosimetry_comparison_all.png`):

1.  **Generate Binaries**: Run the prediction script to save raw patch data.
    ```bash
    python3 test/debug/generate_all_baselines.py
    ```
2.  **Render Figure**: Use the Matplotlib visualizer.
    ```bash
    python3 generate_final_vis_64.py
    ```

## 5. File Structure for Results
*   `elsarticle/dosimetry/benchmarks_final.txt`: Consolidated multi-model results.
*   `elsarticle/dosimetry/vis_results_64/`: Raw binary patches for visual verification.
*   `elsarticle/figures_new/`: Publication-quality infographics and comparison grids.
