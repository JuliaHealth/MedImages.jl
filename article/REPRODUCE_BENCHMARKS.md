# Reproducing the MedImages.jl UDE Benchmarks

This document provides instructions on how to reproduce the framework-level speed and memory benchmarks comparing **Julia SciML**, **PyTorch**, and **JAX** for the Neural Dosimetry Universal Differential Equations (UDEs) described in the article.

## Can these be run on synthetic data?

**Yes.** To allow for fully open-source and unrestricted reproducibility without needing access to protected clinical patient data, all raw computational speed and memory comparisons across the different frameworks were deliberately designed to use **synthetic random tensors**. 

The benchmarks construct a $32 \times 32 \times 32$ voxel patch, populate it with standardized random values simulating the tissue density and activity gradients, and then perform the 300-hour physics integration. This completely isolates the compiler, autodiff backend, and ODE solver performance from any I/O bottleneck or NIfTI loading overhead.

## Requirements

1.  **Hardware**: An NVIDIA GPU (benchmarks in the paper were run on an H100, but any modern CUDA-capable GPU will work, though absolute times will scale).
2.  **Julia Environment**: Julia 1.12.5 (or higher). Ensure the `MedImages.jl` project environment is instantiated.
3.  **Python Environment**: Python 3.10+ with `torch`, `torchdiffeq`, `jax[cuda12]`, `diffrax`, `equinox`, and `scipy` installed.

## How to Run the Benchmarks

All benchmark scripts are located in the `experiments/sciml_dose_refinement/` directory.

### 1. Inference Speed (Forward Pass Only)

This tests how fast each framework can integrate the 300-hour system without tracking gradients.

*   **Julia SciML**:
    ```bash
    julia --project=. experiments/sciml_dose_refinement/benchmark_speed.jl
    ```
*   **PyTorch (`torchdiffeq`) & Hybrid Python (`scipy.integrate`)**:
    ```bash
    python3 experiments/sciml_dose_refinement/benchmark_python_udes.py
    ```
*   **JAX (`diffrax`)**:
    ```bash
    python3 experiments/sciml_dose_refinement/benchmark_jax_ude.py
    ```

### 2. Training Speed (Forward + Backward Pass) & Peak Memory

This tests the execution time and the massive GPU memory footprint required to backpropagate through the ODE solver loops.

*   **Julia SciML (`BacksolveAdjoint` + `ZygoteVJP`)**:
    ```bash
    julia --project=. experiments/sciml_dose_refinement/benchmark_train_julia.jl
    ```
*   **PyTorch**:
    ```bash
    python3 experiments/sciml_dose_refinement/benchmark_train_python.py
    ```
*   **JAX**:
    ```bash
    python3 experiments/sciml_dose_refinement/benchmark_train_jax.py
    ```

*Note: All scripts execute a "warmup" pass before recording times to ensure that Just-In-Time (JIT) compilation overhead is strictly excluded from the benchmark metrics.*