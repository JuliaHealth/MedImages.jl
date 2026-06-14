# Compliance Traceability Matrix (CTM)

This matrix maps project requirements to their corresponding implementation artifacts and verification tests, following the V-Model methodology.

| Req ID | Requirement Description | Implementation Artifact | Verification Level | Verification Artifact | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-001** | High-throughput 3D PET/CT Preprocessing (Scalability) | `src/Basic_transformations.jl`, `src/Load_and_save.jl` | System Testing | `test/test_batched.jl`, `experiments/benchmark/gpu_benchmarks.jl` | Verified |
| **REQ-002** | GPU-accelerated Spatial Transformations (Performance) | `src/Basic_transformations.jl` (KernelAbstractions.jl) | Integration Testing | `experiments/benchmark/run_gpu_benchmarks.jl` | Verified |
| **REQ-003** | End-to-end Differentiability for SciML | `src/Basic_transformations.jl` (Zygote/Enzyme compatibility) | Unit Testing | `test/differentiability_tests/` | Verified |
| **REQ-004** | Metadata-Voxel binding and SUV Accuracy | `src/MedImage_data_struct.jl`, `src/SUV.jl` | Acceptance Testing | `experiments/suv_consistency/run_batch_consistency.jl` | Verified |
| **REQ-005** | Repository Structural Integrity (Refactor) | Project Root Reorganization | Static Analysis | Directory Structure Review | Verified |
| **REQ-006** | Julia 1.10+ Compatibility | `Project.toml` (compat section) | Unit Testing | `test/runtests.jl` | Verified |
| **REQ-007** | Automated Test Data Synthesis | `test/generate_large_test_data_py.jl` | Verification Testing | `test/runtests.jl` (Broken -> Pass) | In Progress |

## Refactor Plan (V-Model Architecture Design)

| Source Path | Destination Path | Justification |
| :--- | :--- | :--- |
| `*.py`, `*.jl`, `*.sh` (root) | `test/` or `article/scripts/` | Clean root directory, separate scripts from library core. |
| `val_outputs*/` | `data/validation/` | Centralize transient validation artifacts under `data/`. |
| `viz_options/` | `article/viz_options/` | Group visualization assets with the manuscript. |
| `*.png`, `*.csv` (root) | `article/figures/` or `data/results/` | Move results to artifact folders. |
