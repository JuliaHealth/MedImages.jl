# Task Definition: SciML Dosimetry UDE

## Overview
Implement the Universal Differential Equation (UDE) model described in `elsarticle/dosimetry/SciML_Dosimetry_UDE.tex` to learn voxel-level dosemaps from SPECT/CT data. The dataset is located in `test_data/dataset_Lu`.

## Checklist
- [x] Research existing codebase for MedImages.jl to understand the project structure and data loading utilities.
- [x] Install missing SciML dependencies: `DifferentialEquations.jl`, `Optimization.jl`, `OptimizationOptimJL`, `SciMLSensitivity.jl`.
- [x] Create the data loading logic to iterate over `test_data/dataset_Lu` subfolders and read `SPECT_DATA/CT.nii.gz`, `SPECT_DATA/NM_Vendor.nii.gz` (SPECT), and `SPECT_DATA/Dosemap.nii.gz` (Target).
- [ ] Implement preprocessing steps:
    - [ ] Convert CT HU to mass density ($\rho$) mapping.
    - [ ] Calculate variable mass map from density and voxel volume.
    - [ ] Apply initial activity calibration using assumed/loaded CF and RC.
- [x] Implement the Universal Differential Equation as described in the LaTeX manuscript:
    - [x] State recovery mechanism.
    - [x] Saturable PK Layer (blood, free, bound compartments).
    - [x] Neural Transport Layer (Neural network corrector $\mathcal{N}_\theta$ normalized by mass).
    - [x] BED Integration.
- [x] Setup the Neural Network architecture (using Lux.jl or Flux.jl as part of the SciML ecosystem).
- [x] Create the training loop using Optimization.jl/SciMLSensitivity.jl to learn the parameters (neural weights and missing kinetic parameters) by comparing the predicted Dosemap to the ground truth `Dosemap.nii.gz`.
- [x] Integrate $64 \times 64 \times 64$ random block patching logic to evaluate full dataset dynamically while mitigating massive continuous automatic differentiation trace memory crashes. 
- [x] Final Verification and Code cleanup.
