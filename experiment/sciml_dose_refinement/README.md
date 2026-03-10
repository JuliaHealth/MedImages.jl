# Lutetium-177 SPECT Voxel Dosimetry Refinement with Julia SciML

This folder contains experiments for refining approximate 3D dose maps into high-accuracy equivalents (emulating Monte Carlo results) using the SciML ecosystem.

## Goal
Train and compare different architectures that accept a 5D tensor `(W, H, D, C, N)` as input and output a refined 3D dose map:
- 3 spatial dimensions (W, H, D)
- 3 channels (C = 3): Initial dosemap (approximate), CT (anatomy/density), Uncorrected SPECT
- Batch dimension (N)

## Approaches Implemented
1. **PINN-style Refinement** (Physics-Informed Neural Network)
2. **UDE-style Hybrid Refinement** (Universal Differential Equations)
3. **FNO-style Spectral Refinement** (Fourier Neural Operator)
