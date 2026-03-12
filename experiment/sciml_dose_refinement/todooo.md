SciML-Based Attenuation Correction & Dosimetry Refinement
This document details how to run the high-resolution quantitative dosimetry and attenuation correction pipeline using the Julia SciML ecosystem within the MedImages.jl framework.

1. Concept Overview
The goal is to refine approximate 3D dose maps from SPECT reconstructions into high-accuracy equivalents (emulating Monte Carlo results). This is achieved by incorporating physical priors (attenuation from CT, activity from SPECT) into neural architectures.

We implement three complementary SciML approaches:

PINN (Physics-Informed Neural Network): Enforces energy conservation via specialized loss functions.
FNO (Fourier Neural Operator): Learns resolution-invariant mappings in the frequency domain.
UDE (Universal Differential Equations): Models dose deposition as a dynamic system where scatter is learned as a residual term.
2. Prerequisites & Environment
Project Structure
experiment/sciml_dose_refinement/run_experiment_lu.jl
: Main entry point.
experiment/sciml_dose_refinement/lu_loader.jl
: Data loading and resampling.
slurm/*.sh: Submission scripts for cluster deployment.
Setup
Activate the environment and install dependencies:

bash
cd experiment/sciml_dose_refinement
julia --project=. -e 'using Pkg; Pkg.develop(path="../../"); Pkg.instantiate()'
3. Data Preparation Flow
The pipeline expects clinical data in NIfTI format organized by patient folders.

Loading: The load_patient_data function reads:
SPECT_Recon_WholeBody.nii.gz: Uncorrected activity proxy.
CT.nii.gz: Anatomy/Density map for attenuation.
Dosemap.nii.gz: Target Monte Carlo ground truth.
Resampling: All modalities are resampled to a unified grid (e.g., 256³) using MedImages.jl interpolation.
Normalization:
SPECT/Dose: Min-max normalized.
CT: Hounsfield Units (HU) are clipped to [-1000, 2000] and normalized.
Tensor Assembly: Data is packed into a 5D tensor: (Width, Height, Depth, Channels, Samples).
Channel 1: SPECT
Channel 2: CT
4. Running the Experiments
You can run individual models or the entire suite.

Local Execution
bash
# Run PINN refinement
julia --project=. run_experiment_lu.jl pinn /path/to/data
# Run UDE refinement
julia --project=. run_experiment_lu.jl ude /path/to/data
Slurm Deployment
Use the provided batch scripts for H100 GPU acceleration. These scripts include automatic cache cleanup and race-condition prevention:

bash
sbatch experiment/sciml_dose_refinement/slurm/pinn_slurm.sh
sbatch experiment/sciml_dose_refinement/slurm/fno_slurm.sh
sbatch experiment/sciml_dose_refinement/slurm/ude_slurm.sh
5. Model Details
PINN: Physics-Informed CNN
The PINN uses a 3D-CNN architecture. Its loss function is defined as: $$Loss = Loss_{Data} + \lambda \cdot Loss_{Physics}$$ where $Loss_{Physics}$ enforces that the total predicted energy $(\sum Dose)$ remains consistent with the total input activity $(\sum SPECT)$.

FNO: Resolution Invariance
The FNO maps input distributions to dose distributions using Fourier spectral layers. This allows the model trained on one resolution to be applied to different voxel sizes without performance degradation.

UDE: Hybrid Differential Equations
The UDE treats dose refinement as an Initial Value Problem (IVP): $$\frac{du}{dt} = f_{mechanistic}(A, CT, u) + NN(A, CT, u; \theta)$$

Mechanistic term: Represents local energy deposition.
Neural term ($NN$): Learns the complex, non-local scatter and attenuation components that are typically hard to model analytically.
The refined dose is obtained by integrating this system over a sequence of "refinement steps".
6. Monitoring & Validation
Logs: Check logs/ for per-epoch loss metrics.
Checkpoints: Models save .jls state every 10 epochs.
Visualization: Use the resulting .jls to run inference and export NIfTI results via MedImages.jl for comparison in 3D Slicer.