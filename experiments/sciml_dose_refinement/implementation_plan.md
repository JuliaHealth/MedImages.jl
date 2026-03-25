# Goal Description
Implement the Universal Differential Equation (UDE) model described in `elsarticle/dosimetry/SciML_Dosimetry_UDE.tex` for $^{177}$Lu-PSMA therapy dosimetry. The script will train a neural network corrector $\mathcal{N}_\theta$ to map homogeneous dose rate convolutions (or baseline physics) to high-fidelity Monte Carlo dosemaps, explicitly incorporating CT-derived mass and density, as well as SPECT-derived activity.

## User Review Required
> [!IMPORTANT]
> To implement the framework exactly as described in the LaTeX paper, we need to use `DifferentialEquations.jl` and `SciMLSensitivity.jl` / `Optimization.jl`. These are currently **not** in the `Project.toml` of `MedImages.jl`. Should I add these dependencies to the project, or should I create a self-contained script (e.g., using `Pkg.add(...)` inside the script) or write a custom explicit solver (like Euler/RK4) using the existing `Lux` and `Zygote`/`Enzyme` dependencies?
> Also, what specific model architecture (e.g., U-Net or simple CNN) and loss function do you prefer for $\mathcal{N}_\theta$? I plan to use a basic 3D CNN setup with `Lux.jl`.

## Proposed Changes
### elsarticle/dosimetry/train_ude.jl
#### [NEW] train_ude.jl
Create the main entry point for the training process. 
- Iterate through subdirectories in `test_data/dataset_Lu`.
- Load `SPECT_DATA/CT.nii.gz` to compute the density $\rho$ and mass map $m(\mathbf{r})$.
- Load `SPECT_DATA/NM_Vendor.nii.gz` (SPECT) to set initial observed activity $A_{obs}$.
- Load `SPECT_DATA/Dosemap.nii.gz` as the ground truth target.
- Implement the ODE definitions from the paper (`universal_dosimetry_ude!`).
- Construct a `Lux.jl` Neural Network for `compute_neural_dose_rate`.
- Define an objective function using `SciMLSensitivity` and `Optimization`.
- Train the UDE using the target Dosemap.

## Verification Plan
### Automated Tests
1. Perform a dry-run of the script on a single patient subdirectory (e.g. `FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_0__Pat10`).
2. Verify that the CT to density conversion outputs reasonable values (~1.0 for soft tissue, ~0.3 for lungs, ~1.9 for bone).
3. Check the gradients of the loss function with respect to the neural network parameters using `Zygote` or `Enzyme` to ensure differentiability.

### Manual Verification
1. Run the script for a few epochs and observe the terminal output for decreasing loss.
2. Ensure the script saves the reconstructed/predicted dosemap back to NIfTI format to visually compare it against the ground truth using a medical image viewer.
