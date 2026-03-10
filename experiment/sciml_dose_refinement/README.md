# Lutetium-177 SPECT Voxel Dosimetry Refinement with Julia SciML

This folder contains experiments for refining approximate 3D dose maps into high-accuracy equivalents (emulating Monte Carlo results) using the SciML ecosystem.

## Goal
Train and compare different architectures that accept a 5D tensor `(W, H, D, C, N)` as input and output a refined 3D dose map:
- 3 spatial dimensions (W, H, D)
- 3 channels (C = 3): Initial dosemap (approximate), CT (anatomy/density), Uncorrected SPECT
- Batch dimension (N)

## Approaches Implemented
1. **PINN-style Refinement** (Physics-Informed Neural Network): Uses 3D Convolutional Neural Networks (CNNs) in `Lux.jl` and a customized loss function to enforce basic physical consistency (e.g., total energy conservation across channels).
2. **FNO-style Spectral Refinement** (Fourier Neural Operator): Uses `NeuralOperators.jl` to learn mapping kernels in the frequency domain, offering resolution invariance (ideal when clinics provide images at different resolutions).
3. **UDE-style Hybrid Refinement** (Universal Differential Equations): Uses `DifferentialEquations.jl` and `SciMLSensitivity.jl` to model dose refinement as an Initial Value Problem (IVP). It computes the corrected dose map by integrating a combined mechanistic equation (representing the Local Deposition Method) alongside a parameterized neural network capturing scatter components over time.

---

## Adapting to Real Medical Data

The `run_experiment.jl` script uses small synthetic, randomly generated arrays to ensure the computational graphs and Zygote gradients compile successfully. To adapt these algorithms for your clinical pipeline, follow the steps below.

### 1. Data Preparation
To map real data into the required 5D structure `(W, H, D, C, N)`:
1. Load your NIfTI or DICOM data using the core `MedImages.jl` functionality.
2. Resample all images (Initial Dosemap, CT, and SPECT) to a common uniform grid spacing and origin using `resample_to_image` or `resample_to_spacing` in `MedImages.jl`.
3. Create a unified patient array matrix where the channels dimension (`C`) concatenates the different modalities.

**Example data loading and tensor creation:**

```julia
using MedImages
using MLUtils

function create_real_data_loader(patient_filepaths, target_filepaths)
    W, H, D = 128, 128, 128
    N = length(patient_filepaths)

    # 3 Channels: Initial Dosemap, CT, Uncorrected SPECT
    C_in = 3
    X_tensor = zeros(Float32, W, H, D, C_in, N)
    Y_tensor = zeros(Float32, W, H, D, 1, N) # Ground truth Monte Carlo

    for (i, (patient_paths, mc_path)) in enumerate(zip(patient_filepaths, target_filepaths))
        # 1. Load MedImages
        dosemap = load_image(patient_paths.dosemap)
        ct = load_image(patient_paths.ct)
        spect = load_image(patient_paths.spect)
        mc_truth = load_image(mc_path)

        # 2. Assume all are already resampled to match dimensions (W,H,D)
        # Extract the 3D voxel data arrays
        dose_arr = get_spatial_data(dosemap)
        ct_arr = get_spatial_data(ct)
        spect_arr = get_spatial_data(spect)
        mc_arr = get_spatial_data(mc_truth)

        # 3. Fill the tensors
        X_tensor[:, :, :, 1, i] .= dose_arr
        X_tensor[:, :, :, 2, i] .= ct_arr
        X_tensor[:, :, :, 3, i] .= spect_arr

        Y_tensor[:, :, :, 1, i] .= mc_arr
    end

    # Generate the loader. Adjust batchsize to fit your GPU memory!
    return DataLoader((X_tensor, Y_tensor); batchsize=2, shuffle=true)
end
```

### 2. Modifying `run_experiment.jl`
Replace the `create_dummy_data` function invocation in the `run_all_experiments` routine with your customized real-data loader shown above.

```julia
# Replace this line:
# loader = create_dummy_data(; W=16, H=16, D=16, C_in=3, C_out=1, N=4, num_samples=8)

# With your implementation:
loader = create_real_data_loader(train_patients, train_targets)
```

### 3. Hyperparameter Tuning
When running on real data, you will likely need to adjust the model architectures to accommodate the structural complexity:
* **FNO**: The Fourier modes `modes=(4, 4, 4)` and `width=16` in `create_fno_model()` work well for small matrices, but for $128^3$ patient volumes, experiment with `modes=(16, 16, 16)` and `width=32`.
* **UDE**: The manual integration steps (`dt=0.5f0`) might need smaller step sizes (`dt=0.1f0`) or replacement with a formal ODE solver interface depending on the non-linearity of the scatter correction.
* **Epochs**: Increase from `epochs=5` to hundreds. Consider adding early stopping mechanisms inside the generic `train_model!` loop.
* **GPU Utilization**: Add `|> gpu` to the models and batch inputs within the training loop to leverage `CUDA.jl` for intensive real data tasks.
