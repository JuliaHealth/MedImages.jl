# Visual Verification of MedImages.jl

This guide outlines the process for verifying the correctness of batched medical image transformations and learning algorithms in `MedImages.jl` using visual inspection.

## Prerequisites

*   **Julia Environment**: Ensure `MedImages.jl` is instantiated.
*   **Medical Image Viewer**: Tools like [ITK-SNAP](http://www.itksnap.org/), [3D Slicer](https://www.slicer.org/), or [ImageJ](https://imagej.nih.gov/ij/) are recommended.
*   **Python**: Required for plotting 3D differentiability trajectories (Matplotlib).

### Acquiring Clinical Data (autoPET-III-Lite)
For experiments requiring clinical PET/CT studies (e.g., Series 4 SUV Consistency), you can use the open-access autoPET-III-Lite dataset.
1.  **Download**: Obtain the dataset from Hugging Face: [YongchengYAO/autoPET-III-Lite](https://huggingface.co/datasets/YongchengYAO/autoPET-III-Lite)
2.  **Extract**: Extract the contents and organize them into patient-specific directories.
3.  **Setup**: Place these directories into the `data/dosimetry_data/` folder (or update the `DATA_ROOT` variable in the respective scripts like `run_batch_consistency.jl`). The scripts expect paired `ct.nii.gz` and `spect.nii.gz` (or PET equivalents) inside each patient's directory.

## Verification Series 1: Batched Transformations

### 1. Run the Generation Script

```bash
julia --project=. test/generate_visual_samples.jl
```

This script creates `test/visual_output/` containing:
*   `input_1.nii.gz`, `input_2.nii.gz`: Original synthetic images.
*   **Rotated**: `rotated_0deg_1.nii.gz`, `rotated_45deg_2.nii.gz`.
*   **Scaled**: `scaled_0.5x_1.nii.gz`, `scaled_0.5x_2.nii.gz`.
*   **Translated**: `translated_1.nii.gz`, `translated_2.nii.gz`.
*   **Cropped/Padded**: `cropped_*.nii.gz`, `padded_*.nii.gz`.
*   **Affine Shear**: `sheared_xy_0.5_2.nii.gz`.
*   **Fused Affine**: `fused_transform_*.nii.gz` (Combined composition).
*   **Resample to Spacing**: `resample_spacing_2mm_*.nii.gz`.
*   **Resample to Image**: `resample_to_img_*.nii.gz`.

### 2. Visual Inspection Guide

#### Rotation (e.g., Unique Angles per Image)

1.  **Open** `input_1.nii.gz` and `rotated_0deg_1.nii.gz`.
    *   **Verify**: They should be identical (0-degree rotation).
2.  **Open** `input_2.nii.gz` and `rotated_45deg_2.nii.gz`.
    *   **Verify**: `rotated_45deg_2` should be rotated 45 degrees relative to `input_2`. Check the center of rotation (typically image center).

#### Scaling (e.g., 0.5x Shared Factor)

1.  **Open** `input_2.nii.gz` and `scaled_0.5x_2.nii.gz`.
2.  **Verify**:
    *   The object in `scaled_0.5x_2` should appear spatially consistent but resampled.
    *   **Spacing check**: In the viewer's "Image Information", verify that the new spacing is different (typically adapted to maintain FOV or resolution logic).

#### Translation (e.g., 10 voxels along X)

1.  **Open** `input_1.nii.gz` and `translated_1.nii.gz` as overlays.
2.  **Verify**:
    *   The structures should be shifted.
    *   Check `Origin` in image metadata. A translation updates the `Origin` coordinate.

#### Cropping & Padding

1.  **Open** `cropped_1.nii.gz`.
    *   **Verify**: Image dimensions are smaller (32^3 vs 64^3).
2.  **Open** `padded_1.nii.gz`.
    *   **Verify**: Image dimensions are larger. Original content centered with padding border.

#### Affine Shearing

1.  **Open** `sheared_xy_0.5_2.nii.gz` alongside `input_2.nii.gz`.
2.  **Verify**:
    *   The object should be skewed. For an XY shear, the X coordinate should shift linearly with Y.

#### Fused Affine (Combined Rotate + Scale + Translate)

1.  **Open** `fused_transform_2.nii.gz` alongside `input_2.nii.gz`.
2.  **Verify**:
    *   The object should be simultaneously rotated (30 deg Z), scaled (0.8x), and translated (X+5, Y-2).
    *   This verifies that the **fused kernel** correctly handles the composition of affine matrices in a single interpolation pass, reducing artifacts.

#### Resample to Spacing

1.  **Open** `resample_spacing_2mm_1.nii.gz`.
2.  **Verify**:
    *   Voxel dimensions should be half of original (32^3 vs 64^3) assuming spacing doubled (1mm -> 2mm).
    *   Object should look blockier (downsampled) but occupy the same physical space.

#### Resample to Image

1.  **Open** `resample_to_img_1.nii.gz` and `input_1.nii.gz`.
2.  **Verify**:
    *   The output should match the geometry (Origin, Spacing, Size) of the *reference* image used in the script.

## Verification Series 2: Visual Differentiation (Learning Experiment)

This experiment proves that `MedImages.jl` transformations are usefully differentiable by training a CNN to "undo" random 3D rotations.

### 1. Run the Differentiability Proof

```bash
julia --project=. experiments/differentiability_proof.jl
```

This generates artifacts in `data/validation/`:
*   `gold_standard.nii.gz`: The target orientation (vertical line).
*   `uncorrected.nii.gz`: A randomly rotated test sample (initial state).
*   `reconstructed.nii.gz`: The test sample after the CNN learned to rotate it back.
*   `endpoints.csv`: Trajectory of the line endpoints over epochs.

### 2. Visual Inspection

1.  **Open** all three `.nii.gz` files in a viewer.
2.  **Verify**: `reconstructed.nii.gz` should be aligned vertically, matching `gold_standard.nii.gz`, while `uncorrected.nii.gz` is tilted. This confirms the gradients flowed correctly through the `rotate_mi` operation.

### 3. Generate Trajectory Plot

```bash
python3 article/scripts/plot_3d_lines.py data/validation
```

*   **Verify**: The resulting plot should show the line segments converging from random initial orientations (red/light) to the vertical target orientation (blue/dark) over training epochs.

## Verification Series 3: Dosimetry Training (Synthetic/Dummy Data)

Verify that the dosimetry training loop (PINN/UDE) converges correctly on synthetic random tensors.

### 1. Run Dummy Training Experiment

```bash
julia --project=experiments/sciml_dose_refinement/ experiments/sciml_dose_refinement/run_experiment.jl
```

### 2. Manual Verification

*   **Observe Console Output**: The script generates 5D dummy tensors (W, H, D, C, N).
*   **Verify Loss**: Ensure that the "Training Loss" and "Validation Loss" decrease over iterations (e.g., from ~0.25 to ~0.08).
*   **Verify Shape**: Confirm the model outputs match the target size (e.g., `(16, 16, 16, 1, 4)` for a batch).

## Verification Series 4: End-to-End SUV Consistency

Verify that the system preserves clinical metadata and quantitative accuracy across compound spatial transformations. This reproduces the results reported in Challenge 4 of the manuscript.

### 1. Run SUV Precision Identity Test

```bash
julia --project=. experiments/suv_consistency/verify_precision_batch.jl
```

*   **Verify Console Output**: The script compares MedImages.jl SUV calculations against the Python `SimpleITK` standard. Confirm that the **Global Max Relative Difference** across the 10 cases is `< 1e-14`, validating absolute precision.

### 2. Run Compound Transformation Consistency

```bash
julia --project=. experiments/suv_consistency/run_batch_consistency.jl
```

*   **Verify Console Output & CSV**: The script uses `TotalSegmentator` to isolate organs, then subjects the images to alignment, 2.0mm isotropic resampling, 45-degree rotation, and translation.
*   **Success Criterion**: Check the printed summary or `experiments/suv_consistency/batch_results_10cases.csv`. The **Mean SUV Deviation** should be exceptionally low (e.g., `~0.17%`), proving that metadata binding is robust to spatial manipulation.

## Verification Series 5: Full-Body Dosimetry Residuals

Visually compare the UDE model's predictive accuracy against the PyTomography analytical baseline and other deep learning models.

### 1. Generate 3x3 Comparison Grid

```bash
python3 experiments/sciml_dose_refinement/plot_full_body_3x3.py
```

### 2. Visual Inspection

*   **Open Artifact**: Check the generated output (usually placed in the validation data directories or `val_outputs/` as `full_body_comparison_3x3.png`).
*   **Verify**: The grid displays the Ground Truth CT, Monte Carlo Dose, and UDE Dose in the top row. The bottom rows display the subtraction residuals (Error Maps).
*   **Success Criterion**: The UDE residual map should appear significantly closer to pure white (zero error) compared to the Analytical Baseline, Spect0Net, and DblurDoseNet, especially at heterogeneous tissue boundaries.

## Verification Series 6: Scalability and Performance Benchmarks

Verify the performance claims (Challenge 1) comparing GPU vs CPU execution times across different transformations.

### 1. Run the GPU Benchmarks

Ensure the test environment has access to a CUDA-capable GPU, then run:

```bash
julia --project=. experiments/benchmark/run_gpu_benchmarks.jl
```

*   **Verify Console Output**: The script executes 3D rotations, resampling, and scaling on synthetic $256^3$ volumes. Confirm that the GPU execution times are significantly lower than CPU times (often 50-100x faster for fused affine operations).
*   **Artifacts**: The results are saved to `data/results/benchmarks_final.txt`.

### 2. Generate Performance Plots

Use the provided Python script to visualize the benchmark results:

```bash
python3 article/scripts/plot_gpu.py
```

*   **Visual Inspection**: Open the generated `gpu_benchmark_plot.png`. Ensure it correctly maps the execution times, clearly demonstrating the performance advantage of the custom `KernelAbstractions.jl` GPU implementation over baseline CPU methods.

## Manual Acceptance Test: Differentiable Dosimetry Refinement

This procedure combines the verification of the Scientific Machine Learning (SciML) training loop with the visual confirmation of gradient-based optimization.

### 1. Test Setup
*   Ensure Julia 1.12.5 and Python 3.10+ are available.
*   Verify that `data/validation/` exists and is writable.

### 2. Execution Phase A: Training Loop Convergence
Run the dummy dosimetry training to confirm the PINN/UDE engine is functional:
```bash
julia --project=experiments/sciml_dose_refinement/ experiments/sciml_dose_refinement/run_experiment.jl
```
*   **Success Criterion**: Loss decreases monotonically from epoch 1 to final epoch.
*   **Verification**: Check that `experiments/sciml_dose_refinement/logs/` contains the training logs.

### 3. Execution Phase B: Visual Gradient Proof
Run the rotation-based differentiability proof to confirm that gradients propagate through spatial transformations correctly:
```bash
julia --project=. experiments/differentiability_proof.jl
python3 article/scripts/plot_3d_lines.py data/validation
```
*   **Success Criterion**: The blue/dark lines in the generated plot (`article/viz_options/option_blue_gradient.png`) are vertical, representing the successful recovery of the target orientation via backpropagation.
*   **Verification**: The loss reported in the console should drop significantly (typically >60% reduction).

### 4. Integration Verification
This combined evidence confirms that:
1.  The **Dosimetry Engine** can ingest tensors and compute losses.
2.  The **Spatial Transformer** is differentiable, allowing the engine to optimize anatomical alignment and physics parameters simultaneously.
