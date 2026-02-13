# Visual Verification of Batched Transformations

This guide outlines the process for verifying the correctness of batched medical image transformations in `MedImages.jl` using visual inspection. While automated tests cover numerical correctness, visual inspection is crucial for understanding geometric validity, interpolation artifacts, and spatial metadata handling.

## Prerequisites

*   **Julia Environment**: Ensure `MedImages.jl` is instantiated.
*   **Medical Image Viewer**: Tools like [ITK-SNAP](http://www.itksnap.org/), [3D Slicer](https://www.slicer.org/), or [ImageJ](https://imagej.nih.gov/ij/) are recommended. SimpleITK with Python/Matplotlib can also be used.

## Verification Workflow

The workflow involves:
1.  **Generating** distinct synthetic images.
2.  **Batching** them into a single structure.
3.  **Applying** transformations (Rotate, Scale, etc.).
4.  **Unbatching** and **Saving** the results.
5.  **Visually Comparing** the input and output.

### 1. Run the Generation Script

A helper script `test/generate_visual_samples.jl` is provided to automate the generation of test files.

```bash
julia --project=. test/generate_visual_samples.jl
```

This script will create a directory `test/visual_output/` containing:
*   `input_1.nii.gz`, `input_2.nii.gz`: Original synthetic images.
*   `rotated_1.nii.gz`, `rotated_2.nii.gz`: Images after batched rotation.
*   `scaled_1.nii.gz`, `scaled_2.nii.gz`: Images after batched scaling.
*   Other transformations if configured.

### 2. Visual Inspection Guide

#### Rotation (e.g., 90 degrees around Z-axis)

1.  **Open** `input_1.nii.gz` and `rotated_1.nii.gz` in your viewer.
2.  **Verify**:
    *   The structure in `rotated_1` should be rotated 90 degrees counter-clockwise (or clockwise depending on convention) relative to `input_1`.
    *   The center of rotation usually defaults to the physical center of the image.
    *   **Check corners**: Ensure no unexpected clipping occurs unless `crop=true` was implicitly used with a smaller field of view.
    *   **Check aliasing**: Step boundaries should look reasonable for the chosen interpolator (Linear vs Nearest).

#### Scaling (e.g., 0.5x)

1.  **Open** `input_2.nii.gz` and `scaled_2.nii.gz`.
2.  **Verify**:
    *   The object in `scaled_2` should appear **larger** relative to the voxel grid (zoomed in) if the scaling factor reduces the spacing, or the field of view covers less physical space.
    *   *Clarification*: `scale_mi` typically scales the image geometry. If scale < 1.0 (e.g., 0.5), the image might shrink (downsample) or the spacing might increase. Check if the physical size matches.
    *   In `MedImages.jl`, `scale_mi` often resamples the image. If you scale by 0.5, the resulting image dimension might be half the original size (if maintaining spacing) or the same size with larger spacing?
    *   **Behavior**: `scale_mi` in this package constructs a new grid. If you scale by 0.5, the output array size usually decreases (downsampling). Verify that the resolution is lower (blockier) or the image size (in voxels) is smaller.

#### Unique per-batch Parameters

The script applies different rotations to each image in the batch (e.g., 0° for image 1, 45° for image 2).

1.  **Verify**:
    *   `rotated_1.nii.gz` (0°): Should look identical to `input_1.nii.gz`.
    *   `rotated_2.nii.gz` (45°): Should be rotated.

## Manual Script Example

If you wish to run this manually in the REPL:

```julia
using MedImages, MedImages.Utils, MedImages.Basic_transformations
using MedImages.MedImage_data_struct

# 1. Create Data
# (See generate_visual_samples.jl for create_synthetic_medimage)
img1 = create_synthetic_medimage((64,64,64), :asym_block)
img2 = create_synthetic_medimage((64,64,64), :sphere)

# 2. Batch
batch = create_batched_medimage([img1, img2])

# 3. Transform (e.g., Rotate Batch with unique angles)
# Image 1 -> 0 deg, Image 2 -> 90 deg
angles = [0.0, 90.0]
batch_rotated = rotate_mi(batch, 3, angles, Linear_en)

# 4. Save
res_imgs = unbatch_medimage(batch_rotated)
create_nii_from_medimage(res_imgs[1], "test/visual_output/rot_0.nii.gz")
create_nii_from_medimage(res_imgs[2], "test/visual_output/rot_90.nii.gz")
```
