# Visual Verification of Batched Transformations

This guide outlines the process for verifying the correctness of batched medical image transformations in `MedImages.jl` using visual inspection. While automated tests cover numerical correctness, visual inspection is crucial for understanding geometric validity, interpolation artifacts, and spatial metadata handling.

## Prerequisites

*   **Julia Environment**: Ensure `MedImages.jl` is instantiated.
*   **Medical Image Viewer**: Tools like [ITK-SNAP](http://www.itksnap.org/), [3D Slicer](https://www.slicer.org/), or [ImageJ](https://imagej.nih.gov/ij/) are recommended. SimpleITK with Python/Matplotlib can also be used.

## Verification Workflow

The workflow involves:
1.  **Generating** distinct synthetic images.
2.  **Batching** them into a single structure.
3.  **Applying** transformations (Rotate, Scale, Translate, Crop, Pad, Affine, Resample).
4.  **Unbatching** and **Saving** the results.
5.  **Visually Comparing** the input and output.

### 1. Run the Generation Script

A helper script `test/generate_visual_samples.jl` is provided to automate the generation of test files.

```bash
julia --project=. test/generate_visual_samples.jl
```

This script will create a directory `test/visual_output/` containing:
*   `input_1.nii.gz`, `input_2.nii.gz`: Original synthetic images.
*   **Rotated**: `rotated_0deg_1.nii.gz`, `rotated_45deg_2.nii.gz` (Unique rotations).
*   **Scaled**: `scaled_0.5x_1.nii.gz`, `scaled_0.5x_2.nii.gz` (Shared scaling).
*   **Translated**: `translated_1.nii.gz`, `translated_2.nii.gz` (Shared translation).
*   **Cropped/Padded**: `cropped_*.nii.gz`, `padded_*.nii.gz`.
*   **Affine Shear**: `sheared_id_1.nii.gz` (Identity), `sheared_xy_0.5_2.nii.gz` (Sheared).
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
    *   **Spacing check**: If the scaler modifies voxel spacing, verify in the viewer's "Image Information" that the new spacing is different (typically adapted to maintain FOV or resolution logic).
    *   *Note*: `scale_mi` resamples the image grid. The object physical size should be consistent if scale implies zooming, or smaller if scale implies shrinking.

#### Translation (e.g., 10 voxels along X)

1.  **Open** `input_1.nii.gz` and `translated_1.nii.gz` as overlays.
2.  **Verify**:
    *   The structures should be shifted.
    *   Check `Origin` in image metadata. A translation usually updates the `Origin` coordinate without moving the voxel data array content, or shifts the voxel data.

#### Cropping & Padding

1.  **Open** `cropped_1.nii.gz`.
    *   **Verify**: Image dimensions are smaller (32^3 vs 64^3). Object centered.
2.  **Open** `padded_1.nii.gz`.
    *   **Verify**: Image dimensions are larger. Original content centered with padding border.

#### Affine Shearing

1.  **Open** `sheared_xy_0.5_2.nii.gz` alongside `input_2.nii.gz`.
2.  **Verify**:
    *   The object should be skewed. For an XY shear, the X coordinate should shift linearly with Y.
    *   `sheared_id_1.nii.gz` should be identical to input (Identity transform).

#### Resample to Spacing

1.  **Open** `resample_spacing_2mm_1.nii.gz`.
2.  **Verify**:
    *   Voxel dimensions should be half of original (32^3 vs 64^3) assuming spacing doubled (1mm -> 2mm).
    *   Object should look blockier (downsampled) but occupy the same physical space.

#### Resample to Image

1.  **Open** `resample_to_img_1.nii.gz` and `input_1.nii.gz`.
2.  **Verify**:
    *   The output should match the geometry (Origin, Spacing, Size) of the *reference* image used in the script (which had offset origin `(10,10,10)`).
    *   The content should be the original object resampled into this new frame. If the reference frame is shifted `(10,10,10)`, the object might be shifted within the FOV or cropped if the FOV moved away.

## Manual Script Example

If you wish to run this manually in the REPL:

```julia
using MedImages, MedImages.Utils, MedImages.Basic_transformations
using MedImages.MedImage_data_struct

# 1. Create Data
img1 = create_synthetic_medimage((64,64,64), :asym_block)
img2 = create_synthetic_medimage((64,64,64), :sphere)

# 2. Batch
batch = create_batched_medimage([img1, img2])

# 3. Transform (e.g., Rotate Batch with unique angles)
angles = [0.0, 90.0]
batch_rotated = rotate_mi(batch, 3, angles, Linear_en)

# 4. Save
res_imgs = unbatch_medimage(batch_rotated)
create_nii_from_medimage(res_imgs[1], "test/visual_output/rot_0.nii.gz")
create_nii_from_medimage(res_imgs[2], "test/visual_output/rot_90.nii.gz")
```
