# Visual Verification of Batched Transformations

This guide outlines the process for verifying the correctness of batched medical image transformations in `MedImages.jl` using visual inspection. While automated tests cover numerical correctness, visual inspection is crucial for understanding geometric validity, interpolation artifacts, and spatial metadata handling.

## Prerequisites

*   **Julia Environment**: Ensure `MedImages.jl` is instantiated.
*   **Medical Image Viewer**: Tools like [ITK-SNAP](http://www.itksnap.org/), [3D Slicer](https://www.slicer.org/), or [ImageJ](https://imagej.nih.gov/ij/) are recommended. SimpleITK with Python/Matplotlib can also be used.

## Verification Workflow

The workflow involves:
1.  **Generating** distinct synthetic images.
2.  **Batching** them into a single structure.
3.  **Applying** transformations (Rotate, Scale, Translate, Crop, Pad).
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
*   `translated_1.nii.gz`, `translated_2.nii.gz`: Images after batched translation.
*   `cropped_1.nii.gz`, `cropped_2.nii.gz`: Images after batched cropping.
*   `padded_1.nii.gz`, `padded_2.nii.gz`: Images after batched padding.

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
    *   **Spacing check**: If the scaler modifies voxel spacing, verify in the viewer's "Image Information" that the new spacing is double the original (if 0.5x scale means 0.5x resolution/zoom out) or half (if 0.5x means shrinking the image size in mm).
    *   *Implementation Note*: `scale_mi` in `Basic_transformations.jl` interpolates into a new grid. If you check the physical coordinates of a landmark (e.g., sphere center), it should remain at the same physical location, but the voxel indices will change.

#### Translation (e.g., 10 voxels along X)

1.  **Open** `input_1.nii.gz` and `translated_1.nii.gz` as overlays.
2.  **Verify**:
    *   The structures should be shifted.
    *   Check `Origin` in image metadata. A translation usually updates the `Origin` coordinate without moving the voxel data array content, or shifts the voxel data.
    *   *Implementation Note*: `translate_mi` updates the `Origin` metadata. The voxel data array itself usually remains unchanged. In a viewer like ITK-SNAP, if you load both, they might appear aligned if the viewer ignores Origin, or shifted if it respects Origin. Ensure your viewer respects physical coordinates.

#### Cropping (e.g., Center Crop)

1.  **Open** `cropped_1.nii.gz`.
2.  **Verify**:
    *   The image dimensions are smaller (e.g., 32x32x32 vs 64x64x64).
    *   The object should still be centered if the crop was centered.
    *   Check boundaries for cut-off structures.

#### Padding (e.g., 5 voxels all sides)

1.  **Open** `padded_1.nii.gz`.
2.  **Verify**:
    *   The image dimensions are larger.
    *   The original image content is surrounded by the padding value (usually 0).
    *   The physical alignment (Origin) should be adjusted so the original content stays in the same physical space.

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
