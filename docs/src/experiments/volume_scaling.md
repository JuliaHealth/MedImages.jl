# Challenge 1: Volume Scaling (HDF5 vs. Caching)

**Objective:** Evaluate framework throughput under biobank-scale production loads by benchmarking a 100-case multi-modal preprocessing pipeline (PET/CT) on the autoPET dataset.

## The Bottleneck: Multimodal Alignment

In clinical practice, aligning a functional image (PET) to an anatomical reference (CT) requires continuous spatial resampling. Both must share the identical voxel spacing, direction cosines, and origin to be stacked into a deep learning tensor. We created a fused affine pipeline that collapses orientation normalization, spacing resampling, and spatial centering into a single 4x4 matrix executed on the GPU.

## Detailed Code Walkthrough

The core advantage of MedImages.jl in volume scaling comes from avoiding sequential filtering loops and instead compiling transformations into a single fused GPU kernel. Let's walk through how this is implemented by comparing it to a typical legacy approach.

### The Legacy Approach vs Fused Approach

In many traditional Python/ITK pipelines, if you want to apply different transformations to a batch of images, you iterate over them or perform multiple sequential interpolation passes. 

```julia
# File: experiments/benchmark/fused_vs_legacy.jl

function legacy_affine_transform_mi(image::BatchedMedImage, affine_matrices::Union{Matrix{Float64}, Vector{Matrix{Float64}}}, Interpolator::Interpolator_enum; output_size=nothing)
    spatial_size = isnothing(output_size) ? size(image.voxel_data)[1:3] : output_size
    backend = get_backend(image.voxel_data)
    
    # 1. Pre-process matrices
    # We iterate over the batch to invert each transformation matrix.
    batch_size = size(image.voxel_data, 4)
    matrices_inv_list = map(1:batch_size) do b
        M = (affine_matrices isa Vector) ? affine_matrices[b] : affine_matrices
        M_inv = inv(M)
        return Float32.(M_inv)
    end
    # We concatenate them into a 3D tensor of matrices [4, 4, Batch]
    matrices_inv = cat(matrices_inv_list..., dims=3)

    # 2. Generate Coordinates
    # This creates the sampling grid for the entire batch based on the inverse matrices.
    points_to_interpolate = generate_affine_coords(spatial_size, matrices_inv, backend)
    
    # 3. Perform Interpolation
    # The actual heavy lifting. `interpolate_my` uses KernelAbstractions.jl to 
    # execute on the GPU. This is a single kernel launch for the entire batch.
    spacing_arg = [ (1.0, 1.0, 1.0) for _ in 1:batch_size ]
    resampled_flat = interpolate_my(points_to_interpolate, image.voxel_data, spacing_arg, Interpolator, false, 0.0, true)
    
    return reshape(resampled_flat, spatial_size[1], spatial_size[2], spatial_size[3], batch_size)
end
```

### Line-by-Line Breakdown:

1. **Lines 9-11:** The function takes a `BatchedMedImage` (a 4D array where the 4th dimension is the batch). It determines the output spatial size and the compute backend (CPU or CUDA).
2. **Lines 13-20:** The pipeline applies transformations by pulling from the *target* grid back to the *source* grid. Therefore, it requires the inverse of the affine matrix (`inv(M)`). This block handles both a single shared matrix or a unique matrix per batch item, converting them to `Float32` for GPU efficiency.
3. **Line 23 (`generate_affine_coords`):** Instead of looping over each image to calculate where its pixels land, this function generates the 3D sampling coordinates for the entire batch simultaneously. If running on CUDA, these coordinates are generated directly in VRAM.
4. **Lines 28-29 (`interpolate_my`):** The generated coordinates are passed to the fused interpolation kernel. `true` at the end of the argument list signifies using the fast GPU kernel path. Because the matrices and coordinates were batched, this interpolation happens in a single massively parallel pass.
5. **Line 31:** The flat array returned from the GPU kernel is reshaped back into a 4D `(X, Y, Z, Batch)` tensor.

## Results

MedImages.jl achieved a 7.2x total turnaround speedup compared to the MONAI PersistentDataset baseline. 

*   **Load from Cache:** ~40 ms (MedImages) vs ~150 ms (MONAI)
*   **Transform (Compute):** ~50 ms (MedImages) vs ~500 ms (MONAI)

The integration with HDF5 bypasses serialization costs, and the Fused Affine kernel eliminates memory traffic associated with sequential operation queues.
