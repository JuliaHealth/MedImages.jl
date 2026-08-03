module Resample_to_target
using Interpolations
using Statistics
using CUDA
using KernelAbstractions
using ChainRulesCore
using LinearAlgebra

using ..MedImage_data_struct, ..Utils, ..Orientation_dicts, ..Spatial_metadata_change, ..Load_and_save
export resample_to_image, scale, resample_to_spacing

# Helper function to compute extrapolation value - not differentiable
function compute_extrapolate_value(voxel_data)
    corners = Utils.extract_corners(voxel_data)
    return median(corners)
end
ChainRulesCore.@non_differentiable compute_extrapolate_value(::Any)

"""
overwriting this function from Interpolations.jl becouse check_ranges giving error
"""
function scale(itp::AbstractInterpolation{T,N,IT}, ranges::Vararg{AbstractRange,N}) where {T,N,IT}
    # overwriting this function becouse check_ranges giving error
    # check_ranges(itpflag(itp), axes(itp), ranges)
    ScaledInterpolation{T,N,typeof(itp),IT,typeof(ranges)}(itp, ranges)
end


"""
    resample_to_image(im_fixed::MedImage, im_moving::MedImage, interpolator_enum::Interpolator_enum, value_to_extrapolate=Nothing)::MedImage

Resample a 'moving' image to match the grid and spatial metadata of a 'fixed' image.

This function aligns the `im_moving` image to the `im_fixed` image's physical space, matching its
origin, spacing, direction, and dimensions. This is essential for multi-modal analysis
where different scans need to be perfectly aligned voxel-by-voxel.

# Arguments
- `im_fixed::MedImage`: The reference image whose grid will be matched.
- `im_moving::MedImage`: The image to be resampled/transformed.
- `interpolator_enum`: Interpolation method (e.g., `Linear_en`).
- `value_to_extrapolate`: Value to use for points outside the input volume. If `Nothing`, the median of corners is used.

# Returns
- `MedImage`: The `im_moving` image resampled onto the `im_fixed` image's grid.

# Notes
- If the images have different orientations, the function automatically runs `change_orientation` first.
"""
function resample_to_image(im_fixed::MedImage, im_moving::MedImage, interpolator_enum::Interpolator_enum, value_to_extrapolate=Nothing)::MedImage

    if (value_to_extrapolate == Nothing)
        # Use helper function marked as non-differentiable
        value_to_extrapolate = compute_extrapolate_value(im_fixed.voxel_data)
    end


    # Calculate the affine transform from Fixed Index (I_f) -> Moving Index (I_m)
    # Physical = Origin + Direction * diag(Spacing) * (Index - 1)
    Df = Float64.(reshape(collect(im_fixed.direction), 3, 3))
    Dm = Float64.(reshape(collect(im_moving.direction), 3, 3))
    
    Sf = Diagonal(Float64.(collect(im_fixed.spacing)))
    Sm = Diagonal(Float64.(collect(im_moving.spacing)))
    
    Of = Float64.(collect(im_fixed.origin))
    Om = Float64.(collect(im_moving.origin))

    # M_rot_scale = inv(Dm * Sm) * (Df * Sf)
    M_rot_scale = inv(Dm * Sm) * (Df * Sf)
    
    # M_trans = inv(Dm * Sm) * (Of - Om)
    M_trans = inv(Dm * Sm) * (Of - Om)

    # I_m = M_rot_scale * I_f - M_rot_scale * 1 + M_trans + 1
    # Build 4x4 matrix mapping 1-based I_f -> 1-based I_m
    M = zeros(Float32, 4, 4, 1)
    M[1:3, 1:3, 1] .= Float32.(M_rot_scale)
    M[1:3, 4, 1] .= Float32.(M_trans .- M_rot_scale * [1.0, 1.0, 1.0] .+ [1.0, 1.0, 1.0])
    M[4, 4, 1] = 1.0f0

    new_size = size(im_fixed.voxel_data)
    
    backend = try KernelAbstractions.get_backend(im_fixed.voxel_data) catch; KernelAbstractions.CPU() end
    device_M = backend isa KernelAbstractions.GPU ? CuArray(M) : M
    
    resampled_flat = Utils.interpolate_fused_affine(im_moving.voxel_data, device_M, new_size, interpolator_enum, false, Float32(value_to_extrapolate), (0.0, 0.0, 0.0))

    new_voxel_data = reshape(resampled_flat, new_size)

    if eltype(im_moving.voxel_data) != Float32
        new_voxel_data = Utils.cast_to_array_b_type(new_voxel_data, im_moving.voxel_data)
    end

    new_im = Load_and_save.update_voxel_and_spatial_data(im_moving, new_voxel_data, im_fixed.origin, im_fixed.spacing, im_fixed.direction)

    return new_im
end

function resample_to_image(im_fixed::BatchedMedImage, im_moving::BatchedMedImage, interpolator_enum::Interpolator_enum, value_to_extrapolate=Nothing)::BatchedMedImage
    batch_size = size(im_fixed.voxel_data, 4)
    if size(im_moving.voxel_data, 4) != batch_size
        error("Batch sizes must match")
    end

    # We must assume fixed images all have same spatial dims (enforced by BatchedMedImage struct logic),
    # but potentially different origins/spacings.
    # And we want to resample each moving image [b] to fixed image [b].

    # Strategy:
    # 1. Generate points_to_interpolate for each batch item.
    #    Since origins/spacings can differ, points might differ.
    #    Output grid size is same for all (im_fixed property).

    new_size = size(im_fixed.voxel_data)[1:3]
    # Calculate affine matrices for the entire batch
    # M maps Target Index -> Source Index
    # I_mov = ( (I_fix - 1)*S_fix + O_fix - O_mov ) / S_mov + 1
    #       = I_fix * (S_fix/S_mov) + [ 1 - S_fix/S_mov + (O_fix - O_mov)/S_mov ]
    
    M_batch = zeros(Float32, 4, 4, batch_size)
    for b in 1:batch_size
        Df = Float64.(reshape(collect(im_fixed.direction[b]), 3, 3))
        Dm = Float64.(reshape(collect(im_moving.direction[b]), 3, 3))
        Sf = Diagonal(Float64.(collect(im_fixed.spacing[b])))
        Sm = Diagonal(Float64.(collect(im_moving.spacing[b])))
        Of = Float64.(collect(im_fixed.origin[b]))
        Om = Float64.(collect(im_moving.origin[b]))

        M_rot_scale = inv(Dm * Sm) * (Df * Sf)
        M_trans = inv(Dm * Sm) * (Of - Om)

        M_batch[1:3, 1:3, b] .= Float32.(M_rot_scale)
        M_batch[1:3, 4, b] .= Float32.(M_trans .- M_rot_scale * [1.0, 1.0, 1.0] .+ [1.0, 1.0, 1.0])
        M_batch[4, 4, b] = 1.0f0
    end
    
    backend = try KernelAbstractions.get_backend(im_fixed.voxel_data) catch; KernelAbstractions.CPU() end
    device_M = backend isa KernelAbstractions.GPU ? CuArray(M_batch) : M_batch
    val_ext = (value_to_extrapolate == Nothing) ? 0.0f0 : Float32(value_to_extrapolate)
    resampled_flat = Utils.interpolate_fused_affine(im_moving.voxel_data, device_M, new_size, interpolator_enum, false, val_ext, (0.0, 0.0, 0.0))

    batch_size = size(im_moving.voxel_data, 4)
    new_data = reshape(resampled_flat, new_size..., batch_size)

    if eltype(im_moving.voxel_data) != Float32

        new_data = cast_to_array_b_type(new_data, im_moving.voxel_data)
    end

    return BatchedMedImage(
        voxel_data = new_data,
        origin = im_fixed.origin,
        spacing = im_fixed.spacing,
        direction = im_fixed.direction,
        image_type = im_moving.image_type,
        image_subtype = im_moving.image_subtype,
        patient_id = im_moving.patient_id,
        current_device = im_moving.current_device,
        date_of_saving = im_moving.date_of_saving,
        acquistion_time = im_moving.acquistion_time,
        study_uid = im_moving.study_uid,
        patient_uid = im_moving.patient_uid,
        series_uid = im_moving.series_uid,
        study_description = im_moving.study_description,
        legacy_file_name = im_moving.legacy_file_name,
        display_data = im_moving.display_data,
        clinical_data = im_moving.clinical_data,
        is_contrast_administered = im_moving.is_contrast_administered,
        metadata = im_moving.metadata
    )
end

"""
    resample_to_spacing(im::MedImage, new_spacing::Tuple{Float64,Float64,Float64}, interpolator_enum::Interpolator_enum, value_to_extrapolate=Nothing)::MedImage

Resample a `MedImage` to a new voxel spacing.

This function changes the resolution of the image by interpolating the voxel data onto
a new grid defined by `new_spacing`. The image dimensions are automatically adjusted
to keep the same physical extent (field of view).

# Arguments
- `im::MedImage`: The input image to resample.
- `new_spacing`: Target spacing in millimeters (Tuple of 3 Float64).
- `interpolator_enum`: Method used for interpolation (e.g., `Linear_en`).
- `value_to_extrapolate`: Value for points outside the source image.

# Returns
- `MedImage`: A new resampled image.

# Examples
```julia
# Upsample to 0.5mm isovolumetric spacing
julia> new_im = resample_to_spacing(im, (0.5, 0.5, 0.5), B_spline_en)
```
"""
function resample_to_spacing(im::MedImage, new_spacing::Tuple{Float64,Float64,Float64}, interpolator_enum::Interpolator_enum, value_to_extrapolate=Nothing)::MedImage
    old_spacing = im.spacing
    old_voxel_data = im.voxel_data
    
    is_4d = (ndims(old_voxel_data) == 4)
    spatial_old_size = is_4d ? size(old_voxel_data)[1:3] : size(old_voxel_data)
    
    # Calculate new spatial size
    new_spatial_size = Tuple{Int,Int,Int}(ceil.((spatial_old_size .* old_spacing) ./ new_spacing))
    
    if is_4d
        num_channels = size(old_voxel_data, 4)
        # Allocate new 4D array
        new_voxel_data = similar(old_voxel_data, new_spatial_size..., num_channels)
        for c in 1:num_channels
            channel_data = selectdim(old_voxel_data, 4, c)
            new_voxel_data[:, :, :, c] .= Utils.resample_kernel_launch(channel_data, old_spacing, new_spacing, new_spatial_size, interpolator_enum)
        end
    else
        new_voxel_data = Utils.resample_kernel_launch(old_voxel_data, old_spacing, new_spacing, new_spatial_size, interpolator_enum)
    end

    return Load_and_save.update_voxel_and_spatial_data(im, new_voxel_data, im.origin, new_spacing, im.direction)
end

"""
    resample_to_spacing(im::BatchedMedImage, new_spacing::Union{Tuple{Float64,Float64,Float64}, Vector{Tuple{Float64,Float64,Float64}}}, interpolator_enum::Interpolator_enum)::BatchedMedImage

Resamples a BatchedMedImage to a new voxel spacing.
Ensures consistent output dimensions across the batch.
"""
function resample_to_spacing(im::BatchedMedImage, new_spacing::Union{Tuple{Float64,Float64,Float64}, Vector{Tuple{Float64,Float64,Float64}}}, interpolator_enum::Interpolator_enum)::BatchedMedImage
    batch_size = size(im.voxel_data, 4)
    old_spatial_size = size(im.voxel_data)[1:3]

    target_spacings = (new_spacing isa Vector) ? new_spacing : [new_spacing for _ in 1:batch_size]

    # Calculate and verify consistent output size
    first_new_size = Tuple{Int,Int,Int}(ceil.((old_spatial_size .* im.spacing[1]) ./ target_spacings[1]))
    
    for b in 2:batch_size
        curr_new_size = Tuple{Int,Int,Int}(ceil.((old_spatial_size .* im.spacing[b]) ./ target_spacings[b]))
        if curr_new_size != first_new_size
            error("Inconsistent output sizes in batch: $first_new_size vs $curr_new_size. All batch items must have the same output size.")
        end
    end
    
    # In batched mode, we assume each batch item is 3D (X,Y,Z). 
    # If we need to support batched multichannel, that would be 5D. 
    # For now, we stick to the standard BatchedMedImage structure.

    # Use interpolate_my which already supports batches
    # We generate physical points in the new grid space
    n_points = prod(first_new_size)
    points_to_interpolate = zeros(Float32, 3, n_points, batch_size)
    
    base_indices = get_base_indicies_arr(first_new_size) # 3 x N
    
    for b in 1:batch_size
        nsp = target_spacings[b]
        osp = im.spacing[b]
        
        # Physical coordinates for each point in the new grid
        for i in 1:n_points
            points_to_interpolate[1, i, b] = (Float32(base_indices[1, i]) - 1.0f0) * Float32(nsp[1]) / Float32(osp[1]) + 1.0f0
            points_to_interpolate[2, i, b] = (Float32(base_indices[2, i]) - 1.0f0) * Float32(nsp[2]) / Float32(osp[2]) + 1.0f0
            points_to_interpolate[3, i, b] = (Float32(base_indices[3, i]) - 1.0f0) * Float32(nsp[3]) / Float32(osp[3]) + 1.0f0
        end
    end

    # Use (1,1,1) for spacing because points are already in index space
    spacing_arg = [(1.0, 1.0, 1.0) for _ in 1:batch_size]
    resampled_flat = interpolate_my(points_to_interpolate, im.voxel_data, spacing_arg, interpolator_enum, false, 0.0, true)
    new_data = reshape(resampled_flat, first_new_size[1], first_new_size[2], first_new_size[3], batch_size)

    return BatchedMedImage(
        voxel_data = new_data,
        origin = im.origin,
        spacing = target_spacings,
        direction = im.direction,
        image_type = im.image_type,
        image_subtype = im.image_subtype,
        patient_id = im.patient_id,
        current_device = im.current_device,
        date_of_saving = im.date_of_saving,
        acquistion_time = im.acquistion_time,
        study_uid = im.study_uid,
        patient_uid = im.patient_uid,
        series_uid = im.series_uid,
        study_description = im.study_description,
        legacy_file_name = im.legacy_file_name,
        display_data = im.display_data,
        clinical_data = im.clinical_data,
        is_contrast_administered = im.is_contrast_administered,
        metadata = im.metadata
    )
end

end#Resample_to_target
