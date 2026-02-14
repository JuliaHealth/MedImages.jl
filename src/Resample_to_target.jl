module Resample_to_target
using Interpolations
using Statistics
using CUDA
using ChainRulesCore

using ..MedImage_data_struct, ..Utils, ..Orientation_dicts, ..Spatial_metadata_change, ..Load_and_save
export resample_to_image, scale

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
given two MedImage objects and a Interpolator enum value return the moving MedImage object resampled to the fixed MedImage object
images should have the same orientation origin and spacing; their pixel arrays should have the same shape
It require multiple steps some idea of implementation is below
1) check origin of both images as for example in case origin of the moving image is not in the fixed image we need to return zeros
2) we should define a grid on the basis of locations of the voxels in the fixed image and interpolate voxels from the moving image to the grid using for example GridInterpolations
"""
function resample_to_image(im_fixed::MedImage, im_moving::MedImage, interpolator_enum::Interpolator_enum, value_to_extrapolate=Nothing)::MedImage

    if (value_to_extrapolate == Nothing)
        # Use helper function marked as non-differentiable
        value_to_extrapolate = compute_extrapolate_value(im_fixed.voxel_data)
    end


    # get direction from one and set it to other
    im_moving = Spatial_metadata_change.change_orientation(im_moving, Orientation_dicts.number_to_enum_orientation_dict[im_fixed.direction])

    # Calculate the transformation from moving image space to fixed image space
    old_spacing = im_moving.spacing
    new_spacing = im_fixed.spacing
    new_size = size(im_fixed.voxel_data)
    points_to_interpolate = get_base_indicies_arr(new_size)

    points_to_interpolate = points_to_interpolate .- 1
    points_to_interpolate = points_to_interpolate .* new_spacing
    points_to_interpolate = points_to_interpolate .+ 1

    #adding diffrence in origin we act as if moving image has origin 0.0,0.0,0.0 - needed for interpolation
    origin_diff = (collect(im_fixed.origin) - collect(im_moving.origin))
    points_to_interpolate = points_to_interpolate .+ origin_diff





    interpolated_points = interpolate_my(points_to_interpolate, im_moving.voxel_data, old_spacing, interpolator_enum, false, value_to_extrapolate)

    new_voxel_data = reshape(interpolated_points, (new_size[1], new_size[2], new_size[3]))
    # new_voxel_data=cast_to_array_b_type(new_voxel_data,im_fixed.voxel_data)


    new_im = Load_and_save.update_voxel_and_spatial_data(im_moving, new_voxel_data, im_fixed.origin, new_spacing, im_fixed.direction)


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
    n_points = prod(new_size)
    points_to_interpolate = zeros(Float64, 3, n_points, batch_size)

    # Base indices are same for all (0-based)
    base_indices = get_base_indicies_arr(new_size) # 3 x N

    for b in 1:batch_size
        # For each batch, transform indices to physical space of fixed image
        # P_fixed = (Idx - 1) * Spacing_fixed + Origin_fixed

        # Then map to "relative" physical space of moving image assuming moving origin is 0?
        # Original logic:
        # points = points .- 1
        # points = points .* new_spacing
        # points = points .+ 1  (index -> physical offset from origin 0?)
        # points = points .+ origin_diff

        # Essentially: target_phys = index_to_phys(fixed)
        # interpolate_my takes physical points relative to moving image origin?
        # No, interpolate_my:
        # real_x = (point - 1) / spacing + 1
        # It assumes point is 1-based physical coordinate relative to origin?
        # Wait, Utils.jl `interpolate_kernel`:
        # real_x = (shared_arr[index_local, 1] - 1.0f0) / Float32(spacing[1]) + 1.0f0
        # This converts "physical-like" coordinate back to index.
        # If input point is P, index is (P-1)/S + 1.
        # This implies P = (I-1)*S + 1.
        # This is 1-based index converted to physical distance if origin was 1?
        # Or origin 0? If I=1 -> P=1.

        # Let's trace `resample_to_image` logic:
        # points = (indices - 1) * new_spacing + 1 + (fixed.origin - moving.origin)
        # This point P is passed to `interpolate_my`.
        # Inside `interpolate_my`:
        # real_x = (P - 1)/old_spacing + 1

        # Let's substitute:
        # P = (I_fix - 1)*S_fix + 1 + O_fix - O_mov
        # I_mov = ( (I_fix - 1)*S_fix + 1 + O_fix - O_mov - 1 ) / S_mov + 1
        #       = ( (I_fix - 1)*S_fix + O_fix - O_mov ) / S_mov + 1
        # This matches the standard mapping:
        # P_phys = (I_fix - 1)*S_fix + O_fix
        # I_mov = (P_phys - O_mov) / S_mov + 1

        # So yes, we need to construct P for each batch b.

        sp_fixed = im_fixed.spacing[b]
        origin_diff = im_fixed.origin[b] .- im_moving.origin[b]

        for i in 1:n_points
            # base_indices is 3xN
            # 1-based index from base_indices
            ix = base_indices[1, i]
            iy = base_indices[2, i]
            iz = base_indices[3, i]

            # Apply formula: (I-1)*S_new + 1 + diff
            px = (ix - 1) * sp_fixed[1] + 1 + origin_diff[1]
            py = (iy - 1) * sp_fixed[2] + 1 + origin_diff[2]
            pz = (iz - 1) * sp_fixed[3] + 1 + origin_diff[3]

            points_to_interpolate[1, i, b] = px
            points_to_interpolate[2, i, b] = py
            points_to_interpolate[3, i, b] = pz
        end
    end

    spacing_arg = im_moving.spacing # Vector of tuples

    val_ext = (value_to_extrapolate == Nothing) ? 0.0 : value_to_extrapolate

    resampled_flat = interpolate_my(points_to_interpolate, im_moving.voxel_data, spacing_arg, interpolator_enum, false, Float64(val_ext), true)

    new_data = reshape(resampled_flat, new_size[1], new_size[2], new_size[3], batch_size)

    return BatchedMedImage(
        voxel_data = new_data,
        origin = im_fixed.origin,
        spacing = im_fixed.spacing,
        direction = im_fixed.direction,
        image_type = im_moving.image_type,
        image_subtype = im_moving.image_subtype,
        patient_id = im_moving.patient_id,
        current_device = im_moving.current_device,
        # ... other fields copied or adapted
    )
end

end#Resample_to_target
