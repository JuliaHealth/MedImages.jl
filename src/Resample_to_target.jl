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
    # Calculate affine matrices for the entire batch
    # M maps Target Index -> Source Index
    # I_mov = ( (I_fix - 1)*S_fix + O_fix - O_mov ) / S_mov + 1
    #       = I_fix * (S_fix/S_mov) + [ 1 - S_fix/S_mov + (O_fix - O_mov)/S_mov ]
    
    M_batch = zeros(Float32, 4, 4, batch_size)
    for b in 1:batch_size
        sf = im_fixed.spacing[b]
        sm = im_moving.spacing[b]
        of = im_fixed.origin[b]
        om = im_moving.origin[b]
        
        m11 = sf[1] / sm[1]
        m22 = sf[2] / sm[2]
        m33 = sf[3] / sm[3]
        
        m14 = 1.0f0 - m11 + (of[1] - om[1]) / sm[1]
        m24 = 1.0f0 - m22 + (of[2] - om[2]) / sm[2]
        m34 = 1.0f0 - m33 + (of[3] - om[3]) / sm[3]
        
        M_batch[1, 1, b] = m11; M_batch[2, 2, b] = m22; M_batch[3, 3, b] = m33
        M_batch[1, 4, b] = m14; M_batch[2, 4, b] = m24; M_batch[3, 4, b] = m34
        M_batch[4, 4, b] = 1.0f0
    end
    
    device_M = is_cuda_array(im_moving.voxel_data) ? CuArray(M_batch) : M_batch
    val_ext = (value_to_extrapolate == Nothing) ? 0.0f0 : Float32(value_to_extrapolate)
    new_data = interpolate_fused_affine(im_moving.voxel_data, device_M, new_size, interpolator_enum)
    
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

end#Resample_to_target
