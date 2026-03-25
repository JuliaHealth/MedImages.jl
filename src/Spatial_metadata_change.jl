module Spatial_metadata_change

using ..MedImage_data_struct
using ..MedImage_data_struct: Nearest_neighbour_en, Linear_en, B_spline_en, Interpolator_enum
using ..Load_and_save
using ..Utils: resample_kernel_launch, interpolate_fused_affine, is_cuda_array, cast_to_array_b_type
using ..Orientation_dicts
using ..Orientation_dicts: Orientation_code, orientation_dict_enum_to_number
using CUDA
using ChainRulesCore
using Statistics

export resample_to_spacing, change_orientation, resample_to_image

"""
    resample_to_spacing(im::MedImage, new_spacing::Tuple{Float64,Float64,Float64}, interpolator_enum::Interpolator_enum, use_cuda=false)::MedImage

Resample a `MedImage` to a new target spacing (voxel size).
"""
function resample_to_spacing(im::MedImage, new_spacing::Tuple{Float64,Float64,Float64}, interpolator_enum::Interpolator_enum, use_cuda=false)::MedImage
    old_spacing = im.spacing
    old_size = size(im.voxel_data)

    new_size = ntuple(i -> Int(round(old_size[i] * old_spacing[i] / new_spacing[i])), 3)

    # Julia array dims map as: dim1 -> X, dim2 -> Y, dim3 -> Z
    # Spacing is already in (x,y,z) order, so no reversal needed
    # Use optimized kernel resampling
    new_voxel_data = resample_kernel_launch(im.voxel_data, old_spacing, new_spacing, new_size, interpolator_enum)

    new_im = Load_and_save.update_voxel_and_spatial_data(im, new_voxel_data, im.origin, new_spacing, im.direction)

    return new_im
end#resample_to_spacing

function resample_to_spacing(im::BatchedMedImage, new_spacing::Union{Tuple{Float64,Float64,Float64}, Vector{Tuple{Float64,Float64,Float64}}}, interpolator_enum::Interpolator_enum)::BatchedMedImage
    batch_size = size(im.voxel_data, 4)
    old_size = size(im.voxel_data)[1:3]

    # Handle both uniform spacing for the whole batch and unique per-image spacing
    target_spacings = if new_spacing isa Vector
        if length(new_spacing) != batch_size
            error("Vector spacing length must match batch size")
        end
        new_spacing
    else
        fill(new_spacing, batch_size)
    end

    # All images in a batch MUST have the same output grid size after resampling
    # We use the first image's spacing ratio to determine the new grid size
    nsp1 = target_spacings[1]
    osp1 = im.spacing[1]
    first_new_size = ntuple(i -> Int(round(old_size[i] * osp1[i] / nsp1[i])), 3)

    # Calculate affine matrices for the entire batch
    # M maps Target Index -> Source Index
    M_batch = zeros(Float32, 4, 4, batch_size)
    for b in 1:batch_size
        nsp = target_spacings[b]
        osp = im.spacing[b]
        
        m11 = nsp[1] / osp[1]
        m22 = nsp[2] / osp[2]
        m33 = nsp[3] / osp[3]
        
        M_batch[1, 1, b] = m11; M_batch[2, 2, b] = m22; M_batch[3, 3, b] = m33
        M_batch[1, 4, b] = 1.0f0 - m11
        M_batch[2, 4, b] = 1.0f0 - m22
        M_batch[3, 4, b] = 1.0f0 - m33
        M_batch[4, 4, b] = 1.0f0
    end
    
    device_M = is_cuda_array(im.voxel_data) ? CuArray(M_batch) : M_batch
    
    # Call the high-performance fused kernel
    new_data = interpolate_fused_affine(im.voxel_data, device_M, first_new_size, interpolator_enum)
    
    if eltype(im.voxel_data) != Float32
        new_data = cast_to_array_b_type(new_data, im.voxel_data)
    end

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

"""
    resample_to_image(fixed_image::MedImage, moving_image::MedImage, interpolator_enum::Interpolator_enum)::MedImage

Resample `moving_image` onto the spatial grid of `fixed_image`.
"""
function resample_to_image(fixed_image::MedImage, moving_image::MedImage, interpolator_enum::Interpolator_enum)::MedImage
    # Calculate relative affine transformation between images
    # M_relative = inv(M_moving) * M_fixed
    M_fixed = Basic_transformations.computeIndexToPhysicalPointMatrices_Julia(fixed_image)
    M_moving = Basic_transformations.computeIndexToPhysicalPointMatrices_Julia(moving_image)
    
    # Pre-allocate 4x4 matrix
    R = Matrix{Float64}(undef, 4, 4)
    R[1:3, 1:3] = inv(M_moving) * M_fixed
    R[1:3, 4] = inv(M_moving) * (collect(fixed_image.origin) .- collect(moving_image.origin))
    R[4, 1:3] .= 0.0
    R[4, 4] = 1.0
    
    new_voxel_data = Basic_transformations.affine_transform_mi(moving_image, R, interpolator_enum; output_size=size(fixed_image.voxel_data))
    
    return Load_and_save.update_voxel_and_spatial_data(
        moving_image, 
        new_voxel_data, 
        fixed_image.origin, 
        fixed_image.spacing, 
        fixed_image.direction
    )
end

"""
    change_orientation(im::MedImage, goal_orientation::Orientation_code)::MedImage
    change_orientation(im::MedImage, goal_orientation::String)::MedImage

Change the orientation of a `MedImage` to a target orientation (e.g., "RAS", "LPS").
"""
function change_orientation(im::MedImage, new_orientation::Orientation_code)::MedImage
    old_orientation = Orientation_dicts.number_to_enum_orientation_dict[im.direction]
    reorient_operation = Orientation_dicts.orientation_pair_to_operation_dict[(old_orientation, new_orientation)]
    return change_orientation_main(im, new_orientation, reorient_operation)
end

function change_orientation(im::MedImage, new_orientation::String)::MedImage
    return change_orientation(im, Orientation_dicts.string_to_orientation_enum[new_orientation])
end

# Custom rrule for change_orientation
function ChainRulesCore.rrule(::typeof(change_orientation), im::MedImage, new_orientation::Orientation_code)
    output = change_orientation(im, new_orientation)
    function change_orientation_pullback(d_output)
        d_output_unthunked = unthunk(d_output)
        d_voxel = d_output_unthunked.voxel_data
        if isnothing(d_voxel) || d_voxel isa ChainRulesCore.NoTangent || d_voxel isa ChainRulesCore.ZeroTangent
            return NoTangent(), Tangent{MedImage}(; voxel_data=ChainRulesCore.ZeroTangent()), NoTangent()
        end
        old_orientation = Orientation_dicts.number_to_enum_orientation_dict[im.direction]
        reorient_op = Orientation_dicts.orientation_pair_to_operation_dict[(old_orientation, new_orientation)]
        perm = reorient_op[1]; reverse_axes = reorient_op[2]
        d_voxel_back = d_voxel
        if length(reverse_axes) == 1
            d_voxel_back = reverse(d_voxel_back; dims=reverse_axes[1])
        elseif length(reverse_axes) > 1
            d_voxel_back = reverse(d_voxel_back; dims=Tuple(reverse_axes))
        end
        if length(perm) > 0
            d_voxel_back = permutedims(d_voxel_back, invperm((perm[1], perm[2], perm[3])))
        end
        return NoTangent(), Tangent{MedImage}(; voxel_data=d_voxel_back), NoTangent()
    end
    return output, change_orientation_pullback
end

function change_orientation_main(im::MedImage, new_orientation::Orientation_code, reorient_operation)::MedImage
    perm = reorient_operation[1]; reverse_axes = reorient_op = reorient_operation[2]
    origin_transforms = reorient_operation[3]; spacing_transforms = reorient_operation[4]
    origin1 = im.origin; sizz = size(im.voxel_data); spacing1 = im.spacing

    res_origin = ntuple(i -> begin
        spac_axis, sizz_axis, prim_origin_axis, op_sign = origin_transforms[i]
        origin1[prim_origin_axis] + ((spacing1[spac_axis] * (sizz[sizz_axis] - 1)) * op_sign)
    end, 3)

    im_voxel_data = im.voxel_data
    if length(perm) > 0; im_voxel_data = permutedims(im_voxel_data, (perm[1], perm[2], perm[3])); end
    if length(reverse_axes) == 1; im_voxel_data = reverse(im_voxel_data; dims=reverse_axes[1])
    elseif length(reverse_axes) > 1; im_voxel_data = reverse(im_voxel_data; dims=Tuple(reverse_axes)); end

    st = spacing_transforms; sp = im.spacing
    new_spacing = (sp[st[1]], sp[st[2]], sp[st[3]])
    return Load_and_save.update_voxel_and_spatial_data(im, im_voxel_data, res_origin, new_spacing, orientation_dict_enum_to_number[new_orientation])
end

end#module
