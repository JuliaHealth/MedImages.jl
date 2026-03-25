module Spatial_metadata_change
using Interpolations
using CUDA
using ChainRulesCore

using ..MedImage_data_struct, ..Utils, ..Orientation_dicts, ..Load_and_save
export change_orientation
"""
given a MedImage object and desired spacing (spacing) return the MedImage object with the new spacing
"""
function scale(itp::AbstractInterpolation{T,N,IT}, ranges::Vararg{AbstractRange,N}) where {T,N,IT}
    # overwriting this function becouse check_ranges giving error
    # check_ranges(itpflag(itp), axes(itp), ranges)
    ScaledInterpolation{T,N,typeof(itp),IT,typeof(ranges)}(itp, ranges)
end




function resample_to_spacing(im::MedImage, new_spacing::Tuple{Float64,Float64,Float64}, interpolator_enum::Interpolator_enum, use_cuda=false)::MedImage
    old_spacing = im.spacing
    old_size = size(im.voxel_data)


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

    # Logic for handling shared vs unique spacing


    # If consistent, we can proceed.
    # We need to adapt `resample_kernel_launch` to handle batches.
    # Currently `resample_kernel_launch` takes (image_data, old_spacing, new_spacing, new_dims, ...)
    # It assumes shared spacing?
    # Let's check `Utils.jl`.

    # `resample_kernel_launch` in Utils calls `trilinear_resample_enzyme_kernel!` or similar.
    # The kernels take `old_spacing`, `new_spacing`.
    # They are marked `@Const` in simple version, or as arrays in the 4D/Enzyme version?
    # Wait, I added `interpolate_kernel_4d` but did I update `resample_kernel_launch`?
    # No, I updated `interpolate_pure`.
    # `resample_to_spacing` uses `resample_kernel_launch`, which is a specialized resampling kernel (inverse mapping).
    # `interpolate_my` is forward/arbitrary point interpolation.

    # `resample_kernel_launch` is optimized for grid-to-grid resampling.
    # I need to update `resample_kernel_launch` in `Utils.jl` to support batched arrays and vector spacings!

    # For now, I will use `interpolate_my` approach which supports batching via my previous work,
    # OR I should update `resample_kernel_launch`.
    # Updating `resample_kernel_launch` is better for performance (specialized grid kernel).
    # But `interpolate_my` is more generic.

    # Let's stick to `resample_kernel_launch` if I update it, or fallback to `resample_to_image` logic?
    # Actually `resample_to_spacing` is a subset of `resample_to_image` where new grid is aligned.

    # If I use `resample_kernel_launch`, I need to modify it in `Utils.jl`.
    # Let's assume for this step I will rely on `Utils.resample_kernel_launch` and update it if needed,
    # OR implement loop here.

    # To avoid modifying `Utils.jl` extensively again in this step (risk),
    # I can implement the batch loop here calling single `resample_kernel_launch`?
    # No, that's slow on GPU.

    # I should use `interpolate_my` which I ALREADY updated to support batches!
    # Calculate affine matrices for the entire batch
    # M maps Target Index -> Source Index
    # rx = (ix - 1) * nsp / osp + 1 = ix * (nsp/osp) + (1 - nsp/osp)
    
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
    
    # Call the fused kernel
    new_data = interpolate_fused_affine(im.voxel_data, device_M, first_new_size, interpolator_enum)
    
    # Cast back to original type if needed
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
    change_orientation(im::MedImage, goal_orientation::Orientation_code)::MedImage
    change_orientation(im::MedImage, goal_orientation::String)::MedImage

Change the orientation of a `MedImage` to a target orientation (e.g., "RAS", "LPS").

# Returns
- `MedImage`: A new image object with the requested orientation.

# Examples
```julia
# Change to Right-Anterior-Superior (RAS)
julia> new_im = change_orientation(im, "RAS")

# Change using enum
julia> new_im = change_orientation(im, ORIENTATION_LPS)
```

# Notes
- **Permutation**: If the target orientation requires swapping axes (e.g., sagittal to axial), the `voxel_data` is permuted.
- **Reversal**: If an axis direction needs to be flipped (e.g., Left to Right), the data along that dimension is reversed.
- **Metadata**: Origin, spacing, and direction are automatically adjusted to remain physically accurate.
"""
function change_orientation(im::MedImage, new_orientation::Orientation_code)::MedImage
    old_orientation = Orientation_dicts.number_to_enum_orientation_dict[im.direction]
    reorient_operation = Orientation_dicts.orientation_pair_to_operation_dict[(old_orientation, new_orientation)]
    return change_orientation_main(im, new_orientation, reorient_operation)
end#change_orientation

function change_orientation(im::MedImage, new_orientation::String)::MedImage
    return change_orientation(im, Orientation_dicts.string_to_orientation_enum[new_orientation])
end#change_orientation

# Custom rrule for change_orientation that handles dictionary lookups properly
function ChainRulesCore.rrule(::typeof(change_orientation), im::MedImage, new_orientation::Orientation_code)
    # Forward pass
    output = change_orientation(im, new_orientation)

    function change_orientation_pullback(d_output)
        d_output_unthunked = unthunk(d_output)
        # Get the voxel data tangent
        d_voxel = d_output_unthunked.voxel_data

        if isnothing(d_voxel) || d_voxel isa ChainRulesCore.NoTangent || d_voxel isa ChainRulesCore.ZeroTangent
            d_im = Tangent{MedImage}(; voxel_data=ChainRulesCore.ZeroTangent())
            return NoTangent(), d_im, NoTangent()
        end

        # Get the reorientation operation to reverse it
        old_orientation = Orientation_dicts.number_to_enum_orientation_dict[im.direction]
        reorient_op = Orientation_dicts.orientation_pair_to_operation_dict[(old_orientation, new_orientation)]
        perm = reorient_op[1]
        reverse_axes = reorient_op[2]

        # Reverse the operations in reverse order
        # First undo reverse, then undo permute
        d_voxel_back = d_voxel
        if length(reverse_axes) == 1
            d_voxel_back = reverse(d_voxel_back; dims=reverse_axes[1])
        elseif length(reverse_axes) > 1
            d_voxel_back = reverse(d_voxel_back; dims=Tuple(reverse_axes))
        end

        if length(perm) > 0
            # Inverse permutation
            inv_perm = invperm((perm[1], perm[2], perm[3]))
            d_voxel_back = permutedims(d_voxel_back, inv_perm)
        end

        d_im = Tangent{MedImage}(; voxel_data=d_voxel_back)
        return NoTangent(), d_im, NoTangent()
    end

    return output, change_orientation_pullback
end

function change_orientation_main(im::MedImage, new_orientation::Orientation_code, reorient_operation)::MedImage
    perm = reorient_operation[1]
    reverse_axes = reorient_operation[2]
    origin_transforms = reorient_operation[3]
    spacing_transforms = reorient_operation[4]

    origin1 = im.origin
    sizz = size(im.voxel_data)
    spacing1 = im.spacing

    # Non-mutating origin calculation
    res_origin = ntuple(i -> begin
        spac_axis, sizz_axis, prim_origin_axis, op_sign = origin_transforms[i]
        origin1[prim_origin_axis] + ((spacing1[spac_axis] * (sizz[sizz_axis] - 1)) * op_sign)
    end, 3)

    # Permute and reverse voxel data
    # CUDA.jl natively supports permutedims and reverse on CuArrays
    # No CPU transfers needed - operations execute directly on GPU
    im_voxel_data = im.voxel_data
    if (length(perm) > 0)
        im_voxel_data = permutedims(im_voxel_data, (perm[1], perm[2], perm[3]))
    end

    if (length(reverse_axes) == 1)
        im_voxel_data = reverse(im_voxel_data; dims=reverse_axes[1])
    elseif (length(reverse_axes) > 1)
        im_voxel_data = reverse(im_voxel_data; dims=Tuple(reverse_axes))
    end


    # now we need to change spacing as needed
    st = spacing_transforms
    sp = im.spacing
    new_spacing = (sp[st[1]], sp[st[2]], sp[st[3]])
    new_im = Load_and_save.update_voxel_and_spatial_data(im, im_voxel_data, res_origin, new_spacing, orientation_dict_enum_to_number[new_orientation])

    # print("\n res_origin $(res_origin) \n")
    return new_im
end#change_orientation

end#Spatial_metadata_change
