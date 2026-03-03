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






#  te force solution get all direction combinetions put it in sitk and try all possible ways to permute and reverse axis to get the same result as sitk then save the result in json or sth and use; do the same with the origin
#     Additionally in similar manner save all directions in a form of a vector and associate it with 3 letter codes


"""
    change_orientation(im::MedImage, goal_orientation::Orientation_code)::MedImage
    change_orientation(im::MedImage, goal_orientation::String)::MedImage

Change the orientation of a `MedImage` to a target orientation (e.g., "RAS", "LPS").

This function permutes and/or reverses the image axes to match the desired `goal_orientation`.
It updates both the voxel data and the spatial metadata (origin, spacing, direction)
consistently.

# Arguments
- `im::MedImage`: The input image to reorient.
- `goal_orientation`: Target orientation as an `Orientation_code` enum or a string.

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
