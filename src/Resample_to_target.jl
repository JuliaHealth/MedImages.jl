include("./MedImage_data_struct.jl")
include("./Load_and_save.jl")
include("./Spatial_metadata_change.jl")
using Interpolations
using Statistics


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

"""
`resample_to_image(im_fixed::MedImage, im_moving::MedImage, interpolator_enum, value_to_extrapolate=Nothing)::MedImage`

Resamples the `im_moving` image to align with the spatial properties of `im_fixed`, using the specified interpolation method and handling differences in origin, spacing, and orientation.
This function is pivotal in medical imaging for aligning images from different modalities or different sessions.

# Arguments
- `im_fixed`: The reference image with the target spatial characteristics.
- `im_moving`: The image to be resampled.
- `interpolator_enum`: The interpolation method used (Nearest neighbor, Linear, B-spline).
- `value_to_extrapolate`: Optional; specifies the value used for extrapolation outside the original image boundaries.

# Returns
- `MedImage`: The resampled `im_moving` image adjusted to the spatial specifications of `im_fixed`.

# Details
This function first aligns the orientation of `im_moving` to match `im_fixed`, calculates the necessary transformations based on spatial metadata (origin, spacing), and then interpolates the moving image to conform to the fixed image's grid. Extrapolation handles any out-of-bound data requests.
"""
function resample_to_image(im_fixed::MedImage, im_moving::MedImage, interpolator_enum=Linear_en, value_to_extrapolate=Nothing)::MedImage

    if(value_to_extrapolate==Nothing)
        value_to_extrapolate= extrapolate_corner_median(im_fixed)
    end

    
    # get direction from one and set it to other
    im_moving=change_orientation(im_moving, number_to_enum_orientation_dict[im_fixed.direction])

    # Calculate the transformation from moving image space to fixed image space
    old_spacing = im_moving.spacing
    new_spacing=im_fixed.spacing
    new_size = size(im_fixed.voxel_data)
    points_to_interpolate = get_base_indicies_arr(new_size)

    points_to_interpolate=points_to_interpolate.-1
    points_to_interpolate=points_to_interpolate.*new_spacing
    points_to_interpolate=points_to_interpolate.+1

    #adding diffrence in origin we act as if moving image has origin 0.0,0.0,0.0 - needed for interpolation
    origin_diff=(collect(im_fixed.origin)-collect(im_moving.origin))
    points_to_interpolate=points_to_interpolate.+origin_diff





    interpolated_points=interpolate_my(points_to_interpolate, im_moving.voxel_data, old_spacing, interpolator_enum, false, value_to_extrapolate)

    new_voxel_data=reshape(interpolated_points, (new_size[1], new_size[2], new_size[3]))
    # new_voxel_data=cast_to_array_b_type(new_voxel_data,im_fixed.voxel_data)


    new_im =update_voxel_and_spatial_data(im_moving, new_voxel_data, im_fixed.origin, new_spacing, im_fixed.direction)


    return new_im
end