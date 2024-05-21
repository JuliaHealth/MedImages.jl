include("./MedImage_data_struct.jl")
include("./Load_and_save.jl")
using Interpolations

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
   function resample_to_image(im_fixed::MedImage, im_moving::MedImage, interpolator_enum::Interpolator_enum)::MedImage
    # Create interpolation object based on the interpolator type
    itp = nothing


    check origin of both images as for example in case origin of the moving image is not in the fixed image we need to return zeros

    if interpolator_enum == Nearest_neighbour_en
        itp = interpolate(im_moving.voxel_data, BSpline(Constant()))
    elseif interpolator_enum == Linear_en
        itp = interpolate(im_moving.voxel_data, BSpline(Linear()))
    elseif interpolator_enum == B_spline_en
        itp = interpolate(im_moving.voxel_data, BSpline(Cubic(Line(OnGrid()))))
    end
    krowa we need to use scaled b spline interpolations from interpolators.jl or gridded ones

    # Create a new voxel_data array for the resampled image
    new_voxel_data = similar(im_fixed.voxel_data)

    krowa get direction from one and set it to other
    # Calculate the transformation from fixed image space to moving image space
    fixed_to_moving = inv(reshape(collect(im_moving.direction), (3, 3))) * collect(im_fixed.spacing ./ im_moving.spacing)

    # Create an array of indices
    indices = get_base_indicies_arr(size(im_fixed.voxel_data)) #CartesianIndices(im_fixed.voxel_data)

    # Transform the index to physical space in the fixed image
    fixed_physical = im_fixed.origin .+ ((indices .- 1).* collect(im_fixed.spacing))

    # Transform the physical space to index space in the moving image
    moving_index = ((fixed_physical .- collect(im_moving.origin)) .+ 1).*fixed_to_moving

    # Interpolate the moving image at the transformed index
    new_voxel_data = itp.(permutedims(moving_index,(1,2)))

    # Create a new MedImage with the resampled voxel_data and the same spatial metadata as the fixed image
    resampled_image = update_voxel_and_spatial_data(im_moving, new_voxel_data, im_fixed.origin, im_fixed.spacing, im_fixed.direction)

    return resampled_image
end


function load_image(path)
  """
  load image from path
  """
  # test_image_equality(p,p)

  medimage_instance_array = load_images(path)
  medimage_instance = medimage_instance_array[1]
  return medimage_instance
end#load_image



# debug_folder="/home/jakubmitura/projects/MedImage.jl/test_data/debug"
# p="/home/jakubmitura/projects/MedImage.jl/test_data/volume-0.nii.gz"



im_fixed=load_image("/home/jakubmitura/projects/MedImage.jl/test_data/pet_data/pat_2_sudy_0_2022-09-16_Standardized_Uptake_Value_body_weight.nii.gz")
im_moving=load_image("/home/jakubmitura/projects/MedImage.jl/test_data/pet_data/pat_2_sudy_1_2023-07-12_Standardized_Uptake_Value_body_weight.nii.gz")
resample_to_image(im_fixed, im_moving,Linear_en)



# """
# get 4 dimensional array of cartesian indicies of a 3 dimensional array
# thats size is passed as an argument dims
# """
# function get_base_indicies_arr(dims)    
#     indices = CartesianIndices(dims)
#     # indices=collect.(Tuple.(collect(indices)))
#     indices=Tuple.(collect(indices))
#     indices=collect(Iterators.flatten(indices))
#     indices=reshape(indices,(3,dims[1]*dims[2]*dims[3]))
#     indices=permutedims(indices,(1,2))
#     return indices
# end#get_base_indicies_arr



# indices=get_base_indicies_arr((4,4,4))
# origin=(2.0,2.0,2.0)
# # fixed_physical = im_fixed.origin .+ ((Tuple.(indices) .- 1).* reshape(collect(im_fixed.spacing), size(Tuple.(indices))))
# fixed_physical = origin .+ (indices)



# vv= zeros(3,3,3)
# indices = map(el->collect(el) ,collect(CartesianIndices(vv)))