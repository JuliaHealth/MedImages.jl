include("./MedImage_data_struct.jl")
include("./Load_and_save.jl")
"""
given two MedImage objects and a Interpolator enum value return the moving MedImage object resampled to the fixed MedImage object
images should have the same orientation origin and spacing; their pixel arrays should have the same shape
It require multiple steps some idea of implementation is below
1) check origin of both images as for example in case origin of the moving image is not in the fixed image we need to return zeros
2) we should define a grid on the basis of locations of the voxels in the fixed image and interpolate voxels from the moving image to the grid using for example GridInterpolations
   in order to achieve it we need to use spatial metadata to get the correct locations of the voxels in the fixed and moving images

   This function first checks if the origin of the moving image is within the fixed image. If not, it returns a new MedImage with the same spatial metadata as the fixed image and a voxel data array filled with zeros.

    Then, it defines a grid for the fixed image based on its origin, spacing, and the size of its voxel data array.
    
    Next, it creates an interpolation object for the moving image using the Interpolations.jl package. The type of interpolation is determined by the interpolator argument.
    
    Finally, it resamples the moving image to the fixed image grid and returns a new MedImage with the resampled voxel data and the same spatial metadata as the fixed image.

   """
function resample_to_image(im_fixed::MedImage, im_moving::MedImage
                          ,Interpolator::Interpolator)::MedImage

    # Check if the origin of the moving image is in the fixed image
    if im_fixed.origin != im_moving.origin
      resampled_voxel_data =  zeros(size(im_fixed.voxel_data))

      return update_voxel_and_spatial_data(im_moving, resampled_voxel_data
    ,im_fixed.origin,im_fixed.spacing,im_fixed.direction)
    end
    print("aaaaaaaaaa")

    # Define the grid for the fixed image
    grid = Tuple((im_fixed.origin[i]:im_fixed.spacing[i]:(im_fixed.origin[i] + im_fixed.spacing[i] * size(im_fixed.voxel_data, i)) for i in 1:ndims(im_fixed.voxel_data)))

    # Create the interpolation object for the moving image
    itp = nothing
    if interpolator == nearest_neighbour
        itp = interpolate(im_moving.voxel_data, BSpline(Constant()))
    elseif interpolator == linear
        itp = interpolate(im_moving.voxel_data, BSpline(Linear()))
    elseif interpolator == b_spline
        itp = interpolate(im_moving.voxel_data, BSpline(Cubic(Line(OnGrid()))))
    end

    # Resample the moving image to the fixed image grid
    resampled_voxel_data = [itp((i .- im_moving.origin) ./ im_moving.spacing) for i in grid]


    # Create a new MedImage with the interpolated data and the same spatial metadata as the fixed image
    return update_voxel_and_spatial_data(im_moving, resampled_voxel_data
    ,im_fixed.origin,im_fixed.spacing,im_fixed.direction)



end#resample_to_image    




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
resample_to_image(im_fixed, im_moving,linear)


