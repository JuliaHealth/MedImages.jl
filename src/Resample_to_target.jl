include("./MedImage_data_struct.jl")
"""
given two MedImage objects and a Interpolator enum value return the moving MedImage object resampled to the fixed MedImage object
images should have the same orientation origin and spacing; their pixel arrays should have the same shape
It require multiple steps some idea of implementation is below
1) check origin of both images as for example in case origin of the moving image is not in the fixed image we need to return zeros
2) we should define a grid on the basis of locations of the voxels in the fixed image and interpolate voxels from the moving image to the grid using GridInterpolations
   in order to achieve it we need to use spatial metadata to get the correct locations of the voxels in the fixed and moving images
"""
function resample_to_image(im_fixed::Array{MedImage}, im_moving::Array{MedImage}, Interpolator::Interpolator)::Array{MedImage}

  nothing

end#scale_mi    
