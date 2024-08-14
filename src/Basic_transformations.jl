include("MedImage_data_struct.jl")
using CoordinateTransformations, Interpolations, StaticArrays, LinearAlgebra, Rotations, Dictionaries
using LinearAlgebra

"""
module implementing basic transformations on 3D images 
like translation, rotation, scaling, and cropping both input and output are MedImage objects
"""

"""
given a MedImage object will rotate it by angle (angle) around axis (rotate_axis)
the center of rotation is set to be center of the image
It modifies both pixel array and not metadata
we are setting Interpolator by using Interpolator enum
return the rotated MedImage object 
"""

#Work in progress

"""
given a MedImage object and a Tuples that contains the location of the begining of the crop (crop_beg) and the size of the crop (crop_size) crops image
It modifies both pixel array and metadata
we are setting Interpolator by using Interpolator enum (in basic implementation it will not be used)
return the cropped MedImage object 
"""
function crop_mi(im::MedImage, crop_beg::Tuple{Int64,Int64,Int64}, crop_size::Tuple{Int64,Int64,Int64}, Interpolator::Interpolator_enum)::MedImage

    # Create a view of the original voxel_data array with the specified crop
    cropped_voxel_data = @view im.voxel_data[crop_beg[1]:(crop_beg[1]+crop_size[1]-1), crop_beg[2]:(crop_beg[2]+crop_size[2]-1), crop_beg[3]:(crop_beg[3]+crop_size[3]-1)]

    # Adjust the origin according to the crop beginning coordinates
    cropped_origin = im.origin .+ (im.spacing .* crop_beg)

    # Create a new MedImage with the cropped voxel_data and adjusted origin
    cropped_im = update_voxel_and_spatial_data(im, cropped_voxel_data
    ,cropped_origin,im.spacing,im.direction)

    return cropped_im

end#  

#Work in progress - documentation not finished
""" 
given a MedImage object and a Tuples that contains the information on how many voxels to add in each axis (pad_beg) and on the end of the axis (pad_end)
we are performing padding by adding voxels with value pad_val
It modifies both pixel array and metadata
we are setting Interpolator by using Interpolator enum (in basic implementation it will not be used)
return the cropped MedImage object or cropped pixel array and origin
"""

function pad_mi(image::Union{MedImage,Array{Float32, 3}}, pad_beg::Tuple{Int64, Int64, Int64}, pad_end::Tuple{Int64, Int64, Int64}, pad_val)
  im = union_check(image)
  pad_beg_x = fill(pad_val, (pad_beg[1], size(im, 2), size(im, 3)))
  pad_end_x = fill(pad_val, (pad_end[1], size(im, 2), size(im, 3)))
  padded_voxel_data = vcat(pad_beg_x, im, pad_end_x)
  
  pad_beg_y = fill(pad_val, (size(padded_voxel_data, 1), pad_beg[2], size(im, 3)))
  pad_end_y = fill(pad_val, (size(padded_voxel_data, 1), pad_end[2], size(im, 3)))
  padded_voxel_data = hcat(pad_beg_y, padded_voxel_data, pad_end_y)


  pad_beg_z = fill(pad_val, (size(padded_voxel_data, 1), size(padded_voxel_data, 2), pad_beg[3]))
  pad_end_z = fill(pad_val, (size(padded_voxel_data, 1), size(padded_voxel_data, 2), pad_end[3]))
  padded_voxel_data = cat(pad_beg_z, padded_voxel_data, pad_end_z, dims=3)

  padded_origin = im.origin .- (im.spacing .* pad_beg)
  if isa(image, MedImage)
      return padded_im = update_voxel_and_spatial_data(im, padded_voxel_data, padded_origin, im.spacing, im.direction)
  else
      return padded_im, padded_origin
  end
end



function pad_mi_stretch(im::Array{Float32, 3}, strech::Tuple{Int64, Int64, Int64})
  im = union_check(im)

  pad_beg_x = im[1:1, :, :] 
  pad_end_x = im[end:end, :, :]
  padded_voxel_data = im
  for i in 1:strech[1]
      padded_voxel_data = cat(pad_beg_x, padded_voxel_data, pad_end_x, dims=1)
  end
  pad_beg_y = padded_voxel_data[:, 1:1, :] 
  pad_end_y = padded_voxel_data[:, end:end, :]
  for i in 1:strech[2]
      padded_voxel_data = cat(pad_beg_y, padded_voxel_data, pad_end_y, dims=2)
  end
  pad_beg_z = padded_voxel_data[:, :, 1:1]
  pad_end_z = padded_voxel_data[:, :, end:end]
  for i in 1:strech[3]
      padded_voxel_data = cat(pad_beg_z, padded_voxel_data, pad_end_z, dims=3)
  end
  return padded_voxel_data
end

"""
given a MedImage object translation value (translate_by) and axis (translate_in_axis) in witch to translate the image return translated image
It is diffrent from pad by the fact that it changes only the metadata of the image do not influence pixel array
we are setting Interpolator by using Interpolator enum (in basic implementation it will not be used)
return the translated MedImage object
"""
function translate_mi(im::MedImage, translate_by::Int64, translate_in_axis::Int64, Interpolator::Interpolator_enum)::MedImage

    # Create a copy of the origin
    translated_origin = copy(im.origin)

    # Modify the origin according to the translation value and axis
    translated_origin[translate_in_axis] += translate_by * im.spacing[translate_in_axis]

    # Create a new MedImage with the translated origin and the original voxel_data
    translated_im = update_voxel_and_spatial_data(im, im.voxel_data
    ,translated_origin,im.spacing,im.direction)

    return translated_im

end  
