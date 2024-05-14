include("./MedImage_data_struct.jl")
using Interpolations

"""
functions to change the metadata of a 3D image like change the orientation of the image
change spaciing to desired etc 
"""

"""
given a MedImage object and desired spacing (spacing) return the MedImage object with the new spacing

"""
function resample_to_spacing(im::Array{MedImage}, new_spacing::Tuple{Float64,Float64,Float64})::MedImage
  old_spacing = im.spacing
  old_size = size(im.voxel_data)
  new_size = Tuple{Int,Int,Int}((old_size .* old_spacing) ./ new_spacing)

  # Choose the interpolator
  interpolator = linear

  # Create the interpolation object
  itp = interpolate(im.voxel_data, interpolator)

  # Create the new voxel data
  new_voxel_data = Array{eltype(im.voxel_data)}(undef, new_size)
  for i in 1:new_size[1], j in 1:new_size[2], k in 1:new_size[3]
      new_voxel_data[i,j,k] = itp(i * old_spacing[1] / new_spacing[1], j * old_spacing[2] / new_spacing[2], k * old_spacing[3] / new_spacing[3])
  end

  # Create the new MedImage object
  new_im =update_voxel_and_spatial_data(im, new_voxel_data
  ,im.origin,im.spacing,im.direction)

  return new_im
end#resample_to_spacing

"""
given a MedImage object and desired orientation encoded as 3 letter string (like RAS or LPS) return the MedImage object with the new orientation
"""
function change_orientation(im::MedImage, new_orientation::String)::MedImage
    # Create a dictionary to map orientation strings to direction cosines
    orientation_dict = Dict(
        "RAS" => [1 0 0; 0 1 0; 0 0 1],
        "LPS" => [-1 0 0; 0 -1 0; 0 0 -1],
        "LAS" => [-1 0 0; 0 1 0; 0 0 1],
        "RSP" => [1 0 0; 0 -1 0; 0 0 -1],
        "LAI" => [-1 0 0; 0 0 -1; 0 1 0],
        "RAI" => [1 0 0; 0 0 -1; 0 1 0]
    )

    # Check if the new orientation is valid
    if !haskey(orientation_dict, new_orientation)
        error("Invalid orientation: $new_orientation")
    end

    # Get the direction cosines for the new orientation
    new_direction = orientation_dict[new_orientation]

    # Create a new MedImage with the new direction and the same voxel data, origin, and spacing
    new_im = update_voxel_and_spatial_data(im, im.voxel_data
    ,im.origin,im.spacing,new_direction)

    return new_im
  end#change_orientation



