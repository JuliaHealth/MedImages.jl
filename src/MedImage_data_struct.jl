using Pkg
Pkg.add(["Dictionaries"])
using Dictionaries







"""
Here we define necessary data structures for the project.
Main data structure is a MedImage object which is a 3D image with some metadata.

!!!! Currently implemented as Struct but will be better to use as metadata arrays
"""

#following struct can be expanded with all the relevant meta data mentioned within the readme.md of MedImage.jl
struct MedImage
  #pixel array data for nifti volume
  pixel_array #stores an array of (512 x 512 x 3)-> 3dimensional pixel data arrays
  
  #some important attributes related to medical imaging 
  direction
  spacing
  origin

  date_of_saving::String
  patient_id::String
  #other data for nifti header
  descrip::NTuple{80,UInt8}
  sizeof_hdr::Int32
  pixdim::NTuple{8, Float32}
  vox_offset::Float32
  #spatial metadata for nifti header
  qform_code::Int16
  sform_code::Int16
  quatern_b::Float32
  quatern_c::Float32
  quatern_d::Float32
  qoffset_x::Float32
  qoffset_y::Float32
  qoffset_z::Float32
  srow_x::NTuple{4, Float32}
  srow_y::NTuple{4, Float32}
  srow_z::NTuple{4, Float32}
end


#constructor function for MedImage
function MedImage(MedImage_struct_attributes::Dictionaries.Dictionary{String,Any})::MedImage
  return MedImage(
    get(MedImage_struct_attributes, "pixel_array", []),
    get(MedImage_struct_attributes, "direction", []),
    get(MedImage_struct_attributes, "spacing", []),
    get(MedImage_struct_attributes, "origin", []),
    get(MedImage_struct_attributes, "date_of_saving", ""),
    get(MedImage_struct_attributes, "patient_id", ""),
    get(MedImage_struct_attributes,"descrip",()),
    get(MedImage_struct_attributes,"sizeof_hdr"),
    get(MedImage_struct_attributes,"pixdim"),
    get(MedImage_struct_attributes,"vox_offset"),
    get(MedImage_struct_attributes,"qform_code"),
    get(MedImage_struct_attributes,"sform_code"),
    get(MedImage_struct_attributes,"quatern_b"),
    get(MedImage_struct_attributes,"quatern_c"),
    get(MedImage_struct_attributes,"quatern_d"),
    get(MedImage_struct_attributes,"qoffset_x",),
    get(MedImage_struct_attributes,"qoffset_y",),
    get(MedImage_struct_attributes,"qoffset_z"),
    get(MedImage_struct_attributes,"srow_x"),
    get(MedImage_struct_attributes,"srow_y"),
    get(MedImage_struct_attributes,"srow_z")
  
  )
end

function MedImage(value_data_array::Array{Any})::MedImage
  return MedImage(value_data_array...)
end

@enum Interpolator nearest_neighbour = 0 linear = 2 b_spline = 3

