using Pkg
# Pkg.add(["Dictionaries"])
using Dictionaries
include("./Nifti_image_struct.jl")
"""
Here we define necessary data structures for the project.
Main data structure is a MedImage object which is a 3D image with some metadata.

!!!! Currently implemented as Struct but will be better to use as metadata arrays
"""




"""
Defining image type enum
"""
@enum Image_type begin
  MRI
  PET
  CT
end


"""
Defining subimage type enum
"""
@enum Image_subtype begin
  subtypes
end


"""
Definition for standardised MedImage Struct
"""
#following struct can be expanded with all the relevant meta data mentioned within the readme.md of MedImage.jl
#struct for now, will switch to MetaArrays when it has GPU support
struct MedImage
  voxel_data::Array{Any}#mutlidimensional array (512,512,3)

  origin
  spacing
  direction #direction cosines for orientation
  spatial_metadata::Dictionaries.Dictionary #dictionary with properties for spacing, offset from spacing,orientation, origin, direction

  image_type::Image_type#enum defining the type of the image
  image_subtype::Image_subtype #enum defining the subtype of the image
  voxel_datatype #type of the voxel data stored
  date_of_saving #date of saving of the relevant imaging data file
  acquistion_time #time at which the data acquisition for the image took place
  patient_id #the id of the patient in the data file
  current_device::String# CPU or GPU , preferrably GPU
  study_uid
  patient_uid
  series_uid
  study_description
  legacy_file_name::String#original file name
  display_data #color values for the data such as RGB or gray
  clinical_data::Dictionary#dictionary with age , gender data of the patient
  is_contrast_administered::Bool #bool, any substance for visibility enhancement given during imaging procedure?
  metadata::Dictionary #dictionary for any other relevant metadata from individual data file
end
#constructor function for MedImage
function MedImage(MedImage_struct_attribute_values::Array{Any})::MedImage
  return MedImage(MedImage_struct_attribute_values...)
end








"""
Definitions of basic interpolators
"""
@enum Interpolator nearest_neighbour = 0 linear = 2 b_spline = 3

"""
Indicating do we want to change underlying pixel array spatial metadata or both
"""
@enum Mode_mi pixel_array_mode = 0 spat_metadata_mode = 2 all_mode = 3

