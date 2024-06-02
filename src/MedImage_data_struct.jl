using Dates
# Pkg.add(["Dictionaries"])
using Dictionaries
using Parameters
# using UUIDs
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
  MRI_type
  PET_type
  CT_type
end

@enum current_device_enum begin
  CPU_current_device
  GPU_current_device

end
"""
Defining subimage type enum
"""
@enum Image_subtype begin
  CT_subtype
  ADC_subtype
  DWI_subtype
  T1_subtype
  T2_subtype
  FLAIR_subtype
  FDG_subtype
  PSMA_subtype
end


"""
Definition for standardised MedImage Struct
"""
#following struct can be expanded with all the relevant meta data mentioned within the readme.md of MedImage.jl
#struct for now, will switch to MetaArrays when it has GPU support
@with_kw struct MedImage
  voxel_data #mutlidimensional array (512,512,3)
  origin::Tuple{Float64,Float64,Float64}
  spacing::Tuple{Float64,Float64,Float64}#spacing between the voxels
  direction::NTuple{9, Float64} #direction cosines for orientation
  image_type::Image_type#enum defining the type of the image
  image_subtype::Image_subtype #enum defining the subtype of the image
  date_of_saving::DateTime = Dates.today()
  acquistion_time::DateTime = Dates.now()
  patient_id::String #the id of the patient in the data file
  current_device::current_device_enum = CPU_current_device# CPU or GPU , preferrably GPU
  study_uid::String=string(UUIDs.uuid4())#unique identifier for the study
  patient_uid::String=string(UUIDs.uuid4())#unique identifier for the patient
  series_uid::String=string(UUIDs.uuid4())#unique identifier for the series
  study_description::String=" "
  legacy_file_name::String=" "#original file name
  display_data::Dict{Any, Any}=Dict() #color values for the data such as RGB or gray
  clinical_data::Dict{Any, Any}=Dict()#dictionary with age , gender data of the patient
  is_contrast_administered::Bool=false #bool, any substance for visibility enhancement given during imaging procedure?
  metadata::Dict{Any, Any}=Dict() #dictionary for any other relevant metadata from individual data file
end


#constructor function for MedImage
# function MedImage(MedImage_struct_attribute_values::Array{Any})::MedImage
#   return MedImage(MedImage_struct_attribute_values...)
# end








"""
Definitions of basic interpolators
"""
@enum Interpolator_enum Nearest_neighbour_en Linear_en B_spline_en

"""
Indicating do we want to change underlying pixel array spatial metadata or both
"""
@enum Mode_mi pixel_array_mode = 0 spat_metadata_mode = 2 all_mode = 3


################## orientation
"""
enums based on https://github.com/InsightSoftwareConsortium/ITK/blob/311b7060ef39e371f3cd209ec135284ff5fde735/Modules/Core/Common/include/itkSpatialOrientation.h#L88
"""
@enum CoordinateTerms begin
    ITK_COORDINATE_UNKNOWN = 0
    ITK_COORDINATE_Right = 2
    ITK_COORDINATE_Left = 3
    ITK_COORDINATE_Posterior = 4
    ITK_COORDINATE_Anterior = 5
    ITK_COORDINATE_Inferior = 8
    ITK_COORDINATE_Superior = 9
end

@enum CoordinateMajornessTerms begin
    PrimaryMinor = 0
    SecondaryMinor = 8
    TertiaryMinor = 16
end

@enum Orientation_code begin
    # ORIENTATION_RIP
    # ORIENTATION_LIP
    # ORIENTATION_RSP
    # ORIENTATION_LSP
    # ORIENTATION_RIA
    # ORIENTATION_LIA
    # ORIENTATION_RSA
    # ORIENTATION_LSA
    # ORIENTATION_IRP
    # ORIENTATION_ILP
    # ORIENTATION_SRP
    # ORIENTATION_SLP
    # ORIENTATION_IRA
    # ORIENTATION_ILA
    # ORIENTATION_SRA
    # ORIENTATION_SLA
    ORIENTATION_RPI
    ORIENTATION_LPI
    ORIENTATION_RAI
    ORIENTATION_LAI
    ORIENTATION_RPS
    ORIENTATION_LPS
    ORIENTATION_RAS
    ORIENTATION_LAS
    # ORIENTATION_PRI
    # ORIENTATION_PLI
    # ORIENTATION_ARI
    # ORIENTATION_ALI
    # ORIENTATION_PRS
    # ORIENTATION_PLS
    # ORIENTATION_ARS
    # ORIENTATION_ALS
    # ORIENTATION_IPR
    # ORIENTATION_SPR
    # ORIENTATION_IAR
    # ORIENTATION_SAR
    # ORIENTATION_IPL
    # ORIENTATION_SPL
    # ORIENTATION_IAL
    # ORIENTATION_SAL
    # ORIENTATION_PIR
    # ORIENTATION_PSR
    # ORIENTATION_AIR 
    # ORIENTATION_ASR
    # ORIENTATION_PIL
    # ORIENTATION_PSL
    # ORIENTATION_AIL 
    # ORIENTATION_ASL
end