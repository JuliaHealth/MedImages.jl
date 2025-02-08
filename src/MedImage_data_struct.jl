module MedImage_data_struct
using Dates, Dictionaries, Parameters
using Dictionaries
using Parameters
using UUIDs

export MedImage, Image_type, Image_subtype, current_device_enum, Interpolator_enum, Mode_mi, CoordinateTerms, CoordinateMajornessTerms, Orientation_code

"""

    @enum Image_type


Defines the type of medical image. Possible values are:

- `MRI_type`: Magnetic Resonance Imaging

- `PET_type`: Positron Emission Tomography

- `CT_type`: Computed Tomography

"""
@enum Image_type begin
  MRI_type
  PET_type
  CT_type
end



"""

    @enum current_device_enum


Specifies the current device used for processing. Possible values are:

- `CPU_current_device`: Central Processing Unit

- `CUDA_current_device`: NVIDIA CUDA-enabled GPU

- `AMD_current_device`: AMD GPU

- `ONEAPI_current_device`: Intel oneAPI-enabled device

"""
@enum current_device_enum begin
  CPU_current_device
  CUDA_current_device
  AMD_current_device
  ONEAPI_current_device

end


"""
    @enum Image_subtype


Defines the subtype of medical image. Possible values include:

- `CT_subtype`: CT scan subtype

- `ADC_subtype`: Apparent Diffusion Coefficient

- `DWI_subtype`: Diffusion Weighted Imaging

- `T1_subtype`: T1-weighted MRI

- `T2_subtype`: T2-weighted MRI

- `FLAIR_subtype`: Fluid-attenuated inversion recovery

- `FDG_subtype`: Fluorodeoxyglucose PET

- `PSMA_subtype`: Prostate-Specific Membrane Antigen PET

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


# following struct can be expanded with all the relevant meta data mentioned within the readme.md of MedImage.jl
  #struct for now, will switch to MetaArrays when it has GPU support

"""

    mutable struct MedImage


A standardized structure for storing medical image data and metadata.


# Fields

- `voxel_data`: Multidimensional array representing the image data.

- `origin`: Tuple of 3 Float64 values indicating the origin of the image.

- `spacing`: Tuple of 3 Float64 values indicating the spacing between voxels.

- `direction`: 9-element tuple of Float64 values for orientation cosines.

- `image_type`: Enum `Image_type` indicating the type of image.

- `image_subtype`: Enum `Image_subtype` indicating the subtype of image.

- `date_of_saving`: DateTime when the image was saved.

- `acquistion_time`: DateTime when the image was acquired.

- `patient_id`: String identifier for the patient.

- `current_device`: Enum `current_device_enum` indicating the processing device.

- `study_uid`: Unique identifier for the study.

- `patient_uid`: Unique identifier for the patient.

- `series_uid`: Unique identifier for the series.

- `study_description`: Description of the study.

- `legacy_file_name`: Original file name.

- `display_data`: Dictionary for color values (e.g., RGB or grayscale).

- `clinical_data`: Dictionary with clinical data (e.g., age, gender).

- `is_contrast_administered`: Boolean indicating if contrast was used.

- `metadata`: Dictionary for additional metadata.

"""
@with_kw mutable struct MedImage
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

    @enum Interpolator_enum


Defines basic interpolators for image processing. Possible values are:

- `Nearest_neighbour_en`: Nearest neighbor interpolation

- `Linear_en`: Linear interpolation

- `B_spline_en`: B-spline interpolation

"""
@enum Interpolator_enum Nearest_neighbour_en Linear_en B_spline_en

"""

    @enum Mode_mi


Indicates the mode of operation for modifying image data. Possible values are:

- `pixel_array_mode`: Modify pixel array only

- `spat_metadata_mode`: Modify spatial metadata only

- `all_mode`: Modify both pixel array and spatial metadata

"""
@enum Mode_mi pixel_array_mode = 0 spat_metadata_mode = 2 all_mode = 3


################## orientation
"""

    @enum CoordinateTerms


Defines coordinate terms based on ITK spatial orientation. Possible values are:

- `ITK_COORDINATE_UNKNOWN`: Unknown coordinate

- `ITK_COORDINATE_Right`: Right

- `ITK_COORDINATE_Left`: Left

- `ITK_COORDINATE_Posterior`: Posterior

- `ITK_COORDINATE_Anterior`: Anterior

- `ITK_COORDINATE_Inferior`: Inferior

- `ITK_COORDINATE_Superior`: Superior

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





"""

    @enum CoordinateMajornessTerms


Defines the majorness of coordinates. Possible values are:

- `PrimaryMinor`: Primary minor coordinate

- `SecondaryMinor`: Secondary minor coordinate

- `TertiaryMinor`: Tertiary minor coordinate

"""
@enum CoordinateMajornessTerms begin
    PrimaryMinor = 0
    SecondaryMinor = 8
    TertiaryMinor = 16
end




"""

    @enum Orientation_code


Defines orientation codes for medical images. Possible values include:

- `ORIENTATION_RPI`: Right-Posterior-Inferior

- `ORIENTATION_LPI`: Left-Posterior-Inferior

- `ORIENTATION_RAI`: Right-Anterior-Inferior

- `ORIENTATION_LAI`: Left-Anterior-Inferior

- `ORIENTATION_RPS`: Right-Posterior-Superior

- `ORIENTATION_LPS`: Left-Posterior-Superior

- `ORIENTATION_RAS`: Right-Anterior-Superior

- `ORIENTATION_LAS`: Left-Anterior-Superior

"""
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

end#MedImage_data_struct
