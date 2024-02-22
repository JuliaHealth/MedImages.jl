using Pkg
Pkg.add(["DICOM", "NIfTI", "Dictionaries", "Dates"])
using DICOM, NIfTI, Dictionaries, Dates
include("./MedImage_data_struct.jl")




"===============================================DICOM FUNCTIONS----------------------------------------------------------------------------"


"""
helper function for dicom #1
returns an array of unique SERIES INSTANCE UID within dicom files within a dicom directory
"""
function unique_series_id_within_dicom_files(dicom_data_array)
  return map(dicom_file_data -> dicom_file_data[tag"SeriesInstanceUID"], dicom_data_array) |>
         Set |>
         collect
end

"""
helper function for dicom #2
returns an array of pixel data for unique ids within dicom files
"""
function get_pixel_data(dicom_data_array)
  if length(dicom_data_array) == 1
    #in case e have 2D image
    return only(dicom_data_array).PixelData
  else
    #in case we have 3D image
    return cat([dcm.PixelData for dcm in dicom_data_array]...; dims=3)
  end
end









"-----------------------------------------------NIFTI FUNCTIONS---------------------------------------------------------------------------------------"

"""
helper function for nifti #1
return a concatenated string for encoded iterables
"""
function formulate_string(iterable)::String
  concatenated_string = ""
  for chr in string.(Char.(iterable))
    concatenated_string *= chr
  end
  return concatenated_string
end


"""
helper function for nifti #2
return relevant xform string names from codes (qform and sform)
"""
function formulate_xform_string(code)::String
  xform_codes = [0, 1, 2, 3, 4] #NOTE: QFORM_CODE(0:2) and SFORM_CODE(0:4)
  xform_strings = ["NIFTI_XFORM_UNKNOWN", "NIFTI_XFORM_SCANNER_ANAT", "NIFTI_XFORM_ALIGNED_ANAT", "NIFTI_XFORM_TALAIRACH", "NIFTI_XFORM_MNI_152"]
  xform_code_dictionary = Dictionaries.Dictionary(xform_codes, xform_strings)
  return xform_code_dictionary[code]
end

"""
helper function nifti
return qfac after calculation
"""
function calculate_qfac(pixdim)
  return (pixdim[1] < 0.0) ? -1.0 : 1.0
end



"""
helper function for nifti
calculates inverse of a 4x4 matrix
"""
function calculate_inverse_44_matrix(input_matrix)
  """
  INPUT MATRIX Is
  [r11 r12 r13 v1]
  [r21 r22 r23 v2]
  [r31 r32 r33 v3]
  [0    0   0   0]
  """
  r11, r12, r13 = input_matrix[1][1], input_matrix[1][2], input_matrix[1][3]
  r21, r22, r23 = input_matrix[2][1], input_matrix[2][2], input_matrix[2][3]
  r31, r32, r33 = input_matrix[3][1], input_matrix[3][2], input_matrix[3][3]
  v1, v2, v3 = input_matrix[1][4], input_matrix[2][4], input_matrix[3][4]
  det = r11 * r22 * r33 - r11 * r32 * r23 - r21 * r12 * r33 + r21 * r32 * r13 + r31 * r12 * r23 - r31 * r22 * r13 #determinant
  if det != 0.0
    det = 1.0 / det
  end
  inverse_44_matrix = [det*(r22*r33-r32*r23) det*(-r12*r33+r32*r13) det*(r12*r23-r22*r13) det*(-r12*r23*v3+r12*v2*r33+r22*r13*v3-r22*v1*r33-r32*r13*v2+r32*v1*r23); det*(-r21*r33+r31*r23) det*(r11*r33-r31*r13) det*(-r11*r23+r21*r13) det*(r11*r23*v3-r11*v2*r33-r21*r13*v3+r21*v1*r33+r31*r13*v2-r31*v1*r23); det*(r21*r32-r31*r22) det*(-r11*r32+r31*r12) det*(r11*r22-r21*r12) det*(-r11*r22*v3+r11*r32*v2+r21*r12*v3-r21*r32*v1-r31*r12*v2+r31*r22*v1); 0.0 0.0 0.0 1.0]
  return inverse_44_matrix
end

"""
helper function for nifti
create a qform matrix from the quaterns
"""
function formulate_qto_xyz(quatern_b, quatern_c, quatern_d, qoffset_x, qoffset_y, qoffset_z, dx, dy, dz, qfac) # a 4x4 matrix

  #dealing with only nifti files
  #NOTE QFORM_CODE should not <= 0, which will mean we then would have to use grid spacings in order to compute the transformation matrix
  #using quaterns for qform_code > 0
  #quatern_to_mat4x4(quatern_b,quatern_c,quatern_d,
  #                qoffset_x, qoffset_y, qoffset_z,
  #                dx, dy, dz, qfac,
  #                0,0,0,1)
  #

  b, c, d = quatern_b, quatern_c, quatern_d
  qx, qy, qz = qoffset_x, qoffset_y, qoffset_z
  #calculating a paramter from the given quaterns
  a = 1.0 - (b^2 + c^2 + d^2)
  if a < 1.e-7 #special case
    a = 1.0 / sqrt(b^2 + c^2 + d^2)
    b *= a #normalizing b,c,d vector
    c *= a
    d *= a
    a = 0.0 #a=0, 180 degree rotation
  else
    a = sqrt(a)
  end

  #loading rotation matrix, including scaling factor for voxel sizes
  #dx , dy and dz are pixdim[2], pixdim[3] and pixdim[4] respectivelyd
  xd = (dx > 0.0) ? dx : 1.0
  yd = (dy > 0.0) ? dy : 1.0
  zd = (dz > 0.0) ? dz : 1.0
  if qfac < 0.0
    zd = -zd #left handedness?
  end
  #last row is always 0 0 0 1
  #the first 3X3 matrix in the below 4x4 matrix is Rotation Matrix
  qform_transformation_matrix = [(a^2+b^2-c^2-d^2)*xd 2.0*(b*c-a*d)*yd 2.0*(b*d+a*c)*zd qx; 2.0*(b*c+a*d)*xd (a*a+c*c-b*b-d*d)*yd 2.0*(c*d-a*b)*zd qy; 2.0*(b*d-a*c)*xd 2.0*(c*d+a*b)*yd (a*a+d*d-c*c-b*b)*zd qz; 0.0 0.0 0.0 1.0]
  return qform_transformation_matrix

end

function formulate_qto_ijk(qto_xyz)
  inverse_qform_transformation_matrix = calculate_inverse_44_matrix(qto_xyz)
  return inverse_qform_transformation_matrix
end


"""
helper function for nifti
returns a 4x4 matrix for srow_x, srow_y and srow_z
"""
function formulate_sto_xyz(srow_x, srow_y, srow_z)
  #earlier i used : hcat((srow_x,srow_y,srow_z)...) for concatenation along the horizontal
  sform_transformation_matrix = [srow_x[1] srow_x[2] srow_x[3] srow_x[4]; srow_y[1] srow_y[2] srow_y[3] srow_y[4]; srow_z[1] srow_z[2] srow_z[3] srow_z[4]; 0.0 0.0 0.0 1.0]
  return sform_transformation_matrix
  #horizontal concatenation
end

function formulate_sto_ijk(sto_xyz)
  inverse_sform_transformation_matrix = calculate_inverse_44_matrix(sto_xyz)
  return inverse_sform_transformation_matrix
end

"""
helper function for nifti
returns a string version for the specified intent code from nifti
"""
function string_intent(intent)
  string_intent_dict = Dictionaries.Dictionary(filter(value -> value != 1, [0:24..., 1001:1011...]), #excluding codes from 2001:2005
    ["NIFTI_INTENT_NONE", "NIFTI_INTENT_CORREL", "NIFTI_INTENT_TTEST",
      "NIFTI_INTENT_FTEST", "NIFTI_INTENT_ZSCORE", "NIFTI_INTENT_CHISQ", "NIFTI_INTENT_BETA", "NIFTI_INTENT_BINOM", "NIFTI_INTENT_GAMMA", "NIFTI_INTENT_POISSON",
      "NIFTI_INTENT_NORMAL", "NIFTI_INTENT_FTEST_NONC", "NIFTI_INTENT_CHISQ_NONC", "NIFTI_INTENT_LOGISTIC", "NIFTI_INTENT_LAPLACE", "NIFTI_INTENT_UNIFORM",
      "NIFTI_INTENT_TTEST_NONC", "NIFTI_INTENT_WEIBULL", "NIFTI_INTENT_CHI", "NIFTI_INTENT_INVGAUSS", "NIFTI_INTENT_EXTVAL", "NIFTI_INTENT_PVAL", "NIFTI_INTENT_LOGPVAL",
      "NIFTI_INTENT_LOG10PVAL", "NIFTI_INTENT_ESTIMATE", "NIFTI_INTENT_LABEL", "NIFTI_INTENT_NEURONAME", "NIFTI_INTENT_GENMATRIX", "NIFTI_INTENT_SYMMATRIX", "NIFTI_INTENT_DISPVECT",
      "NIFTI_INTENT_VECTOR", "NIFTI_INTENT_POINTSET", "NIFTI_INTENT_TRIANGLE", "NIFTI_INTENT_QUATERNION", "NIFTI_INTENT_DIMLESS"])
  if intent in keys(string_intent_dict)
    return string_intent_dict[intent]
  else
    return "UNKNOWN_INTENT"
  end
end

"""
helper function for nifti
creates a data dictionary for header data can be used to create a new NIfTI.NIfTI1Header when saving to a file
"""
function formulate_header_data_dict(nifti_image_header::NIfTI.NIfTI1Header)::Dictionaries.Dictionary

  #below in the auxfile[1] and description[1] are the actual values for the field, followed by the string representation of them
  #for qform_code and sform_code , their string representations of separate feilds such as qform_code_name and sform_code_name

  header_data_keys = ["nifti_type", "dim_info", "dim", "nifti_intent", "nifti_intent_code", "datatype", "bitpix", "slice_start", "pixdim", "vox_offset", "scl_slope", "scl_inter", "slice_end", "slice_code",
    "xyzt_units", "cal_max", "cal_min", "slice_duration", "toffset", "descrip", "aux_file", "qform_code", "qform_code_name", "sform_code", "sform_code_name", "quatern_b", "quatern_c", "quatern_d",
    "qoffset_x", "qoffset_y", "qoffset_z", "srow_x", "srow_y", "srow_z", "intent_name"]
  header_data_values = ["one", nifti_image_header.dim_info, nifti_image_header.dim, (nifti_image_header.intent_p1, nifti_image_header.intent_p2, nifti_image_header.intent_p3), nifti_image_header.intent_code,
    nifti_image_header.datatype, nifti_image_header.bitpix, nifti_image_header.slice_start, nifti_image_header.pixdim, nifti_image_header.vox_offset, nifti_image_header.scl_slope,
    nifti_image_header.scl_inter, nifti_image_header.slice_end, nifti_image_header.slice_code, nifti_image_header.xyzt_units, nifti_image_header.cal_max, nifti_image_header.cal_min,
    nifti_image_header.slice_duration, nifti_image_header.toffset, (nifti_image_header.descrip, formulate_string(nifti_image_header.descrip)), (nifti_image_header.aux_file, formulate_string(nifti_image_header.aux_file)),
    nifti_image_header.qform_code, formulate_xform_string(nifti_image_header.qform_code), nifti_image_header.sform_code, formulate_xform_string(nifti_image_header.sform_code),
    nifti_image_header.quatern_b, nifti_image_header.quatern_c, nifti_image_header.quatern_d, nifti_image_header.qoffset_x, nifti_image_header.qoffset_y, nifti_image_header.qoffset_z,
    nifti_image_header.srow_x, nifti_image_header.srow_y, nifti_image_header.srow_z, nifti_image_header.intent_name]

  header_data_dict = Dictionaries.Dictionary(header_data_keys, header_data_values)
  return header_data_dict
end

"""
helper function for nifti
creates a nifti_image struct which basically encapsulates all the necessary data, contains voxel data
"""
function formulate_nifti_image_struct(nifti_image::NIfTI.NIVolume)::Nifti_image

end


function load_image(path::String)::Array{MedImage}
  if isdir(path)
    dicom_data_array = DICOM.dcmdir_parse(path)
    return unique_series_id_within_dicom_files(dicom_data_array) |>
           series_ids -> map(series_id -> filter(dcm -> dcm.SeriesInstanceUID == series_id, dicom_data_array), series_ids) |> #now we have series of dicom files with the same series id
                         dcom_data_locs -> map(dcom_data_loc -> MedImage([get_pixel_data(dcom_data_loc),
        dcom_data_loc[1].PixelSpacing,
        dcom_data_loc[1].ImageOrientationPatient,
        dcom_data_loc[1].ImagePositionPatient,
        " ",#dcom_data_loc[1].StudyCompletionDate
        dcom_data_loc[1].PatientID]), dcom_data_locs)
  else
    nifti_image = NIfTI.niread(path)
    nifti_image_header = nifti_image.header


    #1 voxel data from the nifti image
    voxel_data = nifti_image.raw #this data is in image coordinates (conversion to world coordinates )

    origin = nothing
    spacing = nothing
    orientation = nothing
    #2 data for the fields within the MedImage struct
    spatial_metadata_keys = ["origin", "spacing", "orientation"]
    spatial_metadata_values = []
    spatial_metadata = Dictionaries.Dictionary(spatial_metadata_keys, spatial_metadata_values)
    #3 Image type
    image_type = nothing #set to MRI/PET/CT
    #4 Image subtype
    image_subtype = nothing #set to subtypes
    #5 voxel datatype
    voxel_datatype = nothing
    #6 date of saving
    date_of_saving = Dates.format(Dates.now(), "yyyy-mm-dd")
    #7 acquisition time
    acquisition_time = Dates.format(Dates.now(), "HH:MM:SS")
    #8 patient ID
    patient_id = nothing
    #9 current device cpu or gpu
    current_device = nothing
    #10 study uid
    study_uid = nothing
    #11 patient uid
    patient_uid = nothing
    #12 series uid
    series_uid = nothing
    #13 study description
    study_description = formulate_string(nifti_image_header.descrip)
    #14 legacy file name
    legacy_file_name = split(path, "/")[length(split(path, "/"))]
    #15 display data : RGB or gray
    display_data = nothing
    #16 clinical data
    clinical_data = nothing
    #17 contrast administration boolean
    is_contrast_administered = false
    #18 metadata dictionary from nifti header

    metadata_keys = ["header_data_dict", "nifti_image_struct"]
    metadata_values = [formulate_header_data_dict(nifti_image_header), formulate_nifti_image_struct(nifti_image)]
    metadata = Dictionaries.Dictionary(metadata_keys, metadata_values)


    return [MedImage([voxel_data, origin, spacing ,orientation, spatial_metadata, image_type, image_subtype, voxel_datatype, date_of_saving, acquisition_time, patient_id, current_device, study_uid, patient_uid, series_uid, study_description, legacy_file_name, display_data, clinical_data, is_contrast_administered, metadata])]
    #return [MedImage([nifti_image.raw, nifti_image_header.pixdim[2:4], (nifti_image_header.srow_x[1:3], nifti_image_header.srow_y[1:3], nifti_image_header.srow_z[1:3]), (nifti_image_header.qoffset_x, nifti_image_header.qoffset_y, nifti_image_header.qoffset_z), " ", " "])]


  end

end



function save_image(im::Array{MedImage}, path::String)
  """
  Creating nifti volumes or each medimage object
  NIfTI.NIVolume() NIfTI.niwrite()
  """
end


"""
NOTES:
conversion and storage n-dimensional voxel data within world-coordinate system (doenst change the RAS system)
for testing purposes within test_data the volume-0.nii.gz constitutes a 3-D nifti volume , whereas the filtered_func_data.nii.gz constitutes up of a 4D nifti volume (4th dimension is time, 3d stuff still applicable)
"""

#array_of_objects = load_image("../test_data/ScalarVolume_0")
# array_of_objects = load_image("/workspaces/MedImage.jl/MedImage3D/test_data/volume-0.nii.gz")
#println(length(array_of_objects[1].pixel_array))
