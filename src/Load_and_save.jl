# using Pkg
# Pkg.add(["DICOM", "NIfTI", "Dictionaries", "Dates"])
using   Dictionaries, Dates, PyCall
using Conda
using Accessors
# Conda.add("SimpleITK")
Conda.add("SimpleITK")
sitk = pyimport("SimpleITK")
include("./MedImage_data_struct.jl")
include("./Nifti_image_struct.jl")
include("./Utils.jl")



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

  # print(input_matrix)
  """
  INPUT MATRIX Is
  [r11 r12 r13 v1]
  [r21 r22 r23 v2]
  [r31 r32 r33 v3]
  [0    0   0   0]
  """
  r11, r12, r13 = input_matrix[1, 1], input_matrix[1, 2], input_matrix[1, 3]
  r21, r22, r23 = input_matrix[2, 1], input_matrix[2, 2], input_matrix[2, 3]
  r31, r32, r33 = input_matrix[3, 1], input_matrix[3, 2], input_matrix[3, 3]
  v1, v2, v3 = input_matrix[1, 4], input_matrix[2, 4], input_matrix[3, 4]
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

  """
  if qform_code < 0 

  qto_xyz 

  [dx 0.0 0.0 0.0]
  [0.0 dy 0.0 0.0]
  [0.0 0.0 dz 0.0]
  [0.0 0.0 0.0 1.0]

  """

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
  header_data_values = ["one", nifti_image_header.dim_info, nifti_image_header.dim, (nifti_image_header.intent_p1, nifti_image_header.intent_p2, nifti_image_header.intent_p3), (nifti_image_header.intent_code, string_intent(nifti_image_header.intent_code)),
    nifti_image_header.datatype, nifti_image_header.bitpix, nifti_image_header.slice_start, nifti_image_header.pixdim, nifti_image_header.vox_offset, nifti_image_header.scl_slope,
    nifti_image_header.scl_inter, nifti_image_header.slice_end, nifti_image_header.slice_code, nifti_image_header.xyzt_units, nifti_image_header.cal_max, nifti_image_header.cal_min,
    nifti_image_header.slice_duration, nifti_image_header.toffset, (nifti_image_header.descrip, formulate_string(nifti_image_header.descrip)), (nifti_image_header.aux_file, formulate_string(nifti_image_header.aux_file)),
    nifti_image_header.qform_code, formulate_xform_string(nifti_image_header.qform_code), nifti_image_header.sform_code, formulate_xform_string(nifti_image_header.sform_code),
    nifti_image_header.quatern_b, nifti_image_header.quatern_c, nifti_image_header.quatern_d, nifti_image_header.qoffset_x, nifti_image_header.qoffset_y, nifti_image_header.qoffset_z,
    nifti_image_header.srow_x, nifti_image_header.srow_y, nifti_image_header.srow_z, nifti_image_header.intent_name]

  header_data_dict = Dictionaries.Dictionary(header_data_keys, header_data_values)
  return header_data_dict
end


function formulate_must_rescale(scl_slope, scl_intercept)

  #checking to rescale voxels with double precision (usage of Float64)
  rescale_slope, rescale_intercept = convert(Float64, scl_slope), convert(Float64, scl_intercept)
  return abs(rescale_slope) > eps(Float64) && (abs(rescale_slope - 1.0) > eps(Float64) || abs(rescale_intercept) > eps(Float64))
end






"""

stuff for additional information
"""

function string_for_xyzt_units_space_code(space_code)
  code_dict = Dictionaries.Dictionary([1, 2, 3], ["NIFTI_UNITS_METER", "NIFTI_UNITS_MM", "NIFTI_UNITS_MICRON"])
  return code_dict[space_code]
end

function string_for_xyzt_units_time_code(time_code)
  code_dict = Dictionaries.Dictionary([8, 16, 24], ["NIFTI_UNITS_SEC", "NIFTI_UNITS_MSEC", "NIFTI_UNITS_USEC"])
  return code_dict[time_code]
end

function get_pixel_type(datatype)
  pixel_type_dict = Dictionaries.Dictionary([2, 4, 8], ["SCALAR", "SCALAR", "SCALAR"])
  if datatype in keys(pixel_type_dict)
    return pixel_type_dict[datatype]
  end
end



"""
helper function for nifti
calculating spacing scale from xyzt_units to space
"""
function formulate_spacing_scale_for_xyzt_space(xyzt_to_space)
  spacing_scale = 1.0
  spacing_scale_dict = Dictionaries.Dictionary(["NIFTI_UNITS_METER", "NIFTI_UNITS_MM", "NIFTI_UNITS_MICRON"], [1e3, 1e0, 1e-3])
  spacing_scale = spacing_scale_dict[xyzt_to_space]
  return spacing_scale
end

"""
helper function for nifti 
"""
function formulate_timing_scale_for_xyzt_time(xyzt_to_time)
  timing_scale = 1.0
  timing_scale_dict = Dictionaries.Dictionary(["NIFTI_UNITS_SEC", "NIFTI_UNITS_MSEC", "NIFTI_UNITS_USEC"], [1.0, 1e-3, 1e-6])
  timing_scale = timing_scale_dict[xyzt_to_time]
  return timing_scale
end


"""
helper function for nifti
creates a nifti_image struct which basically encapsulates all the necessary data, contains voxel data
"""
function formulate_nifti_image_struct(nifti_image::NIfTI.NIVolume)::Nifti_image
  nifti_image_header = nifti_image.header

  #set dimensions of data array 
  ndim = nifti_image_header.dim[1]
  nx = nifti_image_header.dim[2]
  ny = nifti_image_header.dim[3]
  nz = nifti_image_header.dim[4]
  nt = nifti_image_header.dim[5]
  nu = nifti_image_header.dim[6]
  nv = nifti_image_header.dim[7]
  nw = nifti_image_header.dim[8]

  dim = nifti_image_header.dim

  nvox = 1
  for n_index in 2:nifti_image_header.dim[1]
    nvox *= nifti_image_header.dim[n_index]
  end

  datatype = nifti_image_header.datatype

  #set grid spacing 
  dx = nifti_image_header.pixdim[2]
  dy = nifti_image_header.pixdim[3]
  dz = nifti_image_header.pixdim[4]
  dt = nifti_image_header.pixdim[5]
  du = nifti_image_header.pixdim[6]
  dv = nifti_image_header.pixdim[7]
  dw = nifti_image_header.pixdim[8]

  pixdim = nifti_image_header.pixdim

  scl_slope = nifti_image_header.scl_slope
  scl_inter = nifti_image_header.scl_inter

  cal_min = nifti_image_header.cal_min
  cal_max = nifti_image_header.cal_max

  qform_code = nifti_image_header.qform_code
  sform_code = nifti_image_header.sform_code

  freq_dim = Int(nifti_image_header.dim_info & 0x03) #or NIfTI.freqdim(dim_info)
  phase_dim = Int((nifti_image_header.dim_info >> 0x02) & 0x03) #or NIfTI.phasedim(dim_info)
  slice_dim = Int(nifti_image_header.dim_info >> 0x04) #or NIfTI.slicedim(dim_info)

  slice_code = nifti_image_header.slice_code
  slice_start = nifti_image_header.slice_start
  slice_end = nifti_image_header.slice_end
  slice_duration = nifti_image_header.slice_duration

  quatern_b = nifti_image_header.quatern_b
  quatern_c = nifti_image_header.quatern_c
  quatern_d = nifti_image_header.quatern_d
  qoffset_x = nifti_image_header.qoffset_x
  qoffset_y = nifti_image_header.qoffset_y
  qoffset_z = nifti_image_header.qoffset_z
  qfac = calculate_qfac(pixdim)

  qto_xyz = formulate_qto_xyz(quatern_b, quatern_c, quatern_d, qoffset_x, qoffset_y, qoffset_z, dx, dy, dz, qfac)
  #println(qto_xyz)
  # println(qto_xyz[2][1])
  qto_ijk = calculate_inverse_44_matrix(qto_xyz)


  sto_xyz = formulate_sto_xyz(nifti_image_header.srow_x, nifti_image_header.srow_y, nifti_image_header.srow_z)
  #println(sto_xyz)
  sto_ijk = calculate_inverse_44_matrix(sto_xyz)

  toffset = nifti_image_header.toffset
  xyz_units = Int(nifti_image_header.xyzt_units & 0x07)
  time_units = Int(nifti_image_header.xyzt_units & 0x38)
  nifti_type = "one"

  intent_code = nifti_image_header.intent_code
  intent_p1 = nifti_image_header.intent_p1
  intent_p2 = nifti_image_header.intent_p2
  intent_p3 = nifti_image_header.intent_p3
  intent_name = string_intent(intent_code)

  descrip = formulate_string(nifti_image_header.descrip)
  aux_file = formulate_string(nifti_image_header.aux_file)

  #conversion to world coordinates (then  storing the data)
  data = nothing

  num_ext = 0
  ext_list = []


  #instantiating nifti image io struct

  nifti_image_io_scl_slope = abs(convert(Float64, scl_slope)) < eps(Float64) ? 1.0f0 : scl_slope
  nifti_image_io_information = Nifti_image_io(scl_slope, scl_inter, formulate_must_rescale(scl_slope, scl_inter))

  nifti_image_struct_instance = Nifti_image([ndim, nx, ny, nz, nt, nu, nv, nw, dim, nvox, datatype,
    dx, dy, dz, dt, du, dv, dw, pixdim, scl_slope, scl_inter,
    cal_min, cal_max, qform_code, sform_code, freq_dim, phase_dim, slice_dim,
    slice_code, slice_start, slice_end, slice_duration, quatern_b, quatern_c,
    quatern_d, qoffset_x, qoffset_y, qoffset_z, qfac, qto_xyz, qto_ijk, sto_xyz, sto_ijk,
    toffset, xyz_units, time_units, nifti_type, intent_code, intent_p1, intent_p2, intent_p3, intent_name,
    descrip, aux_file, data, num_ext, ext_list, nifti_image_io_information])
  return nifti_image_struct_instance
end


"""
helper function for nifti
setting spacing for 3D nifti filesd(4D nfiti file yet to be added)
"""
function set_spacing_for_nifti_files(dimension)
  spacing_scale = formulate_spacing_scale_for_xyzt_space("NIFTI_UNITS_MM")
  #for 4D nifti files use timing scale
  #timing_scale = formulate_timing_scale_for_xyzt_time("NIFTI_UNITS_SEC")
  dx = dimension[1]
  dy = dimension[2]
  dz = dimension[3] #pixdim from nifti
  return (dx, dy , dz)
end






  

"""
helper function for nifti 
checking similarity of s_transformation_matrix and q_transformation_matrix
"""
function check_sform_qform_similarity(q_transformation_matrix, s_transformation_matrix)
  s_rotation_scale = s_transformation_matrix[1:3,1:3]
  q_rotation_scale = q_transformation_matrix[1:3,1:3]
  
  qform_sform_similar = true
    # Check if matrices have same dimensions
  if size(s_rotation_scale) != size(q_rotation_scale)
        qform_sform_similar =  false
  end
    
    # Check each element for equality within tolerance
    for i in 1:size(s_rotation_scale, 1)
        for j in 1:size(s_rotation_scale, 2)
            if abs(s_rotation_scale[i, j] - q_rotation_scale[i, j]) > 1e-5
                qform_sform_similar =  false
            end
        end
    end
    
    #Second check that the translations are the same with very small tolerance
  if sum(abs.(s_transformation_matrix[:, 4] - q_transformation_matrix[:, 4])) > 1e-7
    qform_sform_similar =  false
  end 
  if sum(abs.(s_transformation_matrix[4, :] - q_transformation_matrix[4, :])) > 1e-7
    qform_sform_similar =  false
  end 
  
  return qform_sform_similar
end


"""
helper function for nifti
setting the direction cosines (orientation) for a 3D nifti file
"""
function set_direction_for_nifti_file(nifti_file_path,qform_xform_string, sform_xform_string, qform_sform_similar)
#using SimpleITk for getting direction cosines
itk_nifti_image = sitk.ReadImage(nifti_file_path)
direction_cosines = itk_nifti_image.GetDirection()
return direction_cosines
end

"""
helper function for nifti
setting the origin for a 3D nifti file
"""
function set_origin_for_nifti_file(qform_sform_similar, s_transformation_matrix )
if qform_sform_similar
    origin = (s_transformation_matrix[1,4], s_transformation_matrix[2,4], s_transformation_matrix[3,4])
    return origin
end
end






function load_images(path::String)::Array{MedImage}
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

    # voxel_data = permutedims(voxel_data, (3, 2, 1))

  """


      origin = nothing
      spacing = nothing
      direction = nothing #direction cosines for oreintation
      #2 data for the fields within the MedImage struct

      spatial_metadata_keys = ["origin", "spacing", "orientation"]
      spatial_metadata_values = [nothing, nothing, nothing]
      spatial_metadata = Dictionaries.Dictionary(spatial_metadata_keys, spatial_metadata_values)
  """

    #3 Image type
    image_type = Image_type(1) #set to MRI/PET/CT
    #4 Image subtype
    image_subtype = Image_subtype(0) #set to subtypes
    #5 voxel datatype
    voxel_datatype = nothing
    #6 date of saving
    date_of_saving = Dates.format(Dates.now(), "yyyy-mm-dd")
    #7 acquisition time
    acquisition_time = Dates.format(Dates.now(), "HH:MM:SS")
    #8 patient ID
    patient_id = nothing
    #9 current device cpu or gpu
    current_device = ""
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
    clinical_data = Dictionaries.Dictionary()
    #17 contrast administration boolean
    is_contrast_administered = false
    #18 metadata dictionary from nifti header

    metadata_keys = ["header_data_dict", "nifti_image_struct"]

    header_data_dict = formulate_header_data_dict(nifti_image_header)
    nifti_image_struct = formulate_nifti_image_struct(nifti_image)
    metadata_values = [header_data_dict, nifti_image_struct]
    metadata = Dictionaries.Dictionary(metadata_keys, metadata_values)

    
    """resetting origin, spacing and direction (since we have all the nifti image struct now)"""
    sform_qform_similar = check_sform_qform_similarity(nifti_image_struct.qto_xyz, nifti_image_struct.sto_xyz)
    

    itk_nifti_image = sitk.ReadImage(path)
    origin = itk_nifti_image.GetOrigin()
    # origin = set_origin_for_nifti_file(sform_qform_similar, nifti_image_struct.sto_xyz)
    spacing = itk_nifti_image.GetSpacing()  #set_spacing_for_nifti_files([nifti_image_struct.dx, nifti_image_struct.dy,nifti_image_struct.dz])
    # spacing=(spacing[3],spacing[2],spacing[1])
    direction = set_direction_for_nifti_file(path,header_data_dict["qform_code_name"], header_data_dict["sform_code_name"], sform_qform_similar)
    voxel_arr=sitk.GetArrayFromImage(itk_nifti_image)
    voxel_arr=permutedims(voxel_arr,(3,2,1))
    spatial_metadata_keys = ["origin","spacing","direction"]
    spatial_metadata_values=  [origin,spacing,direction]
    spatial_metadata = Dictionaries.Dictionary(spatial_metadata_keys,spatial_metadata_values)

    return [MedImage(voxel_data=voxel_arr, origin= origin, spacing=spacing, direction=direction,patient_id="test_id"
    , image_type=CT_type, image_subtype=CT_subtype, legacy_file_name=string(legacy_file_name))]

    #return [MedImage([nifti_image.raw, nifti_image_header.pixdim[2:4], (nifti_image_header.srow_x[1:3], nifti_image_header.srow_y[1:3], nifti_image_header.srow_z[1:3]), (nifti_image_header.qoffset_x, nifti_image_header.qoffset_y, nifti_image_header.qoffset_z), " ", " "])]


  end

end



#testing the above written function with an example file

# medimage_instance_array = load_image("/home/jakubmitura/projects/MedImage.jl/test_data/volume-0.nii.gz")
# medimage_instance = medimage_instance_array[1]
# #println(typeof(medimage_instance.voxel_data))
# println("spacing")
# println(medimage_instance.spacing)
# println("direction")
# println(medimage_instance.direction)
# println("origin")
# println(medimage_instance.origin)

# function save_image(im::MedImage, path::String)
#   """
#   Creating nifti volumes or each medimage object
#   NIfTI.NIVolume() NIfTI.niwrite()
#   """
#   for medimage in im

#   #new nifti header construction
#   #providing default values for somes fields based on the NIfTI1Header struct defintion, spatial metadata fields are populated from medimage objects, which are changed upon any transformation.
#   """
#   struct NIfTI1Header
#   sizeof_hdr     :: Int32
#   data_type      :: NTuple{10, UInt8}
#   db_name        :: NTuple{18, UInt8}
#   extents        :: Int32
#   session_error  :: Int16
#   regular        :: Int8

#   dim_info       :: Int8
#   dim            :: NTuple{8, Int16}
#   intent_p1      :: Float32
#   intent_p2      :: Float32
#   intent_p3      :: Float32
#   intent_code    :: Int16
#   datatype       :: Int16
#   bitpix         :: Int16
#   slice_start    :: Int16
#   pixdim         :: NTuple{8, Float32}
#   vox_offset     :: Float32
#   scl_slope      :: Float32
#   scl_inter      :: Float32
#   slice_end      :: Int16
#   slice_code     :: Int8
#   xyzt_units     :: Int8
#   cal_max        :: Float32
#   cal_min        :: Float32
#   slice_duration :: Float32
#   toffset        :: Float32
#   glmax          :: Int32
#   glmin          :: Int32
#   descrip        :: NTuple{80, UInt8}
#   aux_file       :: NTuple{24, UInt8}
#   qform_code     :: Int16
#   sform_code     :: Int16
#   quatern_b      :: Float32
#   quatern_c      :: Float32
#   quatern_d      :: Float32
#   qoffset_x      :: Float32
#   qoffset_y      :: Float32
#   qoffset_z      :: Float32
#   srow_x         :: NTuple{4, Float32}
#   srow_y         :: NTuple{4, Float32}
#   srow_z         :: NTuple{4, Float32}
#   intent_name    :: NTuple{16, UInt8}
#   magic          :: NTuple{4, UInt8}
#   end
#   """
#   nifti_file_header = NIfTI.NIfTI1Header(348,
#                                         (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)                                         ,
#                                          (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00),
#                                          0,
#                                           0,
#                                          114,
#                                           medimage.metadata["header_data_dict"]["dim_info"],
#                                          medimage.metadata["header_data_dict"]["dim"],
#                                          medimage.metadata["header_data_dict"]["nifti_intent"][1],
#                                          medimage.metadata["header_data_dict"]["nifti_intent"][2] ,
#                                          medimage.metadata["header_data_dict"]["nifti_intent"][3],
#                                          medimage.metadata["header_data_dict"]["nifti_intent_code"][1],
#                                          medimage.metadata["header_data_dict"]["datatype"],
#                                          medimage.metadata["header_data_dict"]["bitpix"],
# medimage.metadata["header_data_dict"]["slice_start"],
#  medimage.metadata["header_data_dict"]["pixdim"],
#  medimage.metadata["header_data_dict"]["vox_offset"],
#  medimage.metadata["header_data_dict"]["scl_slope"],
#  medimage.metadata["header_data_dict"]["scl_inter"],
# medimage.metadata["header_data_dict"]["slice_end"],
#  medimage.metadata["header_data_dict"]["slice_code"],
#  medimage.metadata["header_data_dict"]["xyzt_units"],
#  medimage.metadata["header_data_dict"]["cal_max"],
#  medimage.metadata["header_data_dict"]["cal_min"],
#  medimage.metadata["header_data_dict"]["slice_duration"],
#  medimage.metadata["header_data_dict"]["toffset"],
#  255,
#  0,
#  medimage.metadata["header_data_dict"]["descrip"][1],
#  medimage.metadata["header_data_dict"]["aux_file"][1],
#  medimage.metadata["header_data_dict"]["qform_code"],
#  medimage.metadata["header_data_dict"]["sform_code"],
#  medimage.metadata["header_data_dict"]["quatern_b"],
#  medimage.metadata["header_data_dict"]["quatern_c"],
#  medimage.metadata["header_data_dict"]["quatern_d"],
#  medimage.metadata["header_data_dict"]["qoffset_x"],
#  medimage.metadata["header_data_dict"]["qoffset_y"],
#  medimage.metadata["header_data_dict"]["qoffset_z"],
#  medimage.metadata["header_data_dict"]["srow_x"],
#  medimage.metadata["header_data_dict"]["srow_y"],
#  medimage.metadata["header_data_dict"]["srow_z"],
# medimage.metadata["header_data_dict"]["intent_name"],
# (0x6e, 0x2b, 0x31, 0x00))
#   nifti_file_data =permutedims(medimage.voxel_data,(3,2,1))
#   nifti_file_volume = NIfTI.NIVolume(nifti_file_header, nifti_file_data)
#   NIfTI.niwrite("output_nifti_file.nii.gz",nifti_file_volume)
  
#   end

# end


function create_nii_from_medimage(med_image::MedImage, file_path::String)
  # Convert voxel_data to a numpy array (Assuming voxel_data is stored in Julia array format)
  voxel_data_np =med_image.voxel_data
  voxel_data_np=permutedims(voxel_data_np,(3,2,1))
  # Create a SimpleITK image from numpy array
  image_sitk = sitk.GetImageFromArray(voxel_data_np)
  
  # Set spatial metadata
  image_sitk.SetOrigin(med_image.origin)
  image_sitk.SetSpacing(med_image.spacing)
  image_sitk.SetDirection(med_image.direction)
  
  # Save the image as .nii.gz
  sitk.WriteImage(image_sitk, file_path* ".nii.gz")
end



#**************************************************

function update_voxel_data(old_image::MedImage, new_voxel_data::AbstractArray)
  
  return MedImage(
      new_voxel_data, 
      old_image.origin, 
      old_image.spacing, 
      old_image.direction, 
      # old_image.spatial_metadata, 
      old_image.image_type, 
      old_image.image_subtype, 
      old_image.voxel_datatype, 
      old_image.date_of_saving, 
      old_image.acquistion_time, 
      old_image.patient_id, 
      old_image.current_device, 
      old_image.study_uid, 
      old_image.patient_uid, 
      old_image.series_uid, 
      old_image.study_description, 
      old_image.legacy_file_name, 
      old_image.display_data, 
      old_image.clinical_data, 
      old_image.is_contrast_administered, 
      old_image.metadata)

end


function update_voxel_and_spatial_data(old_image::MedImage, new_voxel_data::AbstractArray
  ,new_origin,new_spacing,new_direction)

  res=@set old_image.voxel_data=new_voxel_data
  res=@set res.origin=ensure_tuple(new_origin)
  res=@set res.spacing=ensure_tuple(new_spacing)
  res=@set res.direction=ensure_tuple(new_direction)
  # voxel_data=new_voxel_data
  # origin=new_origin
  # spacing=new_spacing
  # direction=new_direction

  # return @pack! old_image = voxel_data, origin, spacing, direction
  return res
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


"""
NOTES:
conversion and storage n-dimensional voxel data within world-coordinate system (doesnt change the RAS system)
for testing purposes within test_data the volume-0.nii.gz constitutes a 3-D nifti volume , whereas the filtered_func_data.nii.gz constitutes up of a 4D nifti volume (4th dimension is time, 3d stuff still applicable)
"""

#array_of_objects = load_image("../test_data/ScalarVolume_0")
# array_of_objects = load_image("/workspaces/MedImage.jl/MedImage3D/test_data/volume-0.nii.gz")
#println(length(array_of_objects[1].pixel_array))
#

"""
testing with nifti volume


file_path = "./../test_data/volume-0.nii.gz"
medimage_object_array = load_image(file_path)
save_image(medimage_object_array, "./outputs")
"""