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
  xform_codes = [0, 1, 2, 3, 4] #since the ranges for sform and qform codes is 0-4
  xform_strings = ["NIFTI_XFORM_UNKNOWN", "NIFTI_XFORM_SCANNER_ANAT", "NIFTI_XFORM_ALIGNED_ANAT", "NIFTI_XFORM_TALAIRACH", "NIFTI_XFORM_MNI_152"]
  xform_code_dictionary = Dictionaries.Dictionary(xform_codes, xform_strings)
  return xform_code_dictionary[code]
end

"""
helper function for nifti #3
returns a matrix for srow_x, srow_y and srow_z
"""
function formulate_sto_xyz(srow_values)
  return hcat(srow_values...) #horizontal concatenation 
end

"""
helper function for nifti #4
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
    voxel_data = nifti_image.raw

    #2 spatial metadata dictionary from nifti file 
    spatial_metadata_keys = ["origin", "orientation", "spacing", "direction"]
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
    metadata_keys = ["nifti_type", "dim_info", "dim", "nifti_intent", "nifti_intent_code", "datatype", "bitpix", "slice_start", "pixdim", "vox_offset", "scl_slope", "scl_inter", "slice_end", "slice_code",
      "xyzt_units", "cal_max", "cal_min", "slice_duration", "toffset", "descrip", "aux_file", "qform_code", "qform_code_name", "sform_code", "sform_code_name", "quatern_b", "quatern_c", "quatern_d",
      "qoffset_x", "qoffset_y", "qoffset_z", "srow_x", "srow_y", "srow_z", "sto_xyz_matrix", "intent_name"]
    metadata_values = ["one", nifti_image_header.dim_info, nifti_image_header.dim, (nifti_image_header.intent_p1, nifti_image_header.intent_p2, nifti_image_header.intent_p3), nifti_image_header.intent_code,
      nifti_image_header.datatype, nifti_image_header.bitpix, nifti_image_header.slice_start, nifti_image_header.pixdim, nifti_image_header.vox_offset, nifti_image_header.scl_slope,
      nifti_image_header.scl_inter, nifti_image_header.slice_end, nifti_image_header.slice_code, nifti_image_header.xyzt_units, nifti_image_header.cal_max, nifti_image_header.cal_min,
      nifti_image_header.slice_duration, nifti_image_header.toffset, (nifti_image_header.descrip, formulate_string(nifti_image_header.descrip)), (nifti_image_header.aux_file, formulate_string(nifti_image_header.aux_file)),
      nifti_image_header.qform_code, formulate_xform_string(nifti_image_header.qform_code), nifti_image_header.sform_code, formulate_xform_string(nifti_image_header.sform_code),
      nifti_image_header.quatern_b, nifti_image_header.quatern_c, nifti_image_header.quatern_d, nifti_image_header.qoffset_x, nifti_image_header.qoffset_y, nifti_image_header.qoffset_z,
      nifti_image_header.srow_x, nifti_image_header.srow_y, nifti_image_header.srow_z, formulate_sto_xyz((nifti_image_header.srow_x, nifti_image_header.srow_y, nifti_image_header.srow_z)), nifti_image_header.intent_name]

    metadata = Dictionaries.Dictionary(metadata_keys, metadata_values)



    #Nifti Files stores data in what coordinate system?
    #1. qform_code = 0 and sform_code = 0
    #2. qform_code = 1 and sform_code = 0
    #3. qform_code = 0 and sform_code = 1
    #4. qform_code = 1 and sform_code = 1
    #5. qform_code = 2 and sform_code = 0
    #6. qform_code = 0 and sform_code = 2
    #7. qform_code = 2 and sform_code = 1
    #8. qform_code = 1 and sform_code = 2
    #9. qform_code = 2 and sform_code = 2
    #10. qform_code = 3 and sform_code = 0
    #11. qform_code = 0 and sform_code = 3
    #12. qform_code = 3 and sform_code = 1
    #13. qform_code = 1 and sform_code = 3
    #14. qform_code = 3 and sform_code = 2
    #15. qform_code = 2 and sform_code = 3
    #16. qform_code = 3 and sform_code = 3
    #17. qform_code = 4 and sform_code = 0
    #18. qform_code = 0 and sform_code = 4
    #19. qform_code = 4 and sform_code = 1
    #20. qform_code = 1 and sform_code = 4
    #21. qform_code = 4 and sform_code = 2
    #22. qform_code = 2 and sform_code = 4
    #23. qform_code = 4 and sform_code = 3
    #24. qform_code = 3 and sform_code = 4
    #25. qform_code = 4 and sform_code = 4




    return [MedImage([nifti_image.raw, nifti_image_header.pixdim[2:4], (nifti_image_header.srow_x[1:3], nifti_image_header.srow_y[1:3], nifti_image_header.srow_z[1:3]), (nifti_image_header.qoffset_x, nifti_image_header.qoffset_y, nifti_image_header.qoffset_z), " ", " "])]

  end
end


function save_image(im::Array{MedImage}, path::String)
  """
  Creating nifti volumes or each medimage object 
  NIfTI.NIVolume() NIfTI.niwrite()
  """
end



#array_of_objects = load_image("../test_data/ScalarVolume_0")
# array_of_objects = load_image("/workspaces/MedImage.jl/MedImage3D/test_data/volume-0.nii.gz")
#println(length(array_of_objects[1].pixel_array))
