using Pkg
Pkg.add(["DICOM","NIfTI","Dictionaries"])
using DICOM, NIfTI, Dictionaries
include("./MedImage_data_struct.jl")
"""
helper function for dicom #1
returns an array of unique SERIES INSTANCE UID within dicom files within a dicom directory
"""
function unique_series_id_within_dicom_files(dicom_data_array)
  return map(dicom_file_data->dicom_file_data[tag"SeriesInstanceUID"],dicom_data_array)|>
        Set|>
        collect
end

"""
helper function for dicom #2
returns an array of pixel data for unique ids within dicom files
"""
function get_pixel_data(dicom_data_array)
  print("ddddd dicom_data_array $(length(dicom_data_array)))")
  if length(dicom_data_array) == 1
    #in case e have 2D image
    return only(dicom_data_array).PixelData
  else
      #in case we have 3D image
      return cat([dcm.PixelData for dcm in dicom_data_array]...; dims = 3)
  end
end


function load_image(path::String)::Array{MedImage}
  if isdir(path)
    dicom_data_array= DICOM.dcmdir_parse(path)
    return unique_series_id_within_dicom_files(dicom_data_array)|>
    series_ids->map(series_id->filter(dcm -> dcm.SeriesInstanceUID == series_id, dicom_data_array),series_ids)|> #now we have series of dicom files with the same series id
    dcom_data_locs->map(dcom_data_loc-> MedImage(get_pixel_data(dcom_data_loc),
                            dcom_data_loc[1].ImageOrientationPatient,
                            dcom_data_loc[1].PixelSpacing,
                            dcom_data_loc[1].ImagePositionPatient,
                            dcom_data_loc[1].ImagePositionPatient,
                            " ",#dcom_data_loc[1].StudyCompletionDate
                            dcom_data_loc[1].PatientID),dcom_data_locs)
  else
    nifti_image = NIfTI.niread(path)
    nifti_image_header = nifti_image.header
    return [MedImage(nifti_image.raw
    ,(nifti_image_header.srow_x[1:3],nifti_image_header.srow_y[1:3],nifti_image_header.srow_z[1:3])
    ,nifti_image_header.pixdim[2:4]
    ,(nifti_image_header.srow_x, nifti_image_header.srow_y, nifti_image_header.srow_z)
    ,(nifti_image_header.qoffset_x, nifti_image_header.qoffset_y, nifti_image_header.qoffset_z)
    ," "
    ," ")]
    
  end
end


function save_image(im::Array{MedImage},path::String)
nothing
end



array_of_objects = load_image("../test_data/ScalarVolume_0")
println(length(array_of_objects[1].pixel_array))
