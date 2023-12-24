using Pkg
Pkg.add(["DICOM","NIfTI","Dictionaries"])
using DICOM, NIfTI, Dictionaries
include("./MedImage_data_struct.jl")
####################################helper functions###########################################################
function load_values_from_dicom_directory(path::String)
  """
  helper function for dicom #1
  returns an array of unique SERIES INSTANCE UID within dicom files within a dicom directory
  """
  function unique_series_id_within_dicom_files(dicom_data_array)
    series_ids = []
    for dicom_file_data in dicom_data_array
      push!(series_ids, dicom_file_data[tag"SeriesInstanceUID"])
    end  
    return Set(series_id for series_id in series_ids)
  end

  """
  helper function for dicom #2
  returns an array of pixel data for unique ids within dicom files
  """
  function unique_series_id_pixel_data_array(unique_series_id_array,dicom_data_array)
    pixel_data_array = []
    for series_id in unique_series_id_array
      unique_series_pixel_data_array = []
      for dicom_file_data in dicom_data_array
      if dicom_file_data[tag"SeriesInstanceUID"] == series_id
          push!(unique_series_pixel_data_array, [dicom_file_data[tag"PixelData"]])
      end
      end
      push!(pixel_data_array,unique_series_pixel_data_array)
    end
    return pixel_data_array
  end

  """
  helper function for dicom #3
  returns an array of dicom data with unique series instance ID
  """
  function unique_series_id_dicom_data(unique_series_id_array, dicom_data_array)
  unique_series_id_dicom_data_array = []
  for series_id in unique_series_id_array
    for dicom_data in dicom_data_array
      if dicom_data[tag"SeriesInstanceUID"] == series_id 
          push!(unique_series_id_dicom_data_array,dicom_data)
          break
      end
    end
  end
  return unique_series_id_dicom_data_array
  end

  values = []
  dicom_data_array = DICOM.dcmdir_parse(path)
  unique_series_instance_id_within_dicom_files = unique_series_id_within_dicom_files(dicom_data_array)
  pixel_data_array = unique_series_id_pixel_data_array(unique_series_instance_id_within_dicom_files,
                                                       dicom_data_array)
  unique_series_id_dicom_data_array = unique_series_id_dicom_data(unique_series_instance_id_within_dicom_files,
                                                                  dicom_data_array)
  for (index,pixel_data) in enumerate(pixel_data_array) 
    sub_value=[]
    append!(sub_value,[
      pixel_data,
      unique_series_id_dicom_data_array[index][tag"ImageOrientationPatient"],
      unique_series_id_dicom_data_array[index][tag"PixelSpacing"],
      unique_series_id_dicom_data_array[index][tag"ImagePositionPatient"],
      unique_series_id_dicom_data_array[index][tag"ImagePositionPatient"],
      "",
      unique_series_id_dicom_data_array[index][tag"PatientID"]
    ])
    push!(values,sub_value)
  end
  return values
end

function load_values_from_nifti_file(path::String)
  values = []
  sub_value = []
  nifti_image = NIfTI.niread(path)
  nifti_image_header = nifti_image.header
  append!(sub_value,[
    nifti_image.raw,
    (nifti_image_header.srow_x[1:3],nifti_image_header.srow_y[1:3],nifti_image_header.srow_z[1:3]),
    nifti_image_header.pixdim[2:4],
    (nifti_image_header.srow_x, nifti_image_header.srow_y, nifti_image_header.srow_z),
    (nifti_image_header.qoffset_x, nifti_image_header.qoffset_y, nifti_image_header.qoffset_z),
    "",
    ""
  ])
  push!(values, sub_value)
  return values
end
function create_medimage_object_array(values)
  properties = ["pixel_array","direction","spacing","orientation","origin","date_of_saving","patient_id"]
  medimage_object_array = []

  for value_array in values
    medimage_object = MedImage(Dictionaries.Dictionary(properties,value_array))
    push!(medimage_object_array,medimage_object)
  end
  return medimage_object_array
end

##################################################functions defined by Jakub##################################
function load_image(path::String)::Array{MedImage}
  if isdir(path)
    values_array = load_values_from_dicom_directory(path)
    array_of_medimage_objects = create_medimage_object_array(values_array)
    return array_of_medimage_objects
  else
    values_array = load_values_from_nifti_file(path)
    array_of_medimage_objects = create_medimage_object_array(values_array)
    return array_of_medimage_objects
  end
end


function save_image(im::Array{MedImage},path::String)
nothing
end



array_of_objects = load_image("../test_data/ScalarVolume_0")
println(length(array_of_objects[1].pixel_array))









