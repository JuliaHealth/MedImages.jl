using Pkg
Pkg.add(["DICOM", "NIfTI"])
using DICOM
using NIfTI


path_to_dicom_directory = "/home/hurtbadly/Desktop/julia_stuff/MedImage.jl/MedImage3D/test_data/ScalarVolume_0"
dicom_data_array = DICOM.dcmdir_parse(path_to_dicom_directory)

function getPixelData(dicom_data_array)
  return only(dicom_data_array).PixelData
end


#println(dicom_data_array[1][tag"PixelData"])
#println(dicom_data[1][tag"SeriesId"])
#println("here starts the second file")
#println(dicom_data_array[2][tag"PixelData"])
#
#
#unique indedx, and series_instanceUID so we know exactly how many medimage objects to create since change in that will denote a change within the spatial meta data 
##one way of doing that
function get_pixel_data(unique_series_ids, dicom_data_array)
  if length(dicom_data_array) == 1
    return only(dicom_data_array).PixelData
  else
    return cat([dcm.PixelData for dcm in dicom_data_array]...; dims=3)
  end

end
function unique_id(dicom_data_array)
  return map(dicom_data->dicom_data.SeriesInstanceUID.dicom_data_array) |> Set |> collect
end
function load_image(path, dicom_data_array)
  path = "/home/hurtbadly/Desktop/julia_stuff/MedImage.jl/MedImage3D/test_data/ScalarVolume_0"
  dicom_data_array = DICOM.dcmdir_parse(path)
  unique_series_id = unique_id(dicom_data_array)
  return map(series_id->filter(dcm->dcm.SeriesInstanceUID == series_id,dicom_data_array),unique_series_id) |>dicom_data_array_with_similar_id
end

dicom_files_tuples_series_id = []
unique_series_id = dicom_data_array[1][tag"SeriesInstanceUID"]
dicom_pixel_data_array = getPixelData(dicom_data_array)
println(dicom_pixel_data_array)

"""


for (index, dicom_data_file) in enumerate(dicom_data_array)
  #push!(dicom_files_tuples_series_id, (index, dicom_data_file[tag"SeriesInstanceUID"]))
  if dicom_data_file[tag"SeriesInstanceUID"]  == unique_series_id 
  push!(dicom_pixel_data_array, dicom_data_file[tag"PixelData"])
end
end


size_of_matrix = size(dicom_pixel_data_array[1])
println(dicom_pixel_data_array[1][1:10])
println(size_of_matrix)




"""






#list_of_unique_dicom_files = [(index,series_id) for (index,series_id) in list_of_unique_dicom_files if series_id in list_of_unique_dicom_files_set]
#println(list_of_unique_dicom_files_set)
#
