using Pkg
Pkg.add(["DICOM", "NIfTI"])
using DICOM
using NIfTI


path_to_dicom_directory = "/home/hurtbadly/Desktop/julia_stuff/MedImage.jl/MedImage3D/test_data/ScalarVolume_0"
dicom_data_array = DICOM.dcmdir_parse(path_to_dicom_directory)


#println(dicom_data_array[1][tag"PixelData"])
#println(dicom_data[1][tag"SeriesId"])
#println("here starts the second file")
#println(dicom_data_array[2][tag"PixelData"])
#
#
#unique indedx, and series_instanceUID so we know exactly how many medimage objects to create since change in that will denote a change within the spatial meta data 
##one way of doing that

dicom_files_tuples_series_id = []
list_of_unique_series_id = []
dicom_pixel_data_array = []

for (index, dicom_data_file) in enumerate(dicom_data_array)
  push!(dicom_files_tuples_series_id, (index, dicom_data_file[tag"SeriesInstanceUID"]))
  push!(dicom_pixel_data_array, dicom_data_file[tag"PixelData"])
end


println(list_of_unique_series_id)









#list_of_unique_dicom_files = [(index,series_id) for (index,series_id) in list_of_unique_dicom_files if series_id in list_of_unique_dicom_files_set]
#println(list_of_unique_dicom_files_set)
#

