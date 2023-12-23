using Pkg
Pkg.add(["DICOM", "NIfTI", "Dictionaries"])

using DICOM, NIfTI, Dictionaries

include("./MedImage_data_struct.jl")

"""
functions to load and save 3D images from dicom or nifti files
and save them into nifti
"""

"""
given a path to a Nifti file or a dicom folder return a MedImage object
first it needs to recognise weather it works with a dicom folder or a nifti file
then it will load the image and return a MedImage object with the image data and the metadata
"""
function load_image(path::String)::Array{MedImage}

function handle_single_medimage_object_from_dicom(dicom_data_array, sub_values, path, list_of_dicom_files)
  # we are retreiving data from the first dicom files, since all the other files have same Series ID, so spatial meta data wont change
  append!(sub_values, [
    dicom_data_array[1][tag"PixelData"],
    dicom_data_array[1][tag"ImageOrientationPatient"],
    dicom_data_array[1][tag"PixelSpacing"],
    dicom_data_array[1][tag"ImagePositionPatient"],
    dicom_data_array[1][tag"ImagePositionPatient"],
    joinpath(path, list_of_dicom_files[1]),
    dicom_data_array[1][tag"PatientID"]
  ])
end

function handle_multiple_medimage_objects_from_dicom(dicom_data_array,values,sub_values,path,list_of_dicom_files,dicom_tag_series_instance_uid_tuples, pixel_data_array)
  list_of_unique_series_id_tag = Set(tuple[3] for tuple in dicom_tag_series_instance_uid_tuples) 
  for series_id in list_of_unique_series_id_tag
    selected_tuple = ()
    for tuple in dicom_tag_series_instance_uid_tuples
      if tuple[3] == series_id
        selected_tuple = tuple
      end
    end
    append!(sub_values,[
      pixel_data_array,
      dicom_data_array[selected_tuple[1]][tag"ImageOrientationPatient"],
      dicom_data_array[selected_tuple[1]][tag"PixelSpacing"],
      dicom_data_array[selected_tuple[1]][tag"ImagePositionPatient"],
      dicom_data_array[selected_tuple[1]][tag"ImagePositionPatient"],
      joinpath(path,list_of_dicom_files[selected_tuple[1]]),
      dicom_data_array[selected_tuple[1]][tag"PatientID"]
    ]) 
    push!(values,sub_values)
  end
end




function load_image(path::String)::Union{MedImage, Vector{MedImage}}

  #check if the path is a folder or a file
  properties = ["pixel_array", "direction", "spacing", "orientation", "origin", "date_of_saving", "patient_id"]
  values = Vector{Any}()
  sub_values = Vector{Any}()

  if isdir(path)
    #load the dicom folder

    #dicom directory consists up of multiple dicom files with same metadata , but different pixel array data since each image is a 2d layer, whereas a nifti image  deals with 3d data
    list_of_dicom_files = readdir(path)
    dicom_data_array = DICOM.dcmdir_parse(path)
    #only accessing the first file in the dicom folder for now 
    dicom_tag_series_instance_uid_tuples = []
    list_of_unique_dicom_series_id_tag = []
    dicom_pixel_data_array = [] #contains array of 2d pixel data array
    dicom_data_file = dicom_data_array[1]

    for (index, dicom_data_file) in enumerate(dicom_data_array)
      push!(dicom_tag_series_instance_uid_tuples, (index, dicom_data_file[tag"SeriesNumber"], dicom_data_file[tag"SeriesInstanceUID"]))
      push!(dicom_pixel_data_array, dicom_data_file[tag"PixelData"])
    end


    if length(list_of_unique_dicom_series_id_tag) == 1
      handle_single_medimage_object_from_dicom(dicom_data_array, sub_values, path, list_of_dicom_files)
      push!(values,sub_values)
    else
      handle_multiple_medimage_objects_from_dicom(dicom_data_array,values,sub_values,path,list_of_dicom_files,dicom_tag_series_instance_uid_tuples,dicom_pixel_data_array)
    end
    #tags refernce from below
    #https://github.com/JuliaHealth/DICOM.jl/blob/master/src/dcm_dict.jl
    #apparently a lot of the tags within in the above are not within the dicom file dict
    #more accurate one 
    #https://towardsdatascience.com/dealing-with-dicom-using-imageio-python-package-117f1212ab82



    """
    list_of_dicom_files = readdir(path)  
    #assuming that dicom folder contains unzrchived data
    if len(list_of_dicom_files) != 0 && (list_of_dicom_files[1] !== "." || list_of_dicom_files[1] !== "..")
    file = open(joinpath(path,list_of_dicom_files[1]),"r")
    dicom_file = DICOM>dcm_parse(file)
    end
    """


  else
    #dealing and handling nifti files 
    #load the nifti file
    nifti_image = NIfTI.niread(path)
    header = nifti_image.header

    
    append!(sub_values, [nifti_image.raw,
      (header.srow_x[1:3], header.srow_y[1:3], header.srow_z[1:3]),
      header.pixdim[2:4],
      (header.srow_x, header.srow_y, header.srow_z),
      (header.qoffset_x, header.qoffset_y, header.qoffset_z),
      path,
      ""])
    push!(values,sub_values)
  end
  
  if length(values) == 1
    MedImage_struct_attributes = Dictionaries.Dictionary(properties, values[1])
  return MedImage(MedImage_struct_attributes)
  else
    medimage_object_array = []
    for value_array in values
      MedImage_struct_attributes = Dictionaries.Dictionary(properties,value_array)
      push!(medimage_object_array, MedImage(MedImage_struct_attributes))
    end
    return medimage_object_array
  end
end#load_image

"""
given a MedImage objects and a path to a nifti file save the MedImage object into a nifti file

"""

function save_image(im::Array{MedImage}, path::String)
  
end#save_image


