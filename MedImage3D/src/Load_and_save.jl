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


function load_image(path::String)::MedImage

  #check if the path is a folder or a file
  properties = ["pixel_array", "direction", "spacing", "orientation", "origin", "date_of_saving", "patient_id"]
  values = Vector{Any}()


  if isdir(path)
    #load the dicom folder

    #due to added confusion only the first file is being accessed from the loaded dicom directory
    list_of_dicom_files = readdir(path)
    dicom_data_array = DICOM.dcmdir_parse(path)
    dicom_data_file = dicom_data_array[1] #only accessing the first file in the dicom folder for now 
    #tags refernce from below
    #https://github.com/JuliaHealth/DICOM.jl/blob/master/src/dcm_dict.jl
    #apparently a lot of the tags within in the above are not within the dicom file dict
    #more accurate one 
    #https://towardsdatascience.com/dealing-with-dicom-using-imageio-python-package-117f1212ab82
    append!(values, [
      dicom_data_file[tag"PizelData"], #pixel data 
      dicom_data_file[tag"ImageOrientationPatient"], #direction
      dicom_data_file[tag"PixelSpacing"], #spacing
      dicom_data_file[tag"ImagePositionPatient"], #orientation, and origin similar?
      dicom_data_file[tag"ImagePositionPatient"], #origin
      joinpath(path, list_of_dicom_files[1]),
      dicom_data_file[tag"PatientID"]
    ])

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

    append!(values, [nifti_image.raw,
      (header.srow_x[1:3], header.srow_y[1:3], header.srow_z[1:3]),
      header.pixdim[2:4],
      (header.srow_x, header.srow_y, header.srow_z),
      (header.qoffset_x, header.qoffset_y, header_qoffset_z),
      path,
      ""])
  end
  MedImage_struct_attributes = Dictionaries.Dictionary(properties, values)
  return MedImage(MedImage_struct_attributes)

end#load_image

"""
given a MedImage object and a path to a nifti file save the MedImage object into a nifti file

"""
function save_image(im::MedImage, path::String)


end#save_image



