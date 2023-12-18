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
    if isdir(path)
        #load the dicom folder

    else
        #load the nifti file
    end

end#load_image

"""
given a MedImage object and a path to a nifti file save the MedImage object into a nifti file

"""
function save_image(im::MedImage, path::String)

end#save_image

