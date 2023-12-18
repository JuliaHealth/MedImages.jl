using NIfTI,LinearAlgebra,DICOM
using .dicom_nifti


"""
test if the rotation of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function
    
"""
function test_rotation(path_nifti)
    
    # Load the image from path
    nifti_img = niread(path_nifti)
    #



end