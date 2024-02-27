using NIfTI,LinearAlgebra,DICOM
using .dicom_nifti


"""
test if the resample_to_spacing of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function
we nned to check the nearest neuhnbor interpolation and b spline interpolation
"""
function test_resample_to_spacing(path_nifti)
    
    # Load the image from path
    med_im=load_image(path_nifti)
    sitk.ReadImage(path_nifti)


    
    TODO()
    resample_to_spacing=[[]]
    #rotate the image


    #save both images into nifti files to temporary folder
    

    test_image_equality(path_a,path_b)


end


"""
test if the resample_to_spacing of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function
We need to check can it change between RAS and LPS orientationas those are most common
"""
function test_change_orientation(path_nifti)
    
    # Load the image from path
    med_im=load_image(path_nifti)
    sitk.ReadImage(path_nifti)


    
    TODO()
    resample_to_spacing=[[]]
    #rotate the image


    #save both images into nifti files to temporary folder
    

    test_image_equality(path_a,path_b)


end