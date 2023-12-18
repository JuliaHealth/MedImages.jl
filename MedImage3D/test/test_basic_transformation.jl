using NIfTI,LinearAlgebra,DICOM
using .dicom_nifti


"""
test if the rotation of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function

"""
function test_rotation(path_nifti)
    
    # Load the image from path
    med_im=load_image(path_nifti)
    sitk.ReadImage(path_nifti)


    #series of values to rotate the image by
    
    TODO()
    rotation_vals=[[]]
    #rotate the image


    #save both images into nifti files to temporary folder
    

    test_image_equality(path_a,path_b)


end

"""
test if the cropping of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function

"""
function test_crop(path_nifti)
    
    # Load the image from path
    med_im=load_image(path_nifti)
    sitk.ReadImage(path_nifti)


    
    TODO()


    #save both images into nifti files to temporary folder
    

    test_image_equality(path_a,path_b)


end


"""
test if the translation of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function

"""
function test_translate(path_nifti)
    
    # Load the image from path
    med_im=load_image(path_nifti)
    sitk.ReadImage(path_nifti)


    
    TODO()


    #save both images into nifti files to temporary folder
    

    test_image_equality(path_a,path_b)


end


"""
test if the scaling of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function

"""
function test_scale(path_nifti)
    
    # Load the image from path
    med_im=load_image(path_nifti)
    sitk.ReadImage(path_nifti)


    
    TODO()


    #save both images into nifti files to temporary folder
    

    test_image_equality(path_a,path_b)


end