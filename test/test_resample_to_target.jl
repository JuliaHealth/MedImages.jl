using NIfTI,LinearAlgebra,DICOM
using .dicom_nifti


"""
test if the resample_to_target of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function
We will have two images as the inputs one would be translated and have diffrent spacing than the other
    both images are from the same root image (just artificially translated and resampled) so after resampling
    to target they should basically give exactly the same image
"""
function test_resample_to_target(path_nifti)
    


    # Load the image from path
    med_im=load_image(path_nifti)
    sitk.ReadImage(path_nifti)



    
    TODO()
   


end
