using NIfTI,LinearAlgebra,DICOM
using .dicom_nifti
include("../src/Load_and_save.jl")
# include("../src/Basic_transformations.jl")
# include("./test_visualize.jl")
include("./dicom_nifti.jl")
include("../src/Resample_to_target.jl")

"""
test if the resample_to_target of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function
We will have two images as the inputs one would be translated and have diffrent spacing than the other
    both images are from the same root image (just artificially translated and resampled) so after resampling
    to target they should basically give exactly the same image
"""
function test_resample_to_target(path_nifti_fixed,path_nifti_moving)
    

    # Load SimpleITK
    sitk = pyimport_conda("SimpleITK", "simpleitk")

    # Create some test MedImage objects
    im_fixed = MedImage(...)
    im_moving = MedImage(...)

    # Resample the image using the Julia implementation
    resampled_julia = resample_to_image(im_fixed, im_moving, Interpolator.linear)

    # Resample the image using SimpleITK
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_sitk = resampler.Execute(im_moving)

    # Compare the two images
    test_object_equality(medIm,sitk_trnanslated)

    # Load the image from path
    med_im=load_image(path_nifti)
    sitk.ReadImage(path_nifti)



    
    TODO()
   


end
