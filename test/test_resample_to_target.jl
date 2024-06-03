using LinearAlgebra
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
    im_fixed_sitk = sitk.ReadImage(path_nifti_fixed)
    im_moving_sitk = sitk.ReadImage(path_nifti_moving)
    im_resampled_sitk = sitk.Resample(im_moving_sitk, im_fixed_sitk, sitk.Transform(), sitk.sitkLinear, 0.0, im_moving_sitk.GetPixelIDValue())
    sitk.WriteImage(im_resampled_sitk, "/home/jm/projects_new/MedImage.jl/test_data/debug/resampled_sitk.nii.gz")
    # Create some test MedImage objects
    im_fixed = load_image(path_nifti_fixed)
    im_moving = load_image(path_nifti_moving)

    # Resample the image using the Julia implementation
    resampled_julia = resample_to_image(im_fixed, im_moving, Linear_en,0.0)
    create_nii_from_medimage(resampled_julia, "/home/jm/projects_new/MedImage.jl/test_data/debug/resampled_medimage.nii.gz")
    # Resample the image using SimpleITK

    # Compare the two images
    test_object_equality(resampled_julia,im_resampled_sitk)

end


path_nifti_2 = "/home/jm/projects_new/MedImage.jl/test_data/for_resample_target/ct_soft_pat_3_sudy_0.nii.gz"
path_nifti_1 = "/home/jm/projects_new/MedImage.jl/test_data/for_resample_target/pat_2_SUV_sudy_0.nii.gz"

# test_resample_to_spacing(path_nifti)
test_resample_to_target(path_nifti_1,path_nifti_2)