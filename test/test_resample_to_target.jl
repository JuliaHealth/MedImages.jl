using LinearAlgebra
include("/workspaces/MedImage.jl/src/Utils.jl")
include("/workspaces/MedImage.jl/test/dicom_nifti.jl")
includet("/workspaces/MedImage.jl/src/Orientation_dicts.jl")
includet("/workspaces/MedImage.jl/src/Spatial_metadata_change.jl")
includet("/workspaces/MedImage.jl/src/Resample_to_target.jl")



function create_nii_from_medimagee(med_image, file_path::String)
    # Convert voxel_data to a numpy array (Assuming voxel_data is stored in Julia array format)
    voxel_data_np = med_image.voxel_data
    voxel_data_np = permutedims(voxel_data_np, (3, 2, 1))
    # Create a SimpleITK image from numpy array
    sitk = pyimport("SimpleITK")
    image_sitk = sitk.GetImageFromArray(voxel_data_np)
  
    # Set spatial metadata
    image_sitk.SetOrigin(med_image.origin)
    image_sitk.SetSpacing(med_image.spacing)
    image_sitk.SetDirection(med_image.direction)
  
    # Save the image as .nii.gz
    sitk.WriteImage(image_sitk, file_path * ".nii.gz")
  end


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
    sitk.WriteImage(im_resampled_sitk, "/workspaces/MedImage.jl/test_data/debug/resampled_sitk.nii.gz")
    # Create some test MedImage objects
    im_fixed = load_image(path_nifti_fixed)
    im_moving = load_image(path_nifti_moving)

    # Resample the image using the Julia implementation
    resampled_julia = resample_to_image(im_fixed, im_moving, Linear_en,0.0)
    create_nii_from_medimagee(resampled_julia, "/workspaces/MedImage.jl/test_data/debug/resampled_medimage.nii.gz")
    # Resample the image using SimpleITK

    # Compare the two images
    test_object_equality(resampled_julia,im_resampled_sitk)

end


path_nifti_2 = "/workspaces/MedImage.jl/test_data/for_resample_target/spleen_34.nii.gz"
path_nifti_1 = "/workspaces/MedImage.jl/test_data/for_resample_target/spleen_35.nii.gz"

# # test_resample_to_spacing(path_nifti)
test_resample_to_target(path_nifti_1,path_nifti_2)