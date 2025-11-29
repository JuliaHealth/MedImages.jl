using Test
using LinearAlgebra
using PyCall
using MedImages

function create_nii_from_medimage_resample(med_image, file_path::String)
    sitk = pyimport("SimpleITK")
    np = pyimport("numpy")
    voxel_data_np = np.array(med_image.voxel_data)
    image_sitk = sitk.GetImageFromArray(voxel_data_np)
    image_sitk.SetOrigin(med_image.origin)
    image_sitk.SetSpacing(med_image.spacing)
    image_sitk.SetDirection(med_image.direction)
    sitk.WriteImage(image_sitk, file_path * ".nii.gz")
end

function test_resample_to_target_suite(path_nifti_fixed, path_nifti_moving)
    @testset "Resample to Target Test Suite" begin
        sitk = pyimport("SimpleITK")
        
        @testset "Resample moving to fixed image space" begin
            @test begin
                # Load images using SimpleITK for reference
                im_fixed_sitk = sitk.ReadImage(path_nifti_fixed)
                im_moving_sitk = sitk.ReadImage(path_nifti_moving)
                
                # SimpleITK implementation
                im_resampled_sitk = sitk.Resample(im_moving_sitk, im_fixed_sitk, 
                                                sitk.Transform(), sitk.sitkLinear, 
                                                0.0, im_moving_sitk.GetPixelIDValue())
                
                # Load images using MedImages
                im_fixed = MedImages.load_image(path_nifti_fixed, "CT")
                im_moving = MedImages.load_image(path_nifti_moving, "CT")
                
                # MedImages implementation
                resampled_julia = MedImages.resample_to_image(im_fixed, im_moving, 
                                                            MedImages.Linear_en, 0.0)
                
                # Save debug outputs
                debug_dir = joinpath(dirname(@__FILE__), "..", "test_data", "debug")
                mkpath(debug_dir)
                sitk.WriteImage(im_resampled_sitk, "$(debug_dir)/resampled_sitk.nii.gz")
                create_nii_from_medimage_resample(resampled_julia, "$(debug_dir)/resampled_medimage")
                
                # Compare the results
                test_object_equality(resampled_julia, im_resampled_sitk)
                true
            end
        end
    end
end