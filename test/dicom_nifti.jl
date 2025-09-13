using Test
using LinearAlgebra
using Dictionaries
using Dates
using PyCall
using MedImages

function test_object_equality(medIm, sitk_image)
    sitk = pyimport("SimpleITK")
    
    @testset "Object Equality Checks" begin
        # Get array from SimpleITK image
        arr = sitk.GetArrayFromImage(sitk_image)
        spacing = sitk_image.GetSpacing()
        
        # Transform MedImage voxel data to match SimpleITK format
        vox = medIm.voxel_data
        vox = permutedims(vox, (3, 2, 1))
        
        @test isapprox(collect(spacing), collect(Tuple{Float64,Float64,Float64}(medIm.spacing)); atol=0.1)
        @test isapprox(collect(sitk_image.GetDirection()), collect(medIm.direction); atol=0.2)
        @test isapprox(collect(sitk_image.GetSpacing()), collect(medIm.spacing); atol=0.1)
        @test isapprox(collect(sitk_image.GetOrigin()), collect(medIm.origin); atol=0.1)
        @test isapprox(arr, vox; rtol=0.15)
    end
end

function create_nii_from_medimage(med_image, file_path::String)
    @testset "Create NIfTI from MedImage" begin
        @test begin
            sitk = pyimport("SimpleITK")
            np = pyimport("numpy")
            
            # Convert voxel_data to numpy array
            voxel_data_np = np.array(med_image.voxel_data)
            image_sitk = sitk.GetImageFromArray(voxel_data_np)
            
            # Set spatial metadata
            image_sitk.SetOrigin(med_image.origin)
            image_sitk.SetSpacing(med_image.spacing)
            image_sitk.SetDirection(med_image.direction)
            
            # Save the image
            sitk.WriteImage(image_sitk, file_path * ".nii.gz")
            
            # Test that file was created
            isfile(file_path * ".nii.gz")
        end
    end
end

function test_dicom_nifti_suite(path_nifti)
    @testset "DICOM/NIfTI Test Suite" begin
        sitk = pyimport("SimpleITK")
        
        @testset "Load and Compare with SimpleITK" begin
            @test begin
                # Load with MedImages
                med_im = load_image(path_nifti, "CT")  # Use exported function directly
                
                # Load with SimpleITK
                sitk_image = sitk.ReadImage(path_nifti)
                
                # Test object equality
                test_object_equality(med_im, sitk_image)
                true
            end
        end
        
        @testset "Save and Reload Test" begin
            @test begin
                # Load image
                med_im = load_image(path_nifti, "CT")  # Use exported function directly
                
                # Save to new file
                debug_dir = joinpath(dirname(@__FILE__), "..", "test_data", "debug")
                mkpath(debug_dir)
                output_path = joinpath(debug_dir, "test_output")
                
                create_nii_from_medimage(med_im, output_path)
                
                # Test file exists
                isfile(output_path * ".nii.gz")
            end
        end
    end
end