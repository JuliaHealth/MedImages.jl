# test/load_and_save_tests/test_create_nii_from_medimage.jl
# Tests for create_nii_from_medimage function from Load_and_save module

using Test
using PyCall
using MedImages

# Import test infrastructure (conditionally include if not already defined)
if !isdefined(@__MODULE__, :TestHelpers)
    include(joinpath(@__DIR__, "..", "test_helpers.jl"))
    include(joinpath(@__DIR__, "..", "test_config.jl"))
end
using .TestHelpers
using .TestConfig

@testset "create_nii_from_medimage Tests" begin
    sitk = pyimport("SimpleITK")

    with_temp_output_dir("load_and_save", "create_nii") do output_dir

        @testset "Create NIfTI from loaded image" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = load_test_image()
            output_path = joinpath(output_dir, "test_output.nii.gz")

            MedImages.create_nii_from_medimage(med_im, output_path)

            @test isfile(output_path)
        end

        @testset "Created file can be read by SimpleITK" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = load_test_image()
            output_path = joinpath(output_dir, "readable.nii.gz")

            MedImages.create_nii_from_medimage(med_im, output_path)

            # Read with SimpleITK
            sitk_image = sitk.ReadImage(output_path)

            @test sitk_image.GetSize() == reverse(size(med_im.voxel_data))
        end

        @testset "Metadata preserved in output" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = load_test_image()
            output_path = joinpath(output_dir, "metadata.nii.gz")

            MedImages.create_nii_from_medimage(med_im, output_path)

            sitk_image = sitk.ReadImage(output_path)

            @test isapprox(collect(sitk_image.GetSpacing()), collect(med_im.spacing); atol=0.1)
            @test isapprox(collect(sitk_image.GetOrigin()), collect(med_im.origin); atol=0.1)
        end

        @testset "Output matches original dimensions" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = load_test_image()
            output_path = joinpath(output_dir, "dimensions.nii.gz")

            MedImages.create_nii_from_medimage(med_im, output_path)

            sitk_image = sitk.ReadImage(output_path)
            sitk_size = sitk_image.GetSize()
            med_im_size = size(med_im.voxel_data)

            @test sitk_size[1] == med_im_size[1]
            @test sitk_size[2] == med_im_size[2]
            @test sitk_size[3] == med_im_size[3]
        end
    end
end
