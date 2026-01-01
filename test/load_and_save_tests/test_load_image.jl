# test/load_and_save_tests/test_load_image.jl
# Tests for load_image function from Load_and_save module

using Test
using MedImages

# Import test infrastructure
include(joinpath(@__DIR__, "..", "test_helpers.jl"))
include(joinpath(@__DIR__, "..", "test_config.jl"))
using .TestHelpers
using .TestConfig

@testset "load_image Tests" begin
    with_temp_output_dir("load_and_save", "load_image") do output_dir

        @testset "Load NIfTI file as CT" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found: $TEST_NIFTI_FILE"
                return
            end

            med_im = MedImages.load_image(TEST_NIFTI_FILE, "CT")

            @test med_im isa MedImages.MedImage_data_struct.MedImage
            @test !isempty(med_im.voxel_data)
            @test med_im.image_type == MedImages.MedImage_data_struct.CT_type
        end

        @testset "Load NIfTI file as PET" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = MedImages.load_image(TEST_NIFTI_FILE, "PET")

            @test med_im isa MedImages.MedImage_data_struct.MedImage
            @test !isempty(med_im.voxel_data)
            @test med_im.image_type == MedImages.MedImage_data_struct.PET_type
        end

        @testset "Voxel data dimensions" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = MedImages.load_image(TEST_NIFTI_FILE, "CT")

            @test ndims(med_im.voxel_data) == 3
            @test size(med_im.voxel_data, 1) > 0
            @test size(med_im.voxel_data, 2) > 0
            @test size(med_im.voxel_data, 3) > 0
        end

        @testset "Spatial metadata is populated" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = MedImages.load_image(TEST_NIFTI_FILE, "CT")

            @test length(med_im.origin) == 3
            @test length(med_im.spacing) == 3
            @test length(med_im.direction) == 9

            # Spacing should be positive
            @test all(s -> s > 0, med_im.spacing)
        end

        @testset "Load synthetic small file" begin
            if !isfile(TEST_SYNTHETIC_FILE)
                @test_skip "Synthetic test file not found"
                return
            end

            med_im = MedImages.load_image(TEST_SYNTHETIC_FILE, "CT")

            @test med_im isa MedImages.MedImage_data_struct.MedImage
            @test !isempty(med_im.voxel_data)
        end

        @testset "Load DICOM directory" begin
            if !isdir(TEST_DICOM_DIR)
                @test_skip "DICOM test directory not found"
                return
            end

            med_im = MedImages.load_image(TEST_DICOM_DIR, "CT")

            @test med_im isa MedImages.MedImage_data_struct.MedImage
            @test !isempty(med_im.voxel_data)
            @test ndims(med_im.voxel_data) == 3
        end
    end
end

@testset "load_image Error Handling" begin
    with_temp_output_dir("load_and_save", "load_errors") do output_dir

        @testset "Non-existent file" begin
            @test_throws Exception MedImages.load_image("/nonexistent/path/file.nii.gz", "CT")
        end
    end
end
