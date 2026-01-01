# test/medimage_data_struct_tests/test_medimage_struct.jl
# Tests for MedImage struct from MedImage_data_struct module

using Test
using MedImages

# Import test infrastructure
include(joinpath(@__DIR__, "..", "test_helpers.jl"))
include(joinpath(@__DIR__, "..", "test_config.jl"))
using .TestHelpers
using .TestConfig

@testset "MedImage Struct Tests" begin
    with_temp_output_dir("medimage_data_struct", "medimage_struct") do output_dir

        @testset "Load and inspect MedImage" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found: $TEST_NIFTI_FILE"
                return
            end

            med_im = load_test_image()

            @test med_im isa MedImages.MedImage_data_struct.MedImage

            # Check voxel data
            @test !isempty(med_im.voxel_data)
            @test ndims(med_im.voxel_data) == 3

            # Check spatial metadata
            @test length(med_im.origin) == 3
            @test length(med_im.spacing) == 3
            @test length(med_im.direction) == 9

            # Spacing should be positive
            @test all(s -> s > 0, med_im.spacing)
        end

        @testset "MedImage field types" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = load_test_image()

            @test med_im.origin isa Tuple{Float64,Float64,Float64}
            @test med_im.spacing isa Tuple{Float64,Float64,Float64}
            @test med_im.direction isa NTuple{9,Float64}
        end

        @testset "Image type enum" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im_ct = MedImages.load_image(TEST_NIFTI_FILE, "CT")
            @test med_im_ct.image_type == MedImages.MedImage_data_struct.CT_type

            med_im_pet = MedImages.load_image(TEST_NIFTI_FILE, "PET")
            @test med_im_pet.image_type == MedImages.MedImage_data_struct.PET_type
        end

        @testset "Direction matrix validity" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = load_test_image()

            # Direction should not contain NaN or Inf
            @test all(!isnan, med_im.direction)
            @test all(!isinf, med_im.direction)

            # Direction matrix should be orthogonal (columns should be unit vectors)
            dir = reshape(collect(med_im.direction), 3, 3)
            for col in eachcol(dir)
                norm = sqrt(sum(col .^ 2))
                @test isapprox(norm, 1.0; atol=0.01)
            end
        end

        @testset "Metadata fields exist" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = load_test_image()

            # Check that metadata fields are accessible
            @test hasfield(typeof(med_im), :patient_id)
            @test hasfield(typeof(med_im), :study_uid)
            @test hasfield(typeof(med_im), :series_uid)
            @test hasfield(typeof(med_im), :image_type)
            @test hasfield(typeof(med_im), :image_subtype)
        end
    end
end

@testset "MedImage Mutability Tests" begin
    with_temp_output_dir("medimage_data_struct", "mutability") do output_dir
        if !isfile(TEST_NIFTI_FILE)
            @test_skip "Test file not found"
            return
        end

        med_im = load_test_image()

        @testset "Voxel data can be modified" begin
            original_value = med_im.voxel_data[1, 1, 1]
            med_im.voxel_data[1, 1, 1] = original_value + 100

            @test med_im.voxel_data[1, 1, 1] == original_value + 100

            # Restore
            med_im.voxel_data[1, 1, 1] = original_value
        end
    end
end
