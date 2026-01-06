# test/load_and_save_tests/test_update_voxel_and_spatial_data.jl
# Tests for update_voxel_and_spatial_data function from Load_and_save module

using Test
using MedImages

# Import test infrastructure (conditionally include if not already defined)
if !isdefined(@__MODULE__, :TestHelpers)
    include(joinpath(@__DIR__, "..", "test_helpers.jl"))
    include(joinpath(@__DIR__, "..", "test_config.jl"))
end
using .TestHelpers
using .TestConfig

@testset "update_voxel_and_spatial_data Tests" begin
    with_temp_output_dir("load_and_save", "update_voxel_spatial") do output_dir

        @testset "Update all spatial data" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = load_test_image()

            # Create new data
            new_voxel_data = rand(Float32, 64, 64, 32)
            new_origin = (10.0, 20.0, 30.0)
            new_spacing = (2.0, 2.0, 3.0)
            new_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

            updated = MedImages.update_voxel_and_spatial_data(
                med_im, new_voxel_data, new_origin, new_spacing, new_direction)

            @test size(updated.voxel_data) == size(new_voxel_data)
            @test updated.origin == new_origin
            @test updated.spacing == new_spacing
            @test updated.direction == new_direction
        end

        @testset "Original image unchanged" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = load_test_image()
            original_size = size(med_im.voxel_data)
            original_origin = med_im.origin

            new_voxel_data = rand(Float32, 32, 32, 16)
            new_origin = (100.0, 100.0, 100.0)
            new_spacing = (1.0, 1.0, 1.0)
            new_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

            _ = MedImages.update_voxel_and_spatial_data(
                med_im, new_voxel_data, new_origin, new_spacing, new_direction)

            # Original should be unchanged
            @test size(med_im.voxel_data) == original_size
            @test med_im.origin == original_origin
        end

        @testset "Metadata preserved" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = load_test_image()

            new_voxel_data = rand(Float32, 32, 32, 32)
            new_origin = (0.0, 0.0, 0.0)
            new_spacing = (1.0, 1.0, 1.0)
            new_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

            updated = MedImages.update_voxel_and_spatial_data(
                med_im, new_voxel_data, new_origin, new_spacing, new_direction)

            # Image type and other metadata should be preserved
            @test updated.image_type == med_im.image_type
        end
    end
end
