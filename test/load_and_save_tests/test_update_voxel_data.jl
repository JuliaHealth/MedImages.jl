# test/load_and_save_tests/test_update_voxel_data.jl
# Tests for update_voxel_data function from Load_and_save module

using Test
using MedImages

# Import test infrastructure (conditionally include if not already defined)
if !isdefined(@__MODULE__, :TestHelpers)
    include(joinpath(@__DIR__, "..", "test_helpers.jl"))
    include(joinpath(@__DIR__, "..", "test_config.jl"))
end
using .TestHelpers
using .TestConfig

@testset "update_voxel_data Tests" begin
    with_temp_output_dir("load_and_save", "update_voxel") do output_dir

        @testset "Update voxel data only" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = load_test_image()
            original_origin = med_im.origin
            original_spacing = med_im.spacing
            original_direction = med_im.direction

            # Create new voxel data with same dimensions
            new_voxel_data = rand(Float32, size(med_im.voxel_data)...)

            updated = MedImages.update_voxel_data(med_im, new_voxel_data)

            # Voxel data should be updated
            @test size(updated.voxel_data) == size(new_voxel_data)

            # Spatial metadata should be preserved
            @test updated.origin == original_origin
            @test updated.spacing == original_spacing
            @test updated.direction == original_direction
        end

        @testset "Different dimensions allowed" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = load_test_image()

            # Create voxel data with different dimensions
            new_voxel_data = rand(Float32, 32, 32, 16)

            updated = MedImages.update_voxel_data(med_im, new_voxel_data)

            @test size(updated.voxel_data) == (32, 32, 16)
        end

        @testset "Original unchanged" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = load_test_image()
            original_size = size(med_im.voxel_data)

            new_voxel_data = rand(Float32, 10, 10, 10)
            _ = MedImages.update_voxel_data(med_im, new_voxel_data)

            @test size(med_im.voxel_data) == original_size
        end

        @testset "Type preservation" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            med_im = load_test_image()

            new_voxel_data = rand(Float64, size(med_im.voxel_data)...)
            updated = MedImages.update_voxel_data(med_im, new_voxel_data)

            @test eltype(updated.voxel_data) == Float64
        end
    end
end
