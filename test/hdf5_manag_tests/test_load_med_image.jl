# test/hdf5_manag_tests/test_load_med_image.jl
# Tests for load_med_image function from HDF5_manag module

using Test
using MedImages

# Try to import HDF5 - tests will be skipped if not available
HDF5_AVAILABLE = try
    using HDF5
    true
catch
    @warn "HDF5.jl not available - HDF5 tests will be skipped"
    false
end

# Import test infrastructure
include(joinpath(@__DIR__, "..", "test_helpers.jl"))
include(joinpath(@__DIR__, "..", "test_config.jl"))
using .TestHelpers
using .TestConfig

@testset "load_med_image Tests" begin
    if !HDF5_AVAILABLE
        @test_skip "HDF5.jl not available"
        return
    end

    if !isfile(TEST_NIFTI_FILE)
        @test_skip "Test file not found: $TEST_NIFTI_FILE"
        return
    end

    with_temp_output_dir("hdf5_manag", "load_med_image") do output_dir
        med_im = load_test_image()

        @testset "Basic Load" begin
            h5_path = joinpath(output_dir, "test_load.h5")

            # Save first
            f = h5open(h5_path, "w")
            uid = MedImages.save_med_image(f, "test_image", med_im)
            close(f)

            # Load back
            f = h5open(h5_path, "r")
            med_im_loaded = MedImages.load_med_image(f, "test_image", uid)
            close(f)

            @test !isempty(med_im_loaded.voxel_data)
            @test size(med_im_loaded.voxel_data) == size(med_im.voxel_data)
        end

        @testset "Voxel data preservation" begin
            h5_path = joinpath(output_dir, "test_voxel_preserve.h5")

            f = h5open(h5_path, "w")
            uid = MedImages.save_med_image(f, "test_image", med_im)
            close(f)

            f = h5open(h5_path, "r")
            med_im_loaded = MedImages.load_med_image(f, "test_image", uid)
            close(f)

            @test med_im.voxel_data == med_im_loaded.voxel_data
        end

        @testset "Metadata preservation" begin
            h5_path = joinpath(output_dir, "test_metadata.h5")

            f = h5open(h5_path, "w")
            uid = MedImages.save_med_image(f, "test_image", med_im)
            close(f)

            f = h5open(h5_path, "r")
            med_im_loaded = MedImages.load_med_image(f, "test_image", uid)
            close(f)

            @test med_im.spacing == med_im_loaded.spacing
            @test med_im.origin == med_im_loaded.origin
            @test med_im.direction == med_im_loaded.direction
        end

        @testset "Roundtrip integrity" begin
            h5_path = joinpath(output_dir, "test_roundtrip.h5")

            # Save
            f = h5open(h5_path, "w")
            uid = MedImages.save_med_image(f, "roundtrip", med_im)
            close(f)

            # Load
            f = h5open(h5_path, "r")
            med_im_loaded = MedImages.load_med_image(f, "roundtrip", uid)
            close(f)

            # Full comparison
            @test med_im.voxel_data == med_im_loaded.voxel_data
            @test med_im.spacing == med_im_loaded.spacing
            @test med_im.origin == med_im_loaded.origin
            @test med_im.direction == med_im_loaded.direction
        end
    end
end

@testset "load_med_image Multiple Images" begin
    if !HDF5_AVAILABLE
        @test_skip "HDF5.jl not available"
        return
    end

    if !isfile(TEST_NIFTI_FILE)
        @test_skip "Test file not found: $TEST_NIFTI_FILE"
        return
    end

    with_temp_output_dir("hdf5_manag", "load_multiple") do output_dir
        med_im = load_test_image()

        @testset "Load specific image from multiple" begin
            h5_path = joinpath(output_dir, "test_multi.h5")

            # Save multiple images
            f = h5open(h5_path, "w")
            uid1 = MedImages.save_med_image(f, "images", med_im)
            uid2 = MedImages.save_med_image(f, "images", med_im)
            close(f)

            # Load first image
            f = h5open(h5_path, "r")
            loaded1 = MedImages.load_med_image(f, "images", uid1)
            close(f)

            # Load second image
            f = h5open(h5_path, "r")
            loaded2 = MedImages.load_med_image(f, "images", uid2)
            close(f)

            # Both should have correct data
            @test size(loaded1.voxel_data) == size(med_im.voxel_data)
            @test size(loaded2.voxel_data) == size(med_im.voxel_data)
        end
    end
end

@testset "load_med_image Type Preservation" begin
    if !HDF5_AVAILABLE
        @test_skip "HDF5.jl not available"
        return
    end

    if !isfile(TEST_NIFTI_FILE)
        @test_skip "Test file not found: $TEST_NIFTI_FILE"
        return
    end

    with_temp_output_dir("hdf5_manag", "load_types") do output_dir
        @testset "Image type preserved" begin
            med_im = MedImages.load_image(TEST_NIFTI_FILE, "CT")
            h5_path = joinpath(output_dir, "test_type.h5")

            f = h5open(h5_path, "w")
            uid = MedImages.save_med_image(f, "ct", med_im)
            close(f)

            f = h5open(h5_path, "r")
            loaded = MedImages.load_med_image(f, "ct", uid)
            close(f)

            @test loaded.image_type == med_im.image_type
        end
    end
end
