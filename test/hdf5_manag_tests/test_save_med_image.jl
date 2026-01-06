# test/hdf5_manag_tests/test_save_med_image.jl
# Tests for save_med_image function from HDF5_manag module

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

# Import test infrastructure (conditionally include if not already defined)
if !isdefined(@__MODULE__, :TestHelpers)
    include(joinpath(@__DIR__, "..", "test_helpers.jl"))
    include(joinpath(@__DIR__, "..", "test_config.jl"))
end
using .TestHelpers
using .TestConfig

@testset "save_med_image Tests" begin
    if !HDF5_AVAILABLE
        @test_skip "HDF5.jl not available"
        return
    end

    if !isfile(TEST_NIFTI_FILE)
        @test_skip "Test file not found: $TEST_NIFTI_FILE"
        return
    end

    with_temp_output_dir("hdf5_manag", "save_med_image") do output_dir
        med_im = load_test_image()

        @testset "Basic Save" begin
            h5_path = joinpath(output_dir, "test_save.h5")

            f = h5open(h5_path, "w")
            uid = MedImages.save_med_image(f, "test_group", med_im)
            close(f)

            @test isfile(h5_path)
            @test !isempty(uid)
        end

        @testset "Save creates valid HDF5 structure" begin
            h5_path = joinpath(output_dir, "test_structure.h5")

            f = h5open(h5_path, "w")
            uid = MedImages.save_med_image(f, "images", med_im)
            close(f)

            # Verify structure
            f = h5open(h5_path, "r")
            @test haskey(f, "images")
            g = f["images"]
            @test haskey(g, uid)
            close(f)
        end

        @testset "Save with multiple images in same group" begin
            h5_path = joinpath(output_dir, "test_multiple.h5")

            f = h5open(h5_path, "w")
            uid1 = MedImages.save_med_image(f, "images", med_im)
            uid2 = MedImages.save_med_image(f, "images", med_im)
            close(f)

            # UIDs should be different
            @test uid1 != uid2

            # Both should exist
            f = h5open(h5_path, "r")
            g = f["images"]
            @test haskey(g, uid1)
            @test haskey(g, uid2)
            close(f)
        end

        @testset "Save preserves voxel data" begin
            h5_path = joinpath(output_dir, "test_voxel.h5")

            f = h5open(h5_path, "w")
            uid = MedImages.save_med_image(f, "test", med_im)
            close(f)

            # Read back and verify
            f = h5open(h5_path, "r")
            dset = f["test"][uid]
            saved_data = read(dset)
            close(f)

            @test size(saved_data) == size(med_im.voxel_data)
            @test saved_data == med_im.voxel_data
        end

        @testset "Save stores metadata as attributes" begin
            h5_path = joinpath(output_dir, "test_attrs.h5")

            f = h5open(h5_path, "w")
            uid = MedImages.save_med_image(f, "test", med_im)
            close(f)

            f = h5open(h5_path, "r")
            dset = f["test"][uid]

            # Check for key attributes
            @test haskey(attributes(dset), "origin")
            @test haskey(attributes(dset), "spacing")
            @test haskey(attributes(dset), "direction")

            close(f)
        end
    end
end

@testset "save_med_image Different Image Types" begin
    if !HDF5_AVAILABLE
        @test_skip "HDF5.jl not available"
        return
    end

    if !isfile(TEST_NIFTI_FILE)
        @test_skip "Test file not found: $TEST_NIFTI_FILE"
        return
    end

    with_temp_output_dir("hdf5_manag", "save_types") do output_dir
        @testset "CT image type" begin
            med_im = MedImages.load_image(TEST_NIFTI_FILE, "CT")
            h5_path = joinpath(output_dir, "ct_image.h5")

            f = h5open(h5_path, "w")
            uid = MedImages.save_med_image(f, "ct", med_im)
            close(f)

            @test isfile(h5_path)
        end

        @testset "PET image type" begin
            med_im = MedImages.load_image(TEST_NIFTI_FILE, "PET")
            h5_path = joinpath(output_dir, "pet_image.h5")

            f = h5open(h5_path, "w")
            uid = MedImages.save_med_image(f, "pet", med_im)
            close(f)

            @test isfile(h5_path)
        end
    end
end
