# test/runtests.jl
# Main test entry point for MedImages.jl comprehensive test suite

using Test
using LinearAlgebra
using Dates
using MedImages

# Include test infrastructure
include("test_helpers.jl")
include("test_config.jl")

using .TestHelpers
using .TestConfig

# Check for optional dependencies
HDF5_AVAILABLE = try
    using HDF5
    true
catch e
    @warn "HDF5.jl not available: $e"
    false
end

PYCALL_AVAILABLE = try
    using PyCall
    sitk = pyimport("SimpleITK")
    np = pyimport("numpy")
    true
catch e
    @warn "PyCall/SimpleITK not available - some tests will be skipped: $e"
    false
end

# Create debug directory
mkpath(DEBUG_DIR)

@testset "MedImages.jl Complete Test Suite" begin

    # Module structure tests
    @testset "Module Structure Tests" begin
        @test isa(MedImages, Module)
        @test isdefined(MedImages, :MedImage_data_struct)
        @test isdefined(MedImages, :Load_and_save)
        @test isdefined(MedImages, :Basic_transformations)
        @test isdefined(MedImages, :Spatial_metadata_change)
        @test isdefined(MedImages, :Resample_to_target)
        @test isdefined(MedImages, :Utils)
        @info "Module structure verified"
    end

    # Test data availability
    @testset "Test Data Availability" begin
        @test isdir(TEST_DATA_DIR)

        if isfile(TEST_NIFTI_FILE)
            @test true
            @info "Found primary test NIfTI file: $TEST_NIFTI_FILE"
        else
            @test_broken false
            @warn "Primary NIfTI test file not found: $TEST_NIFTI_FILE"
        end

        if isfile(TEST_SYNTHETIC_FILE)
            @test true
            @info "Found synthetic test file: $TEST_SYNTHETIC_FILE"
        else
            @test_broken false
            @warn "Synthetic test file not found: $TEST_SYNTHETIC_FILE"
        end

        if isdir(TEST_DICOM_DIR)
            @test true
            @info "Found DICOM test directory: $TEST_DICOM_DIR"
        else
            @test_broken false
            @warn "DICOM test directory not found: $TEST_DICOM_DIR"
        end
    end

    # Python dependencies
    if PYCALL_AVAILABLE
        @testset "Python Dependencies" begin
            @test begin
                sitk = pyimport("SimpleITK")
                isa(sitk, PyObject)
            end
            @test begin
                np = pyimport("numpy")
                isa(np, PyObject)
            end
            @info "Python dependencies verified"
        end
    end

    # MedImage Data Struct Tests
    @testset "MedImage Data Struct Tests" begin
        include("medimage_data_struct_tests/test_medimage_struct.jl")
        include("medimage_data_struct_tests/test_enums.jl")
    end

    # Orientation Dicts Tests
    @testset "Orientation Dicts Tests" begin
        include("orientation_dicts_tests/test_orientation_conversion.jl")
    end

    # Utils Tests
    @testset "Utils Tests" begin
        include("utils_tests/test_get_base_indicies_arr.jl")
        include("utils_tests/test_ensure_tuple.jl")
        include("utils_tests/test_interpolate_my.jl")
        include("utils_tests/test_resample_kernel_launch.jl")
    end

    # Kernel Validity Tests
    @testset "Kernel Validity Tests" begin
        include("kernel_validity_tests/test_kernel_validity.jl")
    end

    # Load and Save Tests - require test data
    if isfile(TEST_NIFTI_FILE)
        @testset "Load and Save Tests" begin
            include("load_and_save_tests/test_load_image.jl")
            include("load_and_save_tests/test_update_voxel_and_spatial_data.jl")
            include("load_and_save_tests/test_update_voxel_data.jl")

            if PYCALL_AVAILABLE
                include("load_and_save_tests/test_create_nii_from_medimage.jl")
            end
        end
    else
        @testset "Load and Save Tests" begin
            @test_skip "Primary test file not available"
        end
    end

    # Basic Transformations Tests - require test data and PyCall
    if isfile(TEST_NIFTI_FILE) && PYCALL_AVAILABLE
        @testset "Basic Transformations Tests" begin
            include("basic_transformations_tests/test_rotate_mi.jl")
            include("basic_transformations_tests/test_crop_mi.jl")
            include("basic_transformations_tests/test_pad_mi.jl")
            include("basic_transformations_tests/test_translate_mi.jl")
            include("basic_transformations_tests/test_scale_mi.jl")
        end
    else
        @testset "Basic Transformations Tests" begin
            @test_skip "Test data or PyCall not available"
        end
    end

    # Spatial Metadata Change Tests - require test data and PyCall
    if isfile(TEST_NIFTI_FILE) && PYCALL_AVAILABLE
        @testset "Spatial Metadata Change Tests" begin
            include("spatial_metadata_change_tests/test_resample_to_spacing.jl")
            include("spatial_metadata_change_tests/test_change_orientation.jl")
        end
    else
        @testset "Spatial Metadata Change Tests" begin
            @test_skip "Test data or PyCall not available"
        end
    end

    # Resample to Target Tests - require both test files and PyCall
    if isfile(TEST_NIFTI_FILE) && isfile(TEST_SYNTHETIC_FILE) && PYCALL_AVAILABLE
        @testset "Resample to Target Tests" begin
            include("resample_to_target_tests/test_resample_to_image.jl")
        end
    else
        @testset "Resample to Target Tests" begin
            @test_skip "Test data or PyCall not available"
        end
    end

    # HDF5 Management Tests - require HDF5 and test data
    if HDF5_AVAILABLE && isfile(TEST_NIFTI_FILE)
        @testset "HDF5 Management Tests" begin
            include("hdf5_manag_tests/test_save_med_image.jl")
            include("hdf5_manag_tests/test_load_med_image.jl")
        end
    else
        @testset "HDF5 Management Tests" begin
            @test_skip "HDF5 or test data not available"
        end
    end

    # Brute Force Orientation Tests - require test data and PyCall
    if isfile(TEST_NIFTI_FILE) && PYCALL_AVAILABLE
        @testset "Brute Force Orientation Tests" begin
            include("brute_force_orientation_tests/test_change_image_orientation.jl")
            include("brute_force_orientation_tests/test_brute_force_find_perm_rev.jl")
        end
    else
        @testset "Brute Force Orientation Tests" begin
            @test_skip "Test data or PyCall not available"
        end
    end

    # Differentiability Tests
    @testset "Differentiability Tests" begin
        # Check if Zygote is available (should be from Project.toml extras)
        try
            using Zygote
            include("differentiability_tests/test_gradients.jl")
        catch e
            @warn "Zygote not available or failed to load: $e"
            @test_skip "Zygote required for differentiability tests"
        end
    end

end # main testset

# Summary output
@info """
==========================================
MedImages.jl Test Suite Completed
==========================================

Test Configuration:
- Test data directory: $TEST_DATA_DIR
- Primary test file (volume-0.nii.gz): $(isfile(TEST_NIFTI_FILE) ? "Found" : "Missing")
- Secondary test file (synthethic_small.nii.gz): $(isfile(TEST_SYNTHETIC_FILE) ? "Found" : "Missing")
- DICOM test directory: $(isdir(TEST_DICOM_DIR) ? "Found" : "Missing")
- HDF5 available: $HDF5_AVAILABLE
- PyCall/SimpleITK available: $PYCALL_AVAILABLE

Test output directories created in:
$(joinpath(@__DIR__, "*_tests", "outputs"))

==========================================
"""
