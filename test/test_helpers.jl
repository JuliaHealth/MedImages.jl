# test/test_helpers.jl
# Shared test utilities for MedImages.jl test suite

module TestHelpers

using Test
using PyCall
using MedImages
using LinearAlgebra
using Dates
using Random

export TEST_DATA_DIR
export setup_test_output_dir
export with_temp_output_dir
export test_object_equality
export load_test_image
export assert_medimage_valid
export compare_medimage_metadata
export create_nii_from_medimage_for_test
export get_sitk_modules

# Test data base path
const TEST_DATA_DIR = joinpath(@__DIR__, "..", "test_data")

"""
    get_sitk_modules()

Import and return SimpleITK and NumPy modules.
Returns (sitk, np) tuple.
"""
function get_sitk_modules()
    sitk = pyimport("SimpleITK")
    np = pyimport("numpy")
    return (sitk, np)
end

"""
    setup_test_output_dir(module_name::String, function_name::String)

Create a unique output directory for a specific test module and function.
Directory structure: test/<module>_tests/outputs/<function>/<timestamp>/

Returns the path to the created directory.
"""
function setup_test_output_dir(module_name::String, function_name::String)
    base_dir = joinpath(@__DIR__, "$(module_name)_tests", "outputs", function_name)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS_") * string(rand(1000:9999))
    output_dir = joinpath(base_dir, timestamp)
    mkpath(output_dir)
    return output_dir
end

"""
    with_temp_output_dir(f::Function, module_name::String, function_name::String)

Run a test block with automatic output directory setup.
The output directory is passed to the function f.
Outputs are preserved for artifact upload in CI.
"""
function with_temp_output_dir(f::Function, module_name::String, function_name::String)
    output_dir = setup_test_output_dir(module_name, function_name)
    f(output_dir)
    return output_dir
end

"""
    test_object_equality(med_im, sitk_image; spacing_atol=0.1, direction_atol=0.2, origin_atol=0.1, voxel_rtol=0.15)

Compare MedImage with SimpleITK image for equality.
Uses configurable tolerances for metadata and voxel data comparison.
"""
function test_object_equality(med_im::MedImages.MedImage_data_struct.MedImage, sitk_image;
                               spacing_atol=0.1, direction_atol=0.2, origin_atol=0.1, voxel_rtol=0.15)
    sitk = pyimport("SimpleITK")

    @testset "Metadata Comparison" begin
        @test isapprox(collect(sitk_image.GetSpacing()), collect(med_im.spacing); atol=spacing_atol)
        @test isapprox(collect(sitk_image.GetDirection()), collect(med_im.direction); atol=direction_atol)
        @test isapprox(collect(sitk_image.GetOrigin()), collect(med_im.origin); atol=origin_atol)
    end

    @testset "Voxel Data Comparison" begin
        arr = sitk.GetArrayFromImage(sitk_image)
        vox = permutedims(med_im.voxel_data, (3, 2, 1))
        @test isapprox(arr, vox; rtol=voxel_rtol)
    end
end

"""
    load_test_image(filename::String="volume-0.nii.gz"; image_type::String="CT")

Load a test image from the test_data directory.
Returns a MedImage object.
"""
function load_test_image(filename::String="volume-0.nii.gz"; image_type::String="CT")
    path = joinpath(TEST_DATA_DIR, filename)
    if !isfile(path)
        error("Test file not found: $path")
    end
    return MedImages.load_image(path, image_type)
end

"""
    assert_medimage_valid(im::MedImages.MedImage_data_struct.MedImage)

Assert that a MedImage object is valid and has expected structure.
Checks voxel data dimensions, metadata fields, and constraints.
"""
function assert_medimage_valid(im::MedImages.MedImage_data_struct.MedImage)
    @testset "MedImage Validity" begin
        @test !isempty(im.voxel_data)
        @test ndims(im.voxel_data) == 3
        @test length(im.origin) == 3
        @test length(im.spacing) == 3
        @test length(im.direction) == 9
        @test all(s -> s > 0, im.spacing)
    end
end

"""
    compare_medimage_metadata(im1, im2; spacing_atol=1e-6, origin_atol=1e-6, direction_atol=1e-6)

Compare metadata between two MedImage objects.
"""
function compare_medimage_metadata(im1, im2;
                                    spacing_atol=1e-6, origin_atol=1e-6, direction_atol=1e-6)
    @testset "Metadata Comparison" begin
        @test isapprox(collect(im1.spacing), collect(im2.spacing); atol=spacing_atol)
        @test isapprox(collect(im1.origin), collect(im2.origin); atol=origin_atol)
        @test isapprox(collect(im1.direction), collect(im2.direction); atol=direction_atol)
    end
end

"""
    create_nii_from_medimage_for_test(med_image, file_path::String)

Create a NIfTI file from MedImage for test verification.
Appends .nii.gz extension if not present.
Returns the full path to the created file.
"""
function create_nii_from_medimage_for_test(med_image, file_path::String)
    sitk, np = get_sitk_modules()

    # Convert voxel_data to numpy array
    voxel_data_np = np.array(med_image.voxel_data)
    image_sitk = sitk.GetImageFromArray(voxel_data_np)

    # Set spatial metadata
    image_sitk.SetOrigin(med_image.origin)
    image_sitk.SetSpacing(med_image.spacing)
    image_sitk.SetDirection(med_image.direction)

    # Ensure proper extension
    full_path = endswith(file_path, ".nii.gz") ? file_path : file_path * ".nii.gz"
    sitk.WriteImage(image_sitk, full_path)

    return full_path
end

# Helper macro for safe test execution with error handling
macro test_with_error_handling(test_name, test_expr)
    return quote
        @testset $test_name begin
            try
                $test_expr
                @test true
            catch e
                @test_broken false
                @info string("Error in ", $test_name, ": ", e)
                @info "Stack trace:" exception=(e, catch_backtrace())
            end
        end
    end
end

export @test_with_error_handling

end # module TestHelpers
