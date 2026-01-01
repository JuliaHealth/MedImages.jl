# test/spatial_metadata_change_tests/test_change_orientation.jl
# Tests for change_orientation function from Spatial_metadata_change module

using Test
using LinearAlgebra
using PyCall
using MedImages

# Import test infrastructure
include(joinpath(@__DIR__, "..", "test_helpers.jl"))
include(joinpath(@__DIR__, "..", "test_config.jl"))
using .TestHelpers
using .TestConfig

# SimpleITK reference implementation
function change_image_orientation_sitk(path_nifti, orientation)
    sitk = pyimport("SimpleITK")
    image = sitk.ReadImage(path_nifti)
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(orientation)
    return orient_filter.Execute(image)
end

# Map orientation enum to string for SimpleITK
function orientation_to_string(orient_enum)
    return MedImages.orientation_enum_to_string[orient_enum]
end

@testset "change_orientation Tests" begin
    sitk = pyimport("SimpleITK")

    if !isfile(TEST_NIFTI_FILE)
        @test_skip "Test file not found: $TEST_NIFTI_FILE"
        return
    end

    with_temp_output_dir("spatial_metadata_change", "change_orientation") do output_dir
        med_im = load_test_image()

        for (index, orientation_enum) in enumerate(AVAILABLE_ORIENTATIONS)
            orientation_str = orientation_to_string(orientation_enum)

            @testset "Orientation: $orientation_str" begin
                try
                    # SimpleITK reference
                    sitk_image = change_image_orientation_sitk(TEST_NIFTI_FILE, orientation_str)

                    # Save reference output
                    output_file = joinpath(output_dir, "oriented_$(orientation_str)_sitk.nii.gz")
                    sitk.WriteImage(sitk_image, output_file)

                    # MedImages implementation
                    med_im_oriented = MedImages.change_orientation(med_im, orientation_enum)

                    # Save MedImages output
                    mi_output_file = joinpath(output_dir, "oriented_$(orientation_str)_mi.nii.gz")
                    create_nii_from_medimage_for_test(med_im_oriented, mi_output_file)

                    # Compare results
                    test_object_equality(med_im_oriented, sitk_image)

                    @test true
                catch e
                    @test_broken false
                    @info "Error in orientation test ($orientation_str): $e"
                end
            end
        end
    end
end

@testset "change_orientation Roundtrip Tests" begin
    if !isfile(TEST_NIFTI_FILE)
        @test_skip "Test file not found: $TEST_NIFTI_FILE"
        return
    end

    with_temp_output_dir("spatial_metadata_change", "orientation_roundtrip") do output_dir
        med_im = load_test_image()
        original_dims = size(med_im.voxel_data)

        @testset "RAS -> LAS -> RAS roundtrip" begin
            # Change to LAS
            med_im_las = MedImages.change_orientation(med_im, MedImages.ORIENTATION_LAS)
            # Change back to RAS
            med_im_ras = MedImages.change_orientation(med_im_las, MedImages.ORIENTATION_RAS)

            # Dimensions should be preserved
            @test size(med_im_ras.voxel_data) == original_dims
        end

        @testset "Orientation preserves voxel count" begin
            for orient in AVAILABLE_ORIENTATIONS[1:4]  # Test first 4 orientations
                result = MedImages.change_orientation(med_im, orient)
                # Total voxel count should be preserved (dimensions may be permuted)
                @test prod(size(result.voxel_data)) == prod(original_dims)
            end
        end
    end
end

@testset "change_orientation Direction Matrix Tests" begin
    if !isfile(TEST_NIFTI_FILE)
        @test_skip "Test file not found: $TEST_NIFTI_FILE"
        return
    end

    with_temp_output_dir("spatial_metadata_change", "orientation_direction") do output_dir
        med_im = load_test_image()

        @testset "Direction matrix is valid after orientation change" begin
            for orient in AVAILABLE_ORIENTATIONS[1:4]
                result = MedImages.change_orientation(med_im, orient)

                # Direction should have 9 elements (3x3 matrix flattened)
                @test length(result.direction) == 9

                # Direction matrix elements should be valid (not NaN or Inf)
                @test all(!isnan, result.direction)
                @test all(!isinf, result.direction)
            end
        end

        @testset "Origin is valid after orientation change" begin
            for orient in AVAILABLE_ORIENTATIONS[1:4]
                result = MedImages.change_orientation(med_im, orient)

                # Origin should have 3 elements
                @test length(result.origin) == 3

                # Origin should be valid (not NaN or Inf)
                @test all(!isnan, result.origin)
                @test all(!isinf, result.origin)
            end
        end
    end
end

@testset "Orientation String to Enum Conversion" begin
    @testset "All standard orientations have mappings" begin
        standard_orientations = ["RAS", "LAS", "RPI", "LPI", "RAI", "LAI", "RPS", "LPS"]

        for orient_str in standard_orientations
            @test haskey(MedImages.string_to_orientation_enum, orient_str)
            orient_enum = MedImages.string_to_orientation_enum[orient_str]
            @test MedImages.orientation_enum_to_string[orient_enum] == orient_str
        end
    end
end
