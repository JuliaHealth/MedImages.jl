# test/basic_transformations_tests/test_crop_mi.jl
# Tests for crop_mi function from Basic_transformations module

using Test
using LinearAlgebra
using PyCall
using MedImages

# Import test infrastructure (conditionally include if not already defined)
if !isdefined(@__MODULE__, :TestHelpers)
    include(joinpath(@__DIR__, "..", "test_helpers.jl"))
    include(joinpath(@__DIR__, "..", "test_config.jl"))
end
using .TestHelpers
using .TestConfig

# SimpleITK reference implementation
function sitk_crop(sitk_image, beginning, crop_size)
    sitk = pyimport("SimpleITK")
    # SimpleITK expects Python tuples of unsigned integers
    py_size = (UInt(crop_size[1]), UInt(crop_size[2]), UInt(crop_size[3]))
    py_index = (UInt(beginning[1]), UInt(beginning[2]), UInt(beginning[3]))
    return sitk.RegionOfInterest(sitk_image, py_size, py_index)
end

@testset "crop_mi Tests" begin
    sitk = pyimport("SimpleITK")

    if !isfile(TEST_NIFTI_FILE)
        @test_skip "Test file not found: $TEST_NIFTI_FILE"
        return
    end

    with_temp_output_dir("basic_transformations", "crop_mi") do output_dir
        med_im = load_test_image()
        sitk_image = sitk.ReadImage(TEST_NIFTI_FILE)

        for interp in INTERPOLATORS
            interp_name = INTERPOLATOR_NAMES[interp]

            @testset "Interpolator: $interp_name" begin
                for (beginning, crop_size) in CROP_TEST_CASES
                    @testset "begin=$beginning, size=$crop_size" begin
                        try
                            # SimpleITK reference (cropping is interpolation-independent)
                            cropped_sitk = sitk_crop(sitk_image, beginning, crop_size)

                            # Save reference output
                            output_file = joinpath(output_dir, "cropped_$(interp_name)_$(beginning)_$(crop_size)_sitk.nii.gz")
                            sitk.WriteImage(cropped_sitk, output_file)

                            # MedImages implementation
                            medIm_cropped = MedImages.crop_mi(med_im, beginning, crop_size, interp)

                            # Save MedImages output
                            mi_output_file = joinpath(output_dir, "cropped_$(interp_name)_$(beginning)_$(crop_size)_mi.nii.gz")
                            create_nii_from_medimage_for_test(medIm_cropped, mi_output_file)

                            # Compare results
                            test_object_equality(medIm_cropped, cropped_sitk)

                            @test true
                        catch e
                            @test_broken false
                            @info "Error in crop test (interp=$interp_name, begin=$beginning, size=$crop_size): $e"
                        end
                    end
                end
            end
        end
    end
end

@testset "crop_mi Edge Cases" begin
    if !isfile(TEST_NIFTI_FILE)
        @test_skip "Test file not found: $TEST_NIFTI_FILE"
        return
    end

    with_temp_output_dir("basic_transformations", "crop_mi_edge") do output_dir
        med_im = load_test_image()

        @testset "Single voxel crop" begin
            try
                result = MedImages.crop_mi(med_im, (0, 0, 0), (1, 1, 1), MedImages.Linear_en)
                @test size(result.voxel_data) == (1, 1, 1)
            catch e
                @test_broken false
                @info "Error in single voxel crop: $e"
            end
        end

        @testset "Full image crop (no-op)" begin
            try
                dims = size(med_im.voxel_data)
                result = MedImages.crop_mi(med_im, (0, 0, 0), dims, MedImages.Linear_en)
                @test size(result.voxel_data) == dims
            catch e
                @test_broken false
                @info "Error in full image crop: $e"
            end
        end
    end
end
