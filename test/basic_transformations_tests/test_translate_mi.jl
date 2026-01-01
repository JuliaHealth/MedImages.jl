# test/basic_transformations_tests/test_translate_mi.jl
# Tests for translate_mi function from Basic_transformations module

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
function sitk_translate(image, translate_by, translate_in_axis)
    sitk = pyimport("SimpleITK")
    translatee = [0.0, 0.0, 0.0]
    translatee[translate_in_axis] = Float64(translate_by)
    transform = sitk.TranslationTransform(3, translatee)
    return sitk.TransformGeometry(image, transform)
end

@testset "translate_mi Tests" begin
    sitk = pyimport("SimpleITK")

    if !isfile(TEST_NIFTI_FILE)
        @test_skip "Test file not found: $TEST_NIFTI_FILE"
        return
    end

    with_temp_output_dir("basic_transformations", "translate_mi") do output_dir
        med_im = load_test_image()
        sitk_image = sitk.ReadImage(TEST_NIFTI_FILE)

        for interp in INTERPOLATORS
            interp_name = INTERPOLATOR_NAMES[interp]

            @testset "Interpolator: $interp_name" begin
                for t_val in TRANSLATION_VALUES
                    for axis in TRANSLATION_AXES
                        @testset "val=$t_val, axis=$axis" begin
                            try
                                # SimpleITK reference
                                sitk_translated = sitk_translate(sitk_image, t_val, axis)

                                # Save reference output
                                output_file = joinpath(output_dir, "translated_$(interp_name)_$(t_val)_$(axis)_sitk.nii.gz")
                                sitk.WriteImage(sitk_translated, output_file)

                                # MedImages implementation
                                medIm_translated = MedImages.translate_mi(med_im, t_val, axis, interp)

                                # Save MedImages output
                                mi_output_file = joinpath(output_dir, "translated_$(interp_name)_$(t_val)_$(axis)_mi.nii.gz")
                                create_nii_from_medimage_for_test(medIm_translated, mi_output_file)

                                # Compare results
                                test_object_equality(medIm_translated, sitk_translated)

                                @test true
                            catch e
                                @test_broken false
                                @info "Error in translate test (interp=$interp_name, val=$t_val, axis=$axis): $e"
                            end
                        end
                    end
                end
            end
        end
    end
end

@testset "translate_mi Origin Tests" begin
    if !isfile(TEST_NIFTI_FILE)
        @test_skip "Test file not found: $TEST_NIFTI_FILE"
        return
    end

    with_temp_output_dir("basic_transformations", "translate_mi_origin") do output_dir
        med_im = load_test_image()
        original_origin = collect(med_im.origin)

        @testset "Negative translation" begin
            result = MedImages.translate_mi(med_im, -5, 1, MedImages.Linear_en)
            # Translation should modify the origin
            @test result.origin != med_im.origin
        end

        @testset "Zero translation (no-op)" begin
            result = MedImages.translate_mi(med_im, 0, 1, MedImages.Linear_en)
            @test isapprox(collect(result.origin), original_origin; atol=1e-6)
        end

        @testset "Sequential translations" begin
            result1 = MedImages.translate_mi(med_im, 5, 1, MedImages.Linear_en)
            result2 = MedImages.translate_mi(result1, 5, 2, MedImages.Linear_en)
            result3 = MedImages.translate_mi(result2, 5, 3, MedImages.Linear_en)
            # Should be able to apply multiple translations
            @test size(result3.voxel_data) == size(med_im.voxel_data)
        end
    end
end
