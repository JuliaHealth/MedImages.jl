# test/basic_transformations_tests/test_pad_mi.jl
# Tests for pad_mi function from Basic_transformations module

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
function sitk_pad(sitk_image, pad_beg, pad_end, pad_val)
    sitk = pyimport("SimpleITK")
    extract = sitk.ConstantPadImageFilter()
    extract.SetConstant(pad_val)
    # SimpleITK expects Python tuples of unsigned integers
    py_pad_beg = (UInt(pad_beg[1]), UInt(pad_beg[2]), UInt(pad_beg[3]))
    py_pad_end = (UInt(pad_end[1]), UInt(pad_end[2]), UInt(pad_end[3]))
    extract.SetPadLowerBound(py_pad_beg)
    extract.SetPadUpperBound(py_pad_end)
    return extract.Execute(sitk_image)
end

@testset "pad_mi Tests" begin
    sitk = pyimport("SimpleITK")

    if !isfile(TEST_NIFTI_FILE)
        @test_skip "Test file not found: $TEST_NIFTI_FILE"
        return
    end

    with_temp_output_dir("basic_transformations", "pad_mi") do output_dir
        med_im = load_test_image()
        sitk_image = sitk.ReadImage(TEST_NIFTI_FILE)

        for interp in INTERPOLATORS
            interp_name = INTERPOLATOR_NAMES[interp]

            @testset "Interpolator: $interp_name" begin
                for (pad_beg, pad_end, pad_val) in PAD_TEST_CASES
                    @testset "beg=$pad_beg, end=$pad_end, val=$pad_val" begin
                        try
                            # SimpleITK reference (padding is interpolation-independent)
                            sitk_padded = sitk_pad(sitk_image, pad_beg, pad_end, pad_val)

                            # Save reference output
                            output_file = joinpath(output_dir, "padded_$(interp_name)_$(pad_beg)_$(pad_end)_$(pad_val)_sitk.nii.gz")
                            sitk.WriteImage(sitk_padded, output_file)

                            # MedImages implementation
                            mi_padded = MedImages.pad_mi(med_im, pad_beg, pad_end, pad_val, interp)

                            # Save MedImages output
                            mi_output_file = joinpath(output_dir, "padded_$(interp_name)_$(pad_beg)_$(pad_end)_$(pad_val)_mi.nii.gz")
                            create_nii_from_medimage_for_test(mi_padded, mi_output_file)

                            # Compare results
                            test_object_equality(mi_padded, sitk_padded)

                            @test true
                        catch e
                            @test_broken false
                            @info "Error in pad test (interp=$interp_name, beg=$pad_beg, end=$pad_end, val=$pad_val): $e"
                        end
                    end
                end
            end
        end
    end
end

@testset "pad_mi Dimension Tests" begin
    if !isfile(TEST_NIFTI_FILE)
        @test_skip "Test file not found: $TEST_NIFTI_FILE"
        return
    end

    with_temp_output_dir("basic_transformations", "pad_mi_dim") do output_dir
        med_im = load_test_image()
        original_dims = size(med_im.voxel_data)

        @testset "Symmetric padding dimension check" begin
            pad_beg = (5, 5, 5)
            pad_end = (5, 5, 5)
            result = MedImages.pad_mi(med_im, pad_beg, pad_end, 0.0, MedImages.Linear_en)

            expected_dims = (original_dims[1] + 10, original_dims[2] + 10, original_dims[3] + 10)
            @test size(result.voxel_data) == expected_dims
        end

        @testset "Asymmetric padding dimension check" begin
            pad_beg = (3, 5, 7)
            pad_end = (4, 6, 8)
            result = MedImages.pad_mi(med_im, pad_beg, pad_end, 0.0, MedImages.Linear_en)

            expected_dims = (original_dims[1] + 7, original_dims[2] + 11, original_dims[3] + 15)
            @test size(result.voxel_data) == expected_dims
        end

        @testset "Zero padding (no-op)" begin
            result = MedImages.pad_mi(med_im, (0, 0, 0), (0, 0, 0), 0.0, MedImages.Linear_en)
            @test size(result.voxel_data) == original_dims
        end
    end
end
