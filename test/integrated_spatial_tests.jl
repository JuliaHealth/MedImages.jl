# test/integrated_spatial_tests.jl
# Integrated tests for spatial metadata changes: resampling and orientation

using Test
using LinearAlgebra
using PyCall
using MedImages

# Import test infrastructure
if !isdefined(@__MODULE__, :TestHelpers)
    include("test_helpers.jl")
    include("test_config.jl")
end
using .TestHelpers
using .TestConfig

# Map MedImages interpolator to SimpleITK interpolator for reference
function get_sitk_interpolator(interp)
    sitk = pyimport("SimpleITK")
    if interp == MedImages.Nearest_neighbour_en
        return sitk.sitkNearestNeighbor
    elseif interp == MedImages.Linear_en
        return sitk.sitkLinear
    elseif interp == MedImages.B_spline_en
        return sitk.sitkBSpline
    end
end

# SimpleITK reference implementation for resampling
function sitk_resample(path_nifti, targetSpac, sitk_interp)
    sitk = pyimport("SimpleITK")
    image = sitk.ReadImage(path_nifti)
    origSize = image.GetSize()
    orig_spacing = image.GetSpacing()
    new_size = Tuple{Int64,Int64,Int64}([
        Int64(ceil(origSize[1] * (orig_spacing[1] / targetSpac[1]))),
        Int64(ceil(origSize[2] * (orig_spacing[2] / targetSpac[2]))),
        Int64(ceil(origSize[3] * (orig_spacing[3] / targetSpac[3])))
    ])

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(targetSpac)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk_interp)
    py_size = PyObject((Int(new_size[1]), Int(new_size[2]), Int(new_size[3])))
    resample.SetSize(py_size)
    return resample.Execute(image)
end

# SimpleITK reference implementation for orientation
function change_image_orientation_sitk(path_nifti, orientation)
    sitk = pyimport("SimpleITK")
    image = sitk.ReadImage(path_nifti)
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(orientation)
    return orient_filter.Execute(image)
end

@testset "Integrated Spatial Metadata Tests" begin
    sitk = pyimport("SimpleITK")

    if !isfile(TEST_NIFTI_FILE)
        @test_skip "Primary test file not found: $TEST_NIFTI_FILE"
        return
    end

    med_im = load_test_image()

    @testset "resample_to_spacing Tests" begin
        with_temp_output_dir("spatial_integrated", "resample_spacing") do output_dir
            for interp in INTERPOLATORS[1:2] # Test subset for speed
                interp_name = INTERPOLATOR_NAMES[interp]
                sitk_interp = get_sitk_interpolator(interp)

                @testset "Interpolator: $interp_name" begin
                    for spac in SPACING_TEST_VALUES[1:2]
                        @testset "spacing=$spac" begin
                            # MedImages implementation
                            med_im_resampled = MedImages.resample_to_spacing(med_im, spac, interp)
                            @test isapprox(collect(med_im_resampled.spacing), collect(spac); atol=0.01)
                            
                            # SimpleITK comparison
                            sitk_image = sitk_resample(TEST_NIFTI_FILE, spac, sitk_interp)
                            test_object_equality(med_im_resampled, sitk_image; allow_dimension_mismatch=true, origin_atol=20.0)
                        end
                    end
                end
            end
        end
    end

    @testset "change_orientation Tests" begin
        with_temp_output_dir("spatial_integrated", "change_orientation") do output_dir
            for orientation_enum in AVAILABLE_ORIENTATIONS[1:4]
                orientation_str = MedImages.orientation_enum_to_string[orientation_enum]

                @testset "Orientation: $orientation_str" begin
                    # MedImages implementation
                    med_im_oriented = MedImages.change_orientation(med_im, orientation_enum)
                    
                    # SimpleITK comparison
                    sitk_image = change_image_orientation_sitk(TEST_NIFTI_FILE, orientation_str)
                    test_object_equality(med_im_oriented, sitk_image)
                    
                    @test prod(size(med_im_oriented.voxel_data)) == prod(size(med_im.voxel_data))
                end
            end
        end
    end

    @testset "resample_to_image Tests" begin
        if isfile(TEST_SYNTHETIC_FILE)
            with_temp_output_dir("spatial_integrated", "resample_to_image") do output_dir
                im_fixed = med_im
                im_moving = MedImages.load_image(TEST_SYNTHETIC_FILE, "CT")

                resampled = MedImages.resample_to_image(im_fixed, im_moving, MedImages.Linear_en, 0.0)
                
                @test size(resampled.voxel_data) == size(im_fixed.voxel_data)
                @test isapprox(collect(resampled.spacing), collect(im_fixed.spacing); atol=0.01)
                @test isapprox(collect(resampled.origin), collect(im_fixed.origin); atol=0.1)
            end
        else
            @test_skip "Synthetic test file not found"
        end
    end
end
