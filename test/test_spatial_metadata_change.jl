using Test
using LinearAlgebra
using PyCall
using MedImages

function sitk_resample(path_nifti, targetSpac)
    sitk = pyimport("SimpleITK")
    image = sitk.ReadImage(path_nifti)
    origSize = image.GetSize()
    orig_spacing = image.GetSpacing()
    new_size = Tuple{Int64,Int64,Int64}([Int64(ceil(origSize[1] * (orig_spacing[1] / targetSpac[1]))),
        Int64(ceil(origSize[2] * (orig_spacing[2] / targetSpac[2]))),
        Int64(ceil(origSize[3] * (orig_spacing[3] / targetSpac[3])))])

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(targetSpac)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkLinear)
    # Fix types for SimpleITK - use Tuple for size
    py_size = PyObject((Int(new_size[1]), Int(new_size[2]), Int(new_size[3])))
    resample.SetSize(py_size)
    return resample.Execute(image)
end

function change_image_orientation_sitk(path_nifti, orientation)
    sitk = pyimport("SimpleITK")
    image = sitk.ReadImage(path_nifti)
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(orientation)
    return orient_filter.Execute(image)
end

function create_nii_from_medimage_test(med_image, file_path::String)
    sitk = pyimport("SimpleITK")
    np = pyimport("numpy")
    voxel_data_np = np.array(med_image.voxel_data)
    image_sitk = sitk.GetImageFromArray(voxel_data_np)
    image_sitk.SetOrigin(med_image.origin)
    image_sitk.SetSpacing(med_image.spacing)
    image_sitk.SetDirection(med_image.direction)
    sitk.WriteImage(image_sitk, file_path * ".nii.gz")
end

function test_resample_to_spacing_suite(path_nifti)
    @testset "Resample to Spacing Test Suite" begin
        sitk = pyimport("SimpleITK")
        
        for (index, spac) in enumerate([(5.0, 0.9, 0.7), (1.0, 2.0, 1.1)])  # Reduced for faster testing
            @testset "Resample spacing=$spac" begin
                @test begin
                    # Load image using MedImages
                    med_im = MedImages.load_image(path_nifti, "CT")
                    
                    # SimpleITK implementation
                    sitk_image = sitk_resample(path_nifti, spac)
                    
                    # MedImages implementation
                    med_im_resampled = MedImages.resample_to_spacing(med_im, spac, MedImages.Linear_en)
                    
                    # Save debug outputs
                    debug_dir = joinpath(dirname(@__FILE__), "..", "test_data", "debug")
                    mkpath(debug_dir)
                    sitk.WriteImage(sitk_image, "$(debug_dir)/sitk_resample_$(index).nii.gz")
                    create_nii_from_medimage_test(med_im_resampled, "$(debug_dir)/medim_resample_$(index)")
                    
                    test_object_equality(med_im_resampled, sitk_image)
                    true
                end
            end
        end
    end
end

function test_change_orientation_suite(path_nifti)
    @testset "Change Orientation Test Suite" begin
        sitk = pyimport("SimpleITK")
        
        # Test common orientations
        orientations = ["RAS", "LAS", "RPI", "LPI"]  # Reduced for faster testing
        
        for (index, orientation) in enumerate(orientations)
            @testset "Orientation change to $orientation" begin
                @test begin
                    # Load image using MedImages
                    med_im = MedImages.load_image(path_nifti, "CT")
                    
                    # SimpleITK implementation
                    sitk_image = change_image_orientation_sitk(path_nifti, orientation)
                    
                    # MedImages implementation - convert string to orientation enum
                    orientation_enum = MedImages.string_to_orientation_enum[orientation]
                    med_im_oriented = MedImages.change_orientation(med_im, orientation_enum)
                    
                    # Save debug outputs
                    debug_dir = joinpath(dirname(@__FILE__), "..", "test_data", "debug")
                    mkpath(debug_dir)
                    sitk.WriteImage(sitk_image, "$(debug_dir)/sitk_orient_$(index).nii.gz")
                    create_nii_from_medimage_test(med_im_oriented, "$(debug_dir)/medim_orient_$(index)")
                    
                    test_object_equality(med_im_oriented, sitk_image)
                    true
                end
            end
        end
    end
end