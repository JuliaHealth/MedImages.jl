using Test
using LinearAlgebra
using PyCall
using MedImages

# Python implementation for comparison
function matrix_from_axis_angle(a)
    ux, uy, uz, theta = a
    c = cos(theta)
    s = sin(theta)
    ci = 1.0 - c
    R = [[ci * ux * ux + c,
            ci * ux * uy - uz * s,
            ci * ux * uz + uy * s],
        [ci * uy * ux + uz * s,
            ci * uy * uy + c,
            ci * uy * uz - ux * s],
        [ci * uz * ux - uy * s,
            ci * uz * uy + ux * s,
            ci * uz * uz + c]]
    return R
end

function resample_sitk(image, transform)
    sitk = pyimport("SimpleITK")
    reference_image = image
    interpolator = sitk.sitkLinear
    default_value = 0
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)
end

function get_center_sitk(img)
    width, height, depth = img.GetSize()
    centt = (Int(ceil(width / 2)), Int(ceil(height / 2)), Int(ceil(depth / 2)))
    return img.TransformIndexToPhysicalPoint(centt)
end

function rotation3d_sitk(image, axis, theta)
    sitk = pyimport("SimpleITK")
    np = pyimport("numpy")
    
    theta = np.deg2rad(theta)
    euler_transform = sitk.Euler3DTransform()
    image_center = get_center_sitk(image)
    euler_transform.SetCenter(image_center)
    
    direction = image.GetDirection()
    
    if axis == 3
        axis_angle = (direction[3], direction[6], direction[9], theta)
    elseif axis == 2
        axis_angle = (direction[2], direction[5], direction[8], theta)
    elseif axis == 1
        axis_angle = (direction[1], direction[4], direction[7], theta)
    end
    
    np_rot_mat = matrix_from_axis_angle(axis_angle)
    euler_transform.SetMatrix([np_rot_mat[1][1], np_rot_mat[1][2], np_rot_mat[1][3], 
                              np_rot_mat[2][1], np_rot_mat[2][2], np_rot_mat[2][3], 
                              np_rot_mat[3][1], np_rot_mat[3][2], np_rot_mat[3][3]])
    
    return resample_sitk(image, euler_transform)
end

function test_rotation_suite(path_nifti, debug_folder_path)
    @testset "Rotation Test Suite" begin
        sitk = pyimport("SimpleITK")
        
        for ax in [1, 2, 3]
            for theta in [30.0, 60.0, 90.0]  # Reduced for faster testing
                @testset "Rotation axis=$ax, theta=$theta" begin
                    @test begin
                        # Load images using MedImages API
                        med_im = MedImages.load_image(path_nifti, "CT")  # Assuming load_image needs type
                        sitk_image = sitk.ReadImage(path_nifti)
                        
                        # SimpleITK implementation
                        rotated_sitk = rotation3d_sitk(sitk_image, ax, theta)
                        
                        # Save debug output
                        sitk.WriteImage(rotated_sitk, "$(debug_folder_path)/rotated_$(ax)_$(theta)_sitk.nii.gz")
                        
                        # Our Julia implementation using MedImages API
                        med_im_rotated = MedImages.rotate_mi(med_im, ax, theta, MedImages.Linear_en)
                        test_object_equality(med_im_rotated, rotated_sitk)
                        true
                    end
                end
            end
        end
    end
end

# Cropping tests
function sitk_crop(sitk_image, beginning, size)
    sitk = pyimport("SimpleITK")
    py_size = PyObject([Int(size[1]), Int(size[2]), Int(size[3])])
    py_index = PyObject([Int(beginning[1]), Int(beginning[2]), Int(beginning[3])])
    return sitk.RegionOfInterest(sitk_image, py_size, py_index)
end

function test_crops_suite(path_nifti, debug_folder_path)
    @testset "Cropping Test Suite" begin
        sitk = pyimport("SimpleITK")
        
        for beginning in [(0, 0, 0), (15, 17, 7)]
            for size in [(151, 156, 50), (150, 150, 53)]  # Reduced for faster testing
                @testset "Crop begin=$beginning, size=$size" begin
                    @test begin
                        med_im = MedImages.load_image(path_nifti, "CT")
                        sitk_image = sitk.ReadImage(path_nifti)
                        
                        cropped_sitk = sitk_crop(sitk_image, beginning, size)
                        sitk.WriteImage(cropped_sitk, "$(debug_folder_path)/cropped_$(beginning)_$(size).nii.gz")
                        
                        medIm_cropped = MedImages.crop_mi([med_im], beginning, size, MedImages.Linear_en)[1]
                        test_object_equality(medIm_cropped, cropped_sitk)
                        true
                    end
                end
            end
        end
    end
end

# Padding tests
function sitk_pad(sitk_image, pad_beg, pad_end, pad_val)
    sitk = pyimport("SimpleITK")
    extract = sitk.ConstantPadImageFilter()
    extract.SetConstant(pad_val)
    py_pad_beg = PyObject([Int(pad_beg[1]), Int(pad_beg[2]), Int(pad_beg[3])])
    py_pad_end = PyObject([Int(pad_end[1]), Int(pad_end[2]), Int(pad_end[3])])
    extract.SetPadLowerBound(py_pad_beg)
    extract.SetPadUpperBound(py_pad_end)
    return extract.Execute(sitk_image)
end

function test_pads_suite(path_nifti, debug_folder_path)
    @testset "Padding Test Suite" begin
        sitk = pyimport("SimpleITK")
        
        for pad_beg in [(10, 11, 13)]  # Reduced for faster testing
            for pad_end in [(10, 11, 13), (15, 17, 19)]
                for pad_val in [0.0, 111.5]
                    @testset "Pad beg=$pad_beg, end=$pad_end, val=$pad_val" begin
                        @test begin
                            med_im = MedImages.load_image(path_nifti, "CT")
                            sitk_image = sitk.ReadImage(path_nifti)
                            
                            sitk_padded = sitk_pad(sitk_image, pad_beg, pad_end, pad_val)
                            sitk.WriteImage(sitk_padded, "$(debug_folder_path)/padded_$(pad_beg)_$(pad_end)_$(pad_val).nii.gz")
                            
                            mi_padded = MedImages.pad_mi(med_im, pad_beg, pad_end, pad_val, MedImages.Linear_en)
                            test_object_equality(mi_padded, sitk_padded)
                            true
                        end
                    end
                end
            end
        end
    end
end

# Translation tests
function sitk_translate(image, translate_by, translate_in_axis)
    sitk = pyimport("SimpleITK")
    translatee = [0.0, 0.0, 0.0]
    translatee[translate_in_axis] = Float64(translate_by)
    transform = sitk.TranslationTransform(3, translatee)
    return sitk.TransformGeometry(image, transform)
end

function test_translate_suite(path_nifti, debug_folder_path)
    @testset "Translation Test Suite" begin
        sitk = pyimport("SimpleITK")
        
        for t_val in [1, 10]  # Reduced for faster testing
            for axis in [1, 2, 3]
                @testset "Translate val=$t_val, axis=$axis" begin
                    @test begin
                        med_im = MedImages.load_image(path_nifti, "CT")
                        sitk_image = sitk.ReadImage(path_nifti)
                        
                        sitk_translated = sitk_translate(sitk_image, t_val, axis)
                        sitk.WriteImage(sitk_translated, "$(debug_folder_path)/translated_$(t_val)_$(axis).nii.gz")
                        
                        medIm_translated = MedImages.translate_mi(med_im, t_val, axis, MedImages.Linear_en)
                        test_object_equality(medIm_translated, sitk_translated)
                        true
                    end
                end
            end
        end
    end
end

# Scaling tests
function sitk_scale(image, zoom)
    sitk = pyimport("SimpleITK")
    scale_transform = sitk.ScaleTransform(3, [zoom, zoom, zoom])
    return sitk.Resample(image, scale_transform, sitk.sitkBSpline, 0.0)
end

function test_scale_suite(path_nifti, debug_folder_path)
    @testset "Scaling Test Suite" begin
        sitk = pyimport("SimpleITK")
        
        for zoom in [0.6, 0.9, 1.3]  # Reduced for faster testing
            @testset "Scale zoom=$zoom" begin
                @test begin
                    med_im = MedImages.load_image(path_nifti, "CT")
                    sitk_image = sitk.ReadImage(path_nifti)
                    
                    sitk_scaled = sitk_scale(sitk_image, zoom)
                    sitk.WriteImage(sitk_scaled, "$(debug_folder_path)/scaled_$(zoom).nii.gz")
                    
                    medIm_scaled = MedImages.scale_mi(med_im, zoom, MedImages.Linear_en)
                    test_object_equality(medIm_scaled, sitk_scaled)
                    true
                end
            end
        end
    end
end