using PyCall
using JLD
const sitk = pyimport("SimpleITK")
const np = pyimport("numpy")

# --- Helper Functions (Ported from generate_all_visual_samples.jl) ---

function get_center_sitk(img)
    size_arr = img.GetSize()
    idx_center = Tuple((sz-1)/2.0 for sz in size_arr)
    return img.TransformContinuousIndexToPhysicalPoint(idx_center)
end

function matrix_from_axis_angle(a)
    ux, uy, uz, theta = a
    c = cos(theta)
    s = sin(theta)
    ci = 1.0 - c
    R = [[ci * ux * ux + c, ci * ux * uy - uz * s, ci * ux * uz + uy * s],
         [ci * uy * ux + uz * s, ci * uy * uy + c, ci * uy * uz - ux * s],
         [ci * uz * ux - uy * s, ci * uz * uy + ux * s, ci * uz * uz + c]]
    return R
end

function rotation3d_sitk(image, axis, theta_deg, sitk_interp)
    theta_rad = deg2rad(theta_deg)
    euler_transform = sitk.Euler3DTransform()
    image_center = get_center_sitk(image)
    euler_transform.SetCenter(image_center)
    direction = image.GetDirection()

    if axis == 3
        axis_angle = (direction[3], direction[6], direction[9], theta_rad)
    elseif axis == 2
        axis_angle = (direction[2], direction[5], direction[8], theta_rad)
    elseif axis == 1
        axis_angle = (direction[1], direction[4], direction[7], theta_rad)
    end

    rot_mat = matrix_from_axis_angle(axis_angle)
    euler_transform.SetMatrix([rot_mat[1][1], rot_mat[1][2], rot_mat[1][3],
                               rot_mat[2][1], rot_mat[2][2], rot_mat[2][3],
                               rot_mat[3][1], rot_mat[3][2], rot_mat[3][3]])

    return sitk.Resample(image, image, euler_transform, sitk_interp, 0.0)
end

function sitk_scale(image, zoom, sitk_interp)
    scale_transform = sitk.ScaleTransform(3, [1.0/zoom, 1.0/zoom, 1.0/zoom])
    scale_transform.SetCenter(get_center_sitk(image))
    return sitk.Resample(image, image, scale_transform, sitk_interp, 0.0)
end

function sitk_translate(image, translate_by, sitk_axis)
    translatee = [0.0, 0.0, 0.0]
    spacing = image.GetSpacing()
    translatee[sitk_axis + 1] = Float64(translate_by * spacing[sitk_axis + 1])
    transform = sitk.TranslationTransform(3, translatee)
    return sitk.TransformGeometry(image, transform)
end

function sitk_crop(sitk_image, beginning, crop_size)
    py_size = Tuple(Int(s) for s in crop_size)
    py_index = Tuple(Int(b) for b in beginning)
    return sitk.RegionOfInterest(sitk_image, py_size, py_index)
end

function sitk_pad(sitk_image, pad_beg, pad_end, pad_val)
    extract = sitk.ConstantPadImageFilter()
    extract.SetConstant(pad_val)
    extract.SetPadLowerBound(Tuple(Int(p) for p in pad_beg))
    extract.SetPadUpperBound(Tuple(Int(p) for p in pad_end))
    return extract.Execute(sitk_image)
end

function sitk_resample_to_spacing(image, targetSpac, sitk_interp)
    origSize = image.GetSize()
    orig_spacing = image.GetSpacing()
    new_size = [
        Int(ceil(origSize[i] * (orig_spacing[i] / targetSpac[i]))) for i in 1:3
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(targetSpac)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(sitk_interp)
    resample.SetSize(Tuple(new_size))
    return resample.Execute(image)
end

function sitk_shear(image, shear_xyz, sitk_interp)
    affine = sitk.AffineTransform(3)
    mat = [1.0, shear_xyz[1], shear_xyz[2],
           0.0, 1.0,          +shear_xyz[3],
           0.0, 0.0,          1.0]
    affine.SetMatrix(mat)
    affine.SetCenter(get_center_sitk(image))
    return sitk.Resample(image, image, affine.GetInverse(), sitk_interp, 0.0)
end

function load_sitk_from_jld(jld_path)
    data_dict = load(jld_path)
    voxel_data = data_dict["voxel_data"]
    origin      = data_dict["origin"]
    spacing     = data_dict["spacing"]
    direction   = data_dict["direction"]

    permuted_data = permutedims(voxel_data, (3, 2, 1))
    
    img_sitk = sitk.GetImageFromArray(np.array(permuted_data))
    img_sitk.SetOrigin(collect(origin))
    img_sitk.SetSpacing(collect(spacing))
    img_sitk.SetDirection(collect(direction))
    return img_sitk
end

# --- Main Logic ---

function convert_jld_to_nifti_and_generate_refs(root_dir)
    println("Scanning for .jld files in: ", root_dir)
    
    # 1. Convert JLD to NIfTI (MedImages outputs)
    for (root, dirs, files) in walkdir(root_dir)
        for file in files
            if endswith(file, ".jld")
                jld_path = joinpath(root, file)
                nii_path = replace(jld_path, ".jld" => ".nii.gz")
                println("Converting JLD: ", jld_path)
                try
                    img_sitk = load_sitk_from_jld(jld_path)
                    sitk.WriteImage(img_sitk, nii_path)
                catch e
                    println("Error converting: ", jld_path)
                    println(e)
                end
            end
        end
    end

    # 2. Generate SimpleITK References
    println("Generating SimpleITK Reference Images...")
    
    # Paths (Hardcoded assumptions based on generate_all_visual_samples.jl structure)
    # We verify relative to root_dir
    SITK_BATCHED_DIR = joinpath(root_dir, "batched", "SimpleITK")
    SITK_NON_BATCHED_DIR = joinpath(root_dir, "non_batched", "SimpleITK")
    MI_BATCHED_DIR = joinpath(root_dir, "batched", "MedImages")

    input1_path = joinpath(MI_BATCHED_DIR, "input_1.jld")
    input2_path = joinpath(MI_BATCHED_DIR, "input_2.jld")

    if !isfile(input1_path) || !isfile(input2_path)
        println("ERROR: Input files not found in MedImages directory. Skipping SITK generation.")
        println("Expected: ", input1_path)
        return
    end

    s1 = load_sitk_from_jld(input1_path)
    s2 = load_sitk_from_jld(input2_path)
    
    mkpath(SITK_BATCHED_DIR)
    mkpath(SITK_NON_BATCHED_DIR)

    # Save inputs
    sitk.WriteImage(s1, joinpath(SITK_BATCHED_DIR, "input_1.nii.gz"))
    sitk.WriteImage(s2, joinpath(SITK_BATCHED_DIR, "input_2.nii.gz"))
    sitk.WriteImage(s1, joinpath(SITK_NON_BATCHED_DIR, "input_1.nii.gz"))
    sitk.WriteImage(s2, joinpath(SITK_NON_BATCHED_DIR, "input_2.nii.gz"))

    # BATCHED (Replicated Logic)
    
    # Rotate
    axis = 3
    sitk.WriteImage(rotation3d_sitk(s1, axis, 0.0, sitk.sitkLinear), joinpath(SITK_BATCHED_DIR, "rotated_0deg_axis$(axis)_img1.nii.gz"))
    sitk.WriteImage(rotation3d_sitk(s2, axis, 45.0, sitk.sitkLinear), joinpath(SITK_BATCHED_DIR, "rotated_45deg_axis$(axis)_img2.nii.gz"))

    # Scale
    zoom = 0.5
    sitk.WriteImage(sitk_scale(s1, zoom, sitk.sitkLinear), joinpath(SITK_BATCHED_DIR, "scaled_$(zoom)x_img1.nii.gz"))
    sitk.WriteImage(sitk_scale(s2, zoom, sitk.sitkLinear), joinpath(SITK_BATCHED_DIR, "scaled_$(zoom)x_img2.nii.gz"))

    # Translate
    shift = 10
    trans_axis = 0 # X in SITK (Julia was 1)
    sitk.WriteImage(sitk_translate(s1, shift, trans_axis), joinpath(SITK_BATCHED_DIR, "translated_$(shift)_axis$(trans_axis+1)_img1.nii.gz"))
    sitk.WriteImage(sitk_translate(s2, shift*2, trans_axis), joinpath(SITK_BATCHED_DIR, "translated_$(shift*2)_axis$(trans_axis+1)_img2.nii.gz"))

    # Crop
    crop_beg = (16, 16, 16)
    crop_size = (32, 32, 32)
    sitk.WriteImage(sitk_crop(s1, crop_beg, crop_size), joinpath(SITK_BATCHED_DIR, "cropped_$(crop_size)_img1.nii.gz"))
    sitk.WriteImage(sitk_crop(s2, crop_beg, crop_size), joinpath(SITK_BATCHED_DIR, "cropped_$(crop_size)_img2.nii.gz"))

    # Pad
    pad_amt = (5, 5, 5)
    sitk.WriteImage(sitk_pad(s1, pad_amt, pad_amt, 0.0), joinpath(SITK_BATCHED_DIR, "padded_$(pad_amt)_img1.nii.gz"))
    sitk.WriteImage(sitk_pad(s2, pad_amt, pad_amt, 0.0), joinpath(SITK_BATCHED_DIR, "padded_$(pad_amt)_img2.nii.gz"))

    # Affine Shear
    shear_val = (0.5, 0.0, 0.0)
    sitk.WriteImage(s1, joinpath(SITK_BATCHED_DIR, "affine_identity_img1.nii.gz")) 
    sitk.WriteImage(sitk_shear(s2, shear_val, sitk.sitkLinear), joinpath(SITK_BATCHED_DIR, "affine_sheared_$(shear_val)_img2.nii.gz"))

    # Resample to Spacing
    target_spacing = (2.0, 2.0, 2.0)
    sitk.WriteImage(sitk_resample_to_spacing(s1, target_spacing, sitk.sitkLinear), joinpath(SITK_BATCHED_DIR, "resample_spacing_2mm_img1.nii.gz"))
    sitk.WriteImage(sitk_resample_to_spacing(s2, target_spacing, sitk.sitkLinear), joinpath(SITK_BATCHED_DIR, "resample_spacing_2mm_img2.nii.gz"))

    # Resample to Image
    # Recreate refs
    # ref1: origin=(10,10,10), spacing=(1,1,1), size=(32,32,32)
    # ref2: origin=(-10,-10,-10), spacing=(1,1,1), size=(32,32,32)
    # They share direction/type with img1/img2
    ref1_sitk = sitk.GetImageFromArray(np.zeros((32,32,32))) # Permuted (32,32,32) is same
    ref1_sitk.SetOrigin((10.0, 10.0, 10.0))
    ref1_sitk.SetSpacing((1.0, 1.0, 1.0))
    ref1_sitk.SetDirection(s1.GetDirection())

    ref2_sitk = sitk.GetImageFromArray(np.zeros((32,32,32)))
    ref2_sitk.SetOrigin((-10.0, -10.0, -10.0))
    ref2_sitk.SetSpacing((1.0, 1.0, 1.0))
    ref2_sitk.SetDirection(s2.GetDirection())

    sitk.WriteImage(sitk.Resample(s1, ref1_sitk, sitk.Transform(), sitk.sitkLinear, 0.0, s1.GetPixelIDValue()), joinpath(SITK_BATCHED_DIR, "resample_to_ref_origin10_img1.nii.gz"))
    sitk.WriteImage(sitk.Resample(s2, ref2_sitk, sitk.Transform(), sitk.sitkLinear, 0.0, s2.GetPixelIDValue()), joinpath(SITK_BATCHED_DIR, "resample_to_ref_origin-10_img2.nii.gz"))


    # NON-BATCHED (Replicated Logic)
    
    # Rotation
    sitk.WriteImage(rotation3d_sitk(s2, 3, 45.0, sitk.sitkLinear), joinpath(SITK_NON_BATCHED_DIR, "rotated_45deg_axis3_img2.nii.gz"))
    
    # Scale
    sitk.WriteImage(sitk_scale(s1, 0.5, sitk.sitkLinear), joinpath(SITK_NON_BATCHED_DIR, "scaled_0.5x_img1.nii.gz"))

    # Translate
    sitk.WriteImage(sitk_translate(s1, 10, 0), joinpath(SITK_NON_BATCHED_DIR, "translated_10_axis1_img1.nii.gz"))

    # Crop
    sitk.WriteImage(sitk_crop(s1, crop_beg, crop_size), joinpath(SITK_NON_BATCHED_DIR, "cropped_$(crop_size)_img1.nii.gz"))

    # Pad
    sitk.WriteImage(sitk_pad(s1, pad_amt, pad_amt, 0.0), joinpath(SITK_NON_BATCHED_DIR, "padded_$(pad_amt)_img1.nii.gz"))

    # Resample to Spacing
    sitk.WriteImage(sitk_resample_to_spacing(s1, (2.0, 2.0, 2.0), sitk.sitkLinear), joinpath(SITK_NON_BATCHED_DIR, "resample_spacing_2mm_img1.nii.gz"))

end


if length(ARGS) > 0
    convert_jld_to_nifti_and_generate_refs(ARGS[1])
else
    # Default to test/visual_output relative to this script location or CWD
    # We assume run from project root
    convert_jld_to_nifti_and_generate_refs(abspath("test/visual_output"))
end
