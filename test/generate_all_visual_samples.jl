using MedImages
using MedImages.MedImage_data_struct
using MedImages.Utils
using MedImages.Basic_transformations
using MedImages.Spatial_metadata_change
using MedImages.Resample_to_target
using PyCall
using Dates

# Import SimpleITK and NumPy for reference generation
const sitk = pyimport("SimpleITK")
const np = pyimport("numpy")

# Ensure output directory exists
ROOT_OUTPUT_DIR = abspath(joinpath(@__DIR__, "visual_output"))
println("Absolute Root Output Dir: ", ROOT_OUTPUT_DIR)
flush(stdout)

BATCHED_DIR = joinpath(ROOT_OUTPUT_DIR, "batched")
NON_BATCHED_DIR = joinpath(ROOT_OUTPUT_DIR, "non_batched")

MI_BATCHED_DIR = joinpath(BATCHED_DIR, "MedImages")
SITK_BATCHED_DIR = joinpath(BATCHED_DIR, "SimpleITK")

MI_NON_BATCHED_DIR = joinpath(NON_BATCHED_DIR, "MedImages")
SITK_NON_BATCHED_DIR = joinpath(NON_BATCHED_DIR, "SimpleITK")

for d in [MI_BATCHED_DIR, SITK_BATCHED_DIR, MI_NON_BATCHED_DIR, SITK_NON_BATCHED_DIR]
    mkpath(d)
end

# --- Helpers ---
function save_visual(med_image, file_path::String)
    println("Saving MedImage to: ", file_path, ".nii.gz")
    MedImages.create_nii_from_medimage(med_image, file_path)
end

function medimage_to_sitk(med_im)
    voxel_data_permuted = permutedims(med_im.voxel_data, (3, 2, 1))
    voxel_data_np = np.array(voxel_data_permuted)
    img = sitk.GetImageFromArray(voxel_data_np)
    img.SetOrigin(collect(med_im.origin))
    img.SetSpacing(collect(med_im.spacing))
    img.SetDirection(collect(med_im.direction))
    return img
end

# --- SITK Reference Implementations ---

function get_center_sitk(img)
    size_arr = img.GetSize()
    idx_center = [ (sz-1)/2.0 for sz in size_arr ]
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

function sitk_translate(image, translate_by, sitk_axis, sitk_interp=sitk.sitkLinear)
    translatee = [0.0, 0.0, 0.0]
    spacing = image.GetSpacing()
    translatee[sitk_axis + 1] = Float64(translate_by * spacing[sitk_axis + 1])
    transform = sitk.TranslationTransform(3, translatee)
    return sitk.Resample(image, image, transform.GetInverse(), sitk_interp, 0.0)
end

function sitk_crop(sitk_image, beginning, crop_size)
    v_size = sitk.VectorUInt32()
    for s in crop_size; v_size.push_back(UInt32(s)); end
    v_index = sitk.VectorInt32()
    for b in beginning; v_index.push_back(Int32(b)); end
    return sitk.RegionOfInterest(sitk_image, v_size, v_index)
end

function sitk_pad(sitk_image, pad_beg, pad_end, pad_val)
    extract = sitk.ConstantPadImageFilter()
    extract.SetConstant(pad_val)
    v_beg = sitk.VectorUInt32()
    for p in pad_beg; v_beg.push_back(UInt32(p)); end
    v_end = sitk.VectorUInt32()
    for p in pad_end; v_end.push_back(UInt32(p)); end
    extract.SetPadLowerBound(v_beg)
    extract.SetPadUpperBound(v_end)
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
    
    v_size = sitk.VectorUInt32()
    for s in new_size; v_size.push_back(UInt32(s)); end
    resample.SetSize(v_size)
    
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

# --- Generation Logic ---

function create_synthetic_medimage_local(size_t::Tuple, type::Symbol)
    data = zeros(Float32, size_t)
    cx, cy, cz = size_t .÷ 2

    if type == :asym_block
        data[cx-10:cx+10, cy-5:cy+15, cz-10:cz+10] .= 1.0
        data[cx+12:cx+15, cy, cz] .= 2.0 
    elseif type == :sphere
        for z in 1:size_t[3], y in 1:size_t[2], x in 1:size_t[1]
            if (x-cx)^2 + (y-cy)^2 + (z-cz)^2 <= 15^2
                data[x,y,z] = 1.0
            end
        end
    end

    return MedImage(
        voxel_data=data,
        origin=(0.0,0.0,0.0),
        spacing=(1.0,1.0,1.0),
        direction=(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0),
        image_type=MedImages.MedImage_data_struct.MRI_type,
        image_subtype=MedImages.MedImage_data_struct.T1_subtype,
        patient_id=String(type),
        current_device=MedImages.MedImage_data_struct.CPU_current_device,
        date_of_saving=now(),
        acquistion_time=now(),
        study_uid="study_1",
        patient_uid="patient_1",
        series_uid="series_1",
        study_description="synthetic study",
        legacy_file_name="synthetic.nii.gz"
    )
end

function main()
    println("Generating synthetic images...")
    flush(stdout)
    img1 = create_synthetic_medimage_local((64, 64, 64), :asym_block)
    img2 = create_synthetic_medimage_local((64, 64, 64), :sphere)

    # Convert to SITK for reference
    s1 = medimage_to_sitk(img1)
    s2 = medimage_to_sitk(img2)

    # Save inputs
    save_visual(img1, joinpath(MI_BATCHED_DIR, "input_1"))
    save_visual(img1, joinpath(MI_NON_BATCHED_DIR, "input_1"))
    
    sitk.WriteImage(s1, joinpath(SITK_BATCHED_DIR, "input_1.nii.gz"))
    sitk.WriteImage(s1, joinpath(SITK_NON_BATCHED_DIR, "input_1.nii.gz"))

    save_visual(img2, joinpath(MI_BATCHED_DIR, "input_2"))
    save_visual(img2, joinpath(MI_NON_BATCHED_DIR, "input_2"))
    
    sitk.WriteImage(s2, joinpath(SITK_BATCHED_DIR, "input_2.nii.gz"))
    sitk.WriteImage(s2, joinpath(SITK_NON_BATCHED_DIR, "input_2.nii.gz"))

    # --- BATCHED TRANSFORMATIONS ---
    println("Processing BATCHED transformations...")
    batch = create_batched_medimage([img1, img2])

    # 1. Rotate
    println("  Rotating...")
    angles = [0.0, 45.0]; axis = 3
    batch_rot = MedImages.rotate_mi(batch, axis, angles, Linear_en)
    res_rot = unbatch_medimage(batch_rot)
    save_visual(res_rot[1], joinpath(MI_BATCHED_DIR, "rotated_0deg_axis$(axis)_img1"))
    save_visual(res_rot[2], joinpath(MI_BATCHED_DIR, "rotated_45deg_axis$(axis)_img2"))
    sitk.WriteImage(rotation3d_sitk(s1, axis, 0.0, sitk.sitkLinear), joinpath(SITK_BATCHED_DIR, "rotated_0deg_axis$(axis)_img1.nii.gz"))
    sitk.WriteImage(rotation3d_sitk(s2, axis, 45.0, sitk.sitkLinear), joinpath(SITK_BATCHED_DIR, "rotated_45deg_axis$(axis)_img2.nii.gz"))

    # 2. Scale
    println("  Scaling...")
    zoom = 0.5
    batch_scale = MedImages.scale_mi(batch, zoom, Linear_en)
    res_scale = unbatch_medimage(batch_scale)
    save_visual(res_scale[1], joinpath(MI_BATCHED_DIR, "scaled_$(zoom)x_img1"))
    save_visual(res_scale[2], joinpath(MI_BATCHED_DIR, "scaled_$(zoom)x_img2"))
    sitk.WriteImage(sitk_scale(s1, zoom, sitk.sitkLinear), joinpath(SITK_BATCHED_DIR, "scaled_$(zoom)x_img1.nii.gz"))
    sitk.WriteImage(sitk_scale(s2, zoom, sitk.sitkLinear), joinpath(SITK_BATCHED_DIR, "scaled_$(zoom)x_img2.nii.gz"))

    # 3. Translate
    println("  Translating...")
    shift = 10; trans_axis = 1 
    batch_trans = MedImages.translate_mi(batch, [shift, shift*2], trans_axis, Linear_en)
    res_trans = unbatch_medimage(batch_trans)
    save_visual(res_trans[1], joinpath(MI_BATCHED_DIR, "translated_$(shift)_axis$(trans_axis)_img1"))
    save_visual(res_trans[2], joinpath(MI_BATCHED_DIR, "translated_$(shift*2)_axis$(trans_axis)_img2"))
    sitk.WriteImage(sitk_translate(s1, shift, 0), joinpath(SITK_BATCHED_DIR, "translated_$(shift)_axis$(trans_axis)_img1.nii.gz"))
    sitk.WriteImage(sitk_translate(s2, shift*2, 0), joinpath(SITK_BATCHED_DIR, "translated_$(shift*2)_axis$(trans_axis)_img2.nii.gz"))

    # 4. Crop
    println("  Cropping...")
    crop_beg = (16, 16, 16); crop_size = (32, 32, 32)
    batch_crop = MedImages.crop_mi(batch, crop_beg, crop_size, Linear_en)
    res_crop = unbatch_medimage(batch_crop)
    save_visual(res_crop[1], joinpath(MI_BATCHED_DIR, "cropped_$(crop_size)_img1"))
    save_visual(res_crop[2], joinpath(MI_BATCHED_DIR, "cropped_$(crop_size)_img2"))
    sitk.WriteImage(sitk_crop(s1, crop_beg, crop_size), joinpath(SITK_BATCHED_DIR, "cropped_$(crop_size)_img1.nii.gz"))
    sitk.WriteImage(sitk_crop(s2, crop_beg, crop_size), joinpath(SITK_BATCHED_DIR, "cropped_$(crop_size)_img2.nii.gz"))

    # 5. Pad
    println("  Padding...")
    pad_amt = (5, 5, 5)
    batch_pad = MedImages.pad_mi(batch, pad_amt, pad_amt, 0.0, Linear_en)
    res_pad = unbatch_medimage(batch_pad)
    save_visual(res_pad[1], joinpath(MI_BATCHED_DIR, "padded_$(pad_amt)_img1"))
    save_visual(res_pad[2], joinpath(MI_BATCHED_DIR, "padded_$(pad_amt)_img2"))
    sitk.WriteImage(sitk_pad(s1, pad_amt, pad_amt, 0.0), joinpath(SITK_BATCHED_DIR, "padded_$(pad_amt)_img1.nii.gz"))
    sitk.WriteImage(sitk_pad(s2, pad_amt, pad_amt, 0.0), joinpath(SITK_BATCHED_DIR, "padded_$(pad_amt)_img2.nii.gz"))

    # 6. Affine Shear
    println("  Affining (Shear)...")
    mat_id = create_affine_matrix(); shear_val = (0.5, 0.0, 0.0); mat_shear = create_affine_matrix(shear=shear_val)
    batch_affine = MedImages.affine_transform_mi(batch, [mat_id, mat_shear], Linear_en)
    res_affine = unbatch_medimage(batch_affine)
    save_visual(res_affine[1], joinpath(MI_BATCHED_DIR, "affine_identity_img1"))
    save_visual(res_affine[2], joinpath(MI_BATCHED_DIR, "affine_sheared_$(shear_val)_img2"))
    sitk.WriteImage(s1, joinpath(SITK_BATCHED_DIR, "affine_identity_img1.nii.gz")) 
    sitk.WriteImage(sitk_shear(s2, shear_val, sitk.sitkLinear), joinpath(SITK_BATCHED_DIR, "affine_sheared_$(shear_val)_img2.nii.gz"))

    # 7. Resample to Spacing
    println("  Resampling to Spacing...")
    spacings = [(2.0, 2.0, 2.0), (2.0, 2.0, 2.0)]
    batch_space = MedImages.resample_to_spacing(batch, spacings, Linear_en)
    res_space = unbatch_medimage(batch_space)
    save_visual(res_space[1], joinpath(MI_BATCHED_DIR, "resample_spacing_2mm_img1"))
    save_visual(res_space[2], joinpath(MI_BATCHED_DIR, "resample_spacing_2mm_img2"))
    sitk.WriteImage(sitk_resample_to_spacing(s1, (2.0, 2.0, 2.0), sitk.sitkLinear), joinpath(SITK_BATCHED_DIR, "resample_spacing_2mm_img1.nii.gz"))
    sitk.WriteImage(sitk_resample_to_spacing(s2, (2.0, 2.0, 2.0), sitk.sitkLinear), joinpath(SITK_BATCHED_DIR, "resample_spacing_2mm_img2.nii.gz"))

    # 8. Resample to Image
    println("  Resampling to Image...")
    ref1 = MedImage(voxel_data=zeros(Float32, 32, 32, 32), origin=(10.0,10.0,10.0), spacing=(1.0,1.0,1.0), direction=img1.direction, image_type=img1.image_type, image_subtype=img1.image_subtype, patient_id="ref1", current_device=img1.current_device, date_of_saving=now(), acquistion_time=now())
    ref2 = MedImage(voxel_data=zeros(Float32, 32, 32, 32), origin=(-10.0,-10.0,-10.0), spacing=(1.0,1.0,1.0), direction=img2.direction, image_type=img2.image_type, image_subtype=img2.image_subtype, patient_id="ref2", current_device=img2.current_device, date_of_saving=now(), acquistion_time=now())
    ref_batch = create_batched_medimage([ref1, ref2])
    batch_res_img = MedImages.resample_to_image(ref_batch, batch, Linear_en)
    res_img = unbatch_medimage(batch_res_img)
    save_visual(res_img[1], joinpath(MI_BATCHED_DIR, "resample_to_ref_origin10_img1"))
    save_visual(res_img[2], joinpath(MI_BATCHED_DIR, "resample_to_ref_origin-10_img2"))
    sitk.WriteImage(sitk.Resample(s1, medimage_to_sitk(ref1), sitk.Transform(), sitk.sitkLinear, 0.0, s1.GetPixelIDValue()), joinpath(SITK_BATCHED_DIR, "resample_to_ref_origin10_img1.nii.gz"))
    sitk.WriteImage(sitk.Resample(s2, medimage_to_sitk(ref2), sitk.Transform(), sitk.sitkLinear, 0.0, s2.GetPixelIDValue()), joinpath(SITK_BATCHED_DIR, "resample_to_ref_origin-10_img2.nii.gz"))


    # --- NON-BATCHED TRANSFORMATIONS ---
    println("Processing NON-BATCHED transformations...")
    
    save_visual(MedImages.rotate_mi(img2, 3, 45.0, Linear_en), joinpath(MI_NON_BATCHED_DIR, "rotated_45deg_axis3_img2"))
    sitk.WriteImage(rotation3d_sitk(s2, 3, 45.0, sitk.sitkLinear), joinpath(SITK_NON_BATCHED_DIR, "rotated_45deg_axis3_img2.nii.gz"))

    save_visual(MedImages.scale_mi(img1, 0.5, Linear_en), joinpath(MI_NON_BATCHED_DIR, "scaled_0.5x_img1"))
    sitk.WriteImage(sitk_scale(s1, 0.5, sitk.sitkLinear), joinpath(SITK_NON_BATCHED_DIR, "scaled_0.5x_img1.nii.gz"))

    save_visual(MedImages.translate_mi(img1, 10, 1, Linear_en), joinpath(MI_NON_BATCHED_DIR, "translated_10_axis1_img1"))
    sitk.WriteImage(sitk_translate(s1, 10, 0), joinpath(SITK_NON_BATCHED_DIR, "translated_10_axis1_img1.nii.gz"))

    save_visual(MedImages.crop_mi(img1, crop_beg, crop_size, Linear_en), joinpath(MI_NON_BATCHED_DIR, "cropped_$(crop_size)_img1"))
    sitk.WriteImage(sitk_crop(s1, crop_beg, crop_size), joinpath(SITK_NON_BATCHED_DIR, "cropped_$(crop_size)_img1.nii.gz"))

    save_visual(MedImages.pad_mi(img1, pad_amt, pad_amt, 0.0, Linear_en), joinpath(MI_NON_BATCHED_DIR, "padded_$(pad_amt)_img1"))
    sitk.WriteImage(sitk_pad(s1, pad_amt, pad_amt, 0.0), joinpath(SITK_NON_BATCHED_DIR, "padded_$(pad_amt)_img1.nii.gz"))

    save_visual(MedImages.resample_to_spacing(img1, (2.0, 2.0, 2.0), Linear_en), joinpath(MI_NON_BATCHED_DIR, "resample_spacing_2mm_img1"))
    sitk.WriteImage(sitk_resample_to_spacing(s1, (2.0, 2.0, 2.0), sitk.sitkLinear), joinpath(SITK_NON_BATCHED_DIR, "resample_spacing_2mm_img1.nii.gz"))

    println("Done! Results in $ROOT_OUTPUT_DIR")
end

main()
