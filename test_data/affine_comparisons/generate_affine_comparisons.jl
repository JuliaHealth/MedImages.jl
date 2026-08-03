using Pkg
Pkg.activate("/workspaces/MedImages.jl")

using MedImages
using MedImages.Resample_to_target
using MedImages.MedImage_data_struct
using PyCall
using LinearAlgebra

# Ensure SimpleITK is installed in the PyCall python environment!
try
    pyimport("SimpleITK")
catch e
    println("SimpleITK not found in PyCall. Installing...")
    run(`$(PyCall.python) -m pip install SimpleITK`)
end
sitk = pyimport("SimpleITK")

# Configuration
CT_PATH = "/workspaces/MedImages.jl/test_data/volume-0.nii.gz"
OUT_DIR = "/workspaces/MedImages.jl/test_data/affine_comparisons"

im = load_image(CT_PATH, "CT")
sitk_im = sitk.ReadImage(CT_PATH)

function run_comparison(name::String, rot_matrix, origin_translation, spacing_scale)
    println("Generating $name...")
    
    # 1. MedImages
    # Apply to moving image
    im_moving = MedImage(im, 
        origin = origin_translation, 
        spacing = spacing_scale,
        direction = rot_matrix
    )
    
    resampled_medimages = Resample_to_target.resample_to_image(im, im_moving, Linear_en, Float32(-1000.0))
    create_nii_from_medimage(resampled_medimages, joinpath(OUT_DIR, "medimages_$name.nii.gz"))

    # 2. SimpleITK
    # SimpleITK transform maps from FIXED to MOVING physical space
    # In SITK, if we just want the output to match MedImages where the Moving Image 
    # has modified origin/spacing/direction, we just create an SITK image with that metadata!
    sitk_moving = sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk_im))
    sitk_moving.SetOrigin(origin_translation)
    sitk_moving.SetSpacing(spacing_scale)
    # SITK expects 1D array for direction
    sitk_moving.SetDirection([rot_matrix[1], rot_matrix[4], rot_matrix[7],
                              rot_matrix[2], rot_matrix[5], rot_matrix[8],
                              rot_matrix[3], rot_matrix[6], rot_matrix[9]])
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_im)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000.0)
    # Default transform is identity. It will resample sitk_moving onto sitk_im's grid using their physical metadata
    sitk_resampled = resampler.Execute(sitk_moving)
    sitk.WriteImage(sitk_resampled, joinpath(OUT_DIR, "sitk_$name.nii.gz"))
end


println("Starting Generation of Affine Combinations...")

# Original parameters
orig_spacing = im.spacing
orig_origin = im.origin
orig_dir = reshape(collect(im.direction), 3, 3)

# 1. Pure Rotation (30 degrees around Z)
theta = 30.0 * pi / 180.0
rot_Z = [cos(theta) -sin(theta) 0.0; 
         sin(theta)  cos(theta) 0.0; 
         0.0         0.0        1.0]
rot_dir = Tuple(rot_Z * orig_dir)
run_comparison("01_pure_rotation", rot_dir, orig_origin, orig_spacing)

# 2. Pure Translation (+20mm X, -15mm Y, +10mm Z)
trans = (orig_origin[1] + 20.0, orig_origin[2] - 15.0, orig_origin[3] + 10.0)
run_comparison("02_pure_translation", Tuple(orig_dir), trans, orig_spacing)

# 3. Pure Shear (Shear X by 0.3 * Y)
shear_matrix = [1.0 0.0 0.0; 
                0.3 1.0 0.0; 
                0.0 0.0 1.0]
shear_dir = Tuple(shear_matrix * orig_dir)
run_comparison("03_pure_shear", shear_dir, orig_origin, orig_spacing)

# 4. Composite (Rotation + Translation + Shear + Scale)
scale = (orig_spacing[1] * 1.5, orig_spacing[2] * 0.8, orig_spacing[3] * 1.2)
comp_matrix = rot_Z * shear_matrix
comp_dir = Tuple(comp_matrix * orig_dir)
run_comparison("04_composite", comp_dir, trans, scale)

println("Saved all NIfTI pairs to $OUT_DIR")
