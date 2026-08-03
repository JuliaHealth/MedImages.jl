using Pkg
Pkg.activate("/workspaces/MedImages.jl")

using MedImages
using MedImages.Resample_to_target
using MedImages.MedImage_data_struct
using PyCall

sitk = pyimport("SimpleITK")

# Configuration
CT_PATH = "/workspaces/MedImages.jl/test_data/volume-0.nii.gz"
OUT_DIR = "/workspaces/MedImages.jl/test/visual_output/comparisons"
mkpath(OUT_DIR)

im = load_image(CT_PATH, "CT")
sitk_im = sitk.ReadImage(CT_PATH)

function compare_transform(name, rot_matrix, sitk_rot_matrix)
    println("Generating $name...")
    
    # 1. MedImages: Create version then resample to target
    im_moving = MedImage(
        voxel_data = im.voxel_data, origin = im.origin, spacing = im.spacing,
        direction = rot_matrix, image_type = im.image_type, image_subtype = im.image_subtype, 
        patient_id = im.patient_id, current_device = im.current_device, 
        date_of_saving = im.date_of_saving, acquistion_time = im.acquistion_time, 
        study_uid = im.study_uid, patient_uid = im.patient_uid, series_uid = im.series_uid, 
        study_description = im.study_description, legacy_file_name = im.legacy_file_name, 
        display_data = im.display_data, clinical_data = im.clinical_data, 
        is_contrast_administered = im.is_contrast_administered, metadata = im.metadata
    )
    resampled_medimages = Resample_to_target.resample_to_image(im, im_moving, MedImages.Linear_en, Float32(-1000.0))
    create_nii_from_medimage(resampled_medimages, joinpath(OUT_DIR, "medimages_$name.nii.gz"))

    # 2. SimpleITK
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(sitk_rot_matrix)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_im)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000.0)
    resampler.SetTransform(transform)
    sitk_resampled = resampler.Execute(sitk_im)
    sitk.WriteImage(sitk_resampled, joinpath(OUT_DIR, "sitk_$name.nii.gz"))
end

# Transform 1: Rotation 30 degrees
theta = 30.0 * pi / 180.0
compare_transform(
    "rotated30", 
    (cos(theta), -sin(theta), 0.0, sin(theta), cos(theta), 0.0, 0.0, 0.0, 1.0), 
    [cos(-theta), -sin(-theta), 0.0, sin(-theta), cos(-theta), 0.0, 0.0, 0.0, 1.0]
)

# Transform 2: Scale and Rotation
theta2 = -15.0 * pi / 180.0
compare_transform(
    "scale_rotate",
    (0.8*cos(theta2), -0.8*sin(theta2), 0.0, 0.8*sin(theta2), 0.8*cos(theta2), 0.0, 0.0, 0.0, 1.0),
    [1.25*cos(-theta2), -1.25*sin(-theta2), 0.0, 1.25*sin(-theta2), 1.25*cos(-theta2), 0.0, 0.0, 0.0, 1.0]
)

println("Saved NIfTI pairs to $OUT_DIR")
