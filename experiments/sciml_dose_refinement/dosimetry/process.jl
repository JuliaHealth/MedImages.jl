using Pkg
Pkg.activate(".")
using MedImages

directory = "test_data/doses"
pre_dir = joinpath(directory, "pre")
post_dir = joinpath(directory, "post")

mkpath(pre_dir)
mkpath(post_dir)

ct_path = joinpath(directory, "CT.nii.gz")
println("Loading $ct_path as fixed image...")
im_fixed = load_image(ct_path)

files_to_process = filter(x -> endswith(x, ".nii.gz"), readdir(directory))

for file in files_to_process
    file_path = joinpath(directory, file)
    println("Loading $file_path...")
    im_moving = load_image(file_path)
    
    # 1. Resample
    if occursin("liver", lowercase(file)) || occursin("mask", lowercase(file)) || occursin("seg", lowercase(file))
        interp = MedImages.Nearest_neighbour_en
        println("Using Nearest Neighbour interpolation for $file")
    else
        interp = MedImages.Linear_en
        println("Using Linear interpolation for $file")
    end
    
    if file == "CT.nii.gz"
        im_resampled = im_moving
    else
        println("Resampling $file...")
        im_resampled = resample_to_image(im_fixed, im_moving, interp)
    end
    
    pre_path = joinpath(pre_dir, file)
    println("Saving resampled image to $pre_path")
    MedImages.create_nii_from_medimage(im_resampled, pre_path)
    
    # 2. Rotate 60 degrees in y axis
    println("Rotating $file by 60 degrees in the y axis...")
    # rotate_mi(image::MedImage, axis::Int, angle::Float64, Interpolator::Interpolator_enum, crop::Bool=true)
    im_rotated = rotate_mi(im_resampled, 2, 60.0, interp, true)
    
    post_path = joinpath(post_dir, file)
    println("Saving rotated image to $post_path")
    MedImages.create_nii_from_medimage(im_rotated, post_path)
end

println("Done processing all images.")
