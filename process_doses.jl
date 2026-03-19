using Pkg
Pkg.activate("/home/user/MedImages.jl")
Pkg.instantiate()
using MedImages

function main()
    base_dir = joinpath("/home/user/MedImages.jl", "test_data", "doses")
    pre_dir = joinpath(base_dir, "pre")
    post_dir = joinpath(base_dir, "post")

    mkpath(pre_dir)
    mkpath(post_dir)

    # target image
    ct_path = joinpath(base_dir, "CT.nii.gz")
    fixed_image = load_image(ct_path, "CT")

    files_to_process = [
        joinpath(base_dir, "Dosemap.nii.gz"),
        joinpath(base_dir, "NM_Vendor.nii.gz"),
        joinpath(base_dir, "SPECT_Recon_WholeBody.nii.gz"),
        joinpath(base_dir, "dosemap_Lu177_v2.nii.gz"),
        joinpath(base_dir, "dosemap_Lu177_v2_smoothed12mm.nii.gz"),
        joinpath(base_dir, "liver_seg", "liver.nii.gz"),
        ct_path
    ]

    for file_path in files_to_process
        println("Processing $file_path")
        filename = basename(file_path)
        
        # determine interp type
        interp_type = Linear_en
        if filename == "liver.nii.gz"
            interp_type = Nearest_neighbour_en
        end
        
        # load image
        type_str = filename == "CT.nii.gz" ? "CT" : "PET"
        moving_image = load_image(file_path, type_str)

        # Resample
        if filename == "CT.nii.gz"
            resampled_image = fixed_image
        else
            resampled_image = resample_to_image(fixed_image, moving_image, interp_type)
        end
        
        # Save to pre
        pre_path = joinpath(pre_dir, filename)
        create_nii_from_medimage(resampled_image, pre_path)
        
        # Rotate 60 degrees in y axis (axis=2)
        rotated_image = rotate_mi(resampled_image, 2, 60.0, interp_type)
        
        # Save to post
        post_path = joinpath(post_dir, filename)
        create_nii_from_medimage(rotated_image, post_path)
    end

    println("Done!")
end

main()
