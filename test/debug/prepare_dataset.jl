using Pkg
Pkg.activate("/home/user/MedImages.jl")
using MedImages
using NIfTI

src_dir = "/DATA/"
target_dir = "/home/user/MedImages.jl/elsarticle/dosimetry/data/"
mkpath(target_dir)

for pat_name in readdir(src_dir)
    pat_path = joinpath(src_dir, pat_name)
    if !isdir(pat_path)
        continue
    end
    
    mc_path = joinpath(pat_path, "DOSEMAP", "dosemap.nii.gz")
    ct_path = joinpath(pat_path, "SPECT_DATA", "nifti_files", "CT.nii.gz")
    spect_path = joinpath(pat_path, "SPECT_DATA", "nifti_files", "NM_Vendor.nii.gz")
    approx_path = joinpath(pat_path, "SPECT_DATA", "nifti_files", "Dosemap.nii.gz")
    
    if !isfile(mc_path) || !isfile(ct_path) || !isfile(spect_path) || !isfile(approx_path)
        println("Missing files for $pat_name, skipping.")
        continue
    end
    
    out_pat_dir = joinpath(target_dir, pat_name)
    mkpath(out_pat_dir)
    
    println("Processing $pat_name...")
    
    # Load MC Dosemap as target geometry
    mc_im = load_image(mc_path, "PET")
    
    # Symlink target dosemap
    target_mc_link = joinpath(out_pat_dir, "dosemap_mc.nii.gz")
    if !isfile(target_mc_link)
        symlink(mc_path, target_mc_link)
    end
    
    # Process CT
    out_ct = joinpath(out_pat_dir, "ct.nii.gz")
    if !isfile(out_ct)
        ct_im = load_image(ct_path, "CT")
        ct_res = resample_to_image(mc_im, ct_im, Linear_en, -1000.0)
        create_nii_from_medimage(ct_res, out_ct)
    end
    
    # Process SPECT
    out_spect = joinpath(out_pat_dir, "spect.nii.gz")
    if !isfile(out_spect)
        spect_im = load_image(spect_path, "PET")
        # Ensure we do not add negative noise to spect, extrapolate with 0
        spect_res = resample_to_image(mc_im, spect_im, Linear_en, 0.0)
        create_nii_from_medimage(spect_res, out_spect)
    end
    
    # Process Approx Dosemap
    out_approx = joinpath(out_pat_dir, "dosemap_approx.nii.gz")
    if !isfile(out_approx)
        approx_im = load_image(approx_path, "PET")
        approx_res = resample_to_image(mc_im, approx_im, Linear_en, 0.0)
        create_nii_from_medimage(approx_res, out_approx)
    end
    println("  -> Done.")
end
println("Dataset prepared successfully.")
