using Pkg
Pkg.activate("/home/user/MedImages.jl")
using MedImages
using NIfTI

src_dir = "/DATA/"
target_dir = "/home/user/MedImages.jl/data/dosimetry_data/"
mkpath(target_dir)

# The 56 cases from user
raw_cases = """FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat45(16.05.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat46(14.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat54(08.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat51(30.05.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat54(06.06.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat61(30.10.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat47(07.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat52(13.06.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat56(20.06.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat54(25.04.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat48(08.02.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat46(25.04.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_5__Pat47(07.11.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat51(17.10.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat49(18.04.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_6__Pat49(20.02.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat60(06.03.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_6__Pat47(06.02.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat60(29.08.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat44(07.12.2023)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat61(20.06.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_5__Pat49(07.11.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat45(22.02.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat47(18.04.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat47(25.01.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_7__Pat47(10.04.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat52(24.07.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat52(30.04.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat55(19.08.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat50(07.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat46(11.12.2023)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat49(07.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_4__Pat52(13.10.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat51(04.07.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat53(25.04.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_5__Pat52(24.11.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat58(16.05.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat55(06.06.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat54(25.07.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat52(01.09.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_7__Pat49(08.05.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat53(08.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_4__Pat51(06.02.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat46(01.02.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat56(02.05.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat58(27.06.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat60(23.01.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_4__Pat47(22.08.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat49(26.01.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat50(26.01.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat55(17.10.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat48(21.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat45(18.01.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat45(28.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat49(30.05.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat47(06.06.2024)"""

# Parsing and mapping
cases = []
for line in split(raw_cases, '\n')
    m = match(r"FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_(\d)__Pat(\d+)", line)
    if m !== nothing
        x, y = m.captures
        push!(cases, (x, y))
    end
end

for (x, y) in cases
    # Try different possible SPECT identifiers if Tc fails
    pat_dir_names = [
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_$(x)__Pat$(y)",
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Iodine_$(x)__Pat$(y)"
    ]
    
    found = false
    src_pat_path = ""
    target_pat_name = ""
    
    for dir_name in pat_dir_names
        if isdir(joinpath(src_dir, dir_name))
            src_pat_path = joinpath(src_dir, dir_name)
            target_pat_name = dir_name
            found = true
            break
        end
    end
    
    if !found
        println("Warning: Case Pat$y index $x not found in /DATA")
        continue
    end
    
    mc_path = joinpath(src_pat_path, "DOSEMAP", "dosemap.nii.gz")
    ct_path = joinpath(src_pat_path, "SPECT_DATA", "nifti_files", "CT.nii.gz")
    spect_path = joinpath(src_pat_path, "SPECT_DATA", "nifti_files", "NM_Vendor.nii.gz")
    approx_path = joinpath(src_pat_path, "SPECT_DATA", "nifti_files", "Dosemap.nii.gz")
    
    if !isfile(mc_path) || !isfile(ct_path) || !isfile(spect_path) || !isfile(approx_path)
        println("Warning: Missing files for $target_pat_name")
        continue
    end
    
    out_pat_dir = joinpath(target_dir, target_pat_name)
    if isdir(out_pat_dir) && isfile(joinpath(out_pat_dir, "dosemap_approx.nii.gz"))
        continue # Already processed
    end
    mkpath(out_pat_dir)
    
    println("Processing $target_pat_name...")
    
    try
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
    catch e
        println("  -> Error processing $target_pat_name: $e")
    end
end
println("Dataset preparation complete.")
