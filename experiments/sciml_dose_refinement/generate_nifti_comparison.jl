using Pkg
Pkg.activate("/home/user/MedImages.jl")

using NIfTI, Statistics

const DOSE_CONV = 8.478f-8 

function hu_to_den(hu)
    return hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) : 1.0f0 + 0.0007f0 * Float32(hu)
end

function generate_analytical_nifti()
    pat_name = "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat48"
    dataset_dir = "data/dosimetry_data"
    pat_dir = joinpath(dataset_dir, pat_name)
    
    ct_f = niread(joinpath(pat_dir, "ct.nii.gz"))
    sp_f = niread(joinpath(pat_dir, "spect.nii.gz"))
    
    extract(f) = ndims(f) == 3 ? f : f[:,:,:,1]
    ct_i = extract(ct_f)
    sp_i = extract(sp_f)
    vol_p = Float32(prod(ct_f.header.pixdim[2:4]))

    # Analytical Baseline Calculation
    den = hu_to_den.(ct_i)
    analytical_dose = (Float32.(sp_i) .* DOSE_CONV) ./ (vol_p .* den .+ 1f-4)

    # Save NIfTI
    out_dir = "val_outputs_comparison/Pat48"
    mkpath(out_dir)
    
    ni = NIVolume(analytical_dose)
    ni.header.pixdim = ct_f.header.pixdim
    niwrite(joinpath(out_dir, "baseline_analytical.nii.gz"), ni)
    
    # Copy existing ones for convenience
    cp(joinpath(pat_dir, "dosemap_mc.nii.gz"), joinpath(out_dir, "monte_carlo_gold.nii.gz"), force=true)
    cp("val_outputs_parallel/$(pat_name)_parallel.nii.gz", joinpath(out_dir, "ude_improved.nii.gz"), force=true)
    
    println("NIfTI comparison set generated in: $out_dir")
end

generate_analytical_nifti()
