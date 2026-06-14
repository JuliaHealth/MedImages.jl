using Pkg
Pkg.activate("/home/user/MedImages.jl")

using NIfTI, Statistics, Random

const DOSE_CONV = 8.478f-8 

function hu_to_den(hu)
    return hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) : 1.0f0 + 0.0007f0 * Float32(hu)
end

function evaluate_analytical_baseline()
    val_cases = String[]
    open("experiments/sciml_dose_refinement/splits.txt", "r") do f
        in_val = false
        for line in eachline(f)
            if occursin("VALIDATION:", line); in_val = true; continue; end
            if occursin("TRAINING:", line); in_val = false; continue; end
            if in_val && strip(line) != ""; push!(val_cases, strip(line)); end
        end
    end

    dataset_dir = "data/dosimetry_data"
    pearsons = Float64[]; maes = Float64[]

    println("Evaluating Analytical Baseline on validation set...")
    for pat in val_cases
        pat_dir = joinpath(dataset_dir, pat); if !isdir(pat_dir); continue; end
        ct_f = niread(joinpath(pat_dir, "ct.nii.gz")); sp_f = niread(joinpath(pat_dir, "spect.nii.gz")); mc_f = niread(joinpath(pat_dir, "dosemap_mc.nii.gz"))
        
        extract(f) = ndims(f) == 3 ? f : f[:,:,:,1]
        ct_i = extract(ct_f); sp_i = extract(sp_f); mc_i = extract(mc_f)
        cx, cy, cz = size(ct_i) .÷ 2; xr, yr, zr = cx-31:cx+32, cy-31:cy+32, cz-31:cz+32
        
        target = mc_i[xr,yr,zr]; if maximum(target) < 1e-3; continue; end
        
        den = hu_to_den.(ct_i[xr,yr,zr])
        vol = Float32(prod(ct_f.header.pixdim[2:4]))
        spect = Float32.(sp_i[xr,yr,zr])
        
        # Analytical Baseline Equation
        pred = (spect .* DOSE_CONV) ./ (vol .* den .+ 1f-4)

        println("Debug: $pat | spect max=$(maximum(spect)) | pred max=$(maximum(pred)) | target max=$(maximum(target))")
        c = cor(reshape(pred, :), reshape(target, :))
        m = mean(abs.(pred .- target))
        push!(pearsons, c); push!(maes, m)
        println("  $pat | Pearson: $(round(c, digits=4)) | MAE: $(round(m, digits=4))")
    end
    println("\n--- Analytical Baseline Results ---")
    println("Mean Pearson: $(round(mean(pearsons), digits=4))")
    println("Mean MAE:     $(round(mean(maes), digits=4))")
end

evaluate_analytical_baseline()
