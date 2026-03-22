using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations
using Lux, LuxCUDA, CUDA
using Random
using ComponentArrays
using NIfTI
using Serialization
using Optimisers

println("Starting Evaluation...")

# Physical Constants (Exactly match train_ude.jl)
const λ_phys = Float32(log(2) / 159.5) 
const k10_pop = Float32(log(2) / 40.0) 
const f_pop = 1.0f0  
const k_in_pop = 0.01f0  
const k_out_pop = 0.02f0 
const k3 = 0.05f0 
const k4 = 0.01f0 
const B_MAX_val = 1000.0f0 
const DOSE_CONV = 8.478f-8 

function load_patient_data(patient_dir::String)
    ct_path = joinpath(patient_dir, "ct.nii.gz")
    spect_path = joinpath(patient_dir, "spect.nii.gz")
    dose_path = joinpath(patient_dir, "dosemap_mc.nii.gz")
    approx_path = joinpath(patient_dir, "dosemap_approx.nii.gz")
    if !isfile(ct_path) || !isfile(spect_path) || !isfile(dose_path) || !isfile(approx_path); return nothing; end
    ct_full = niread(ct_path)
    spect_full = niread(spect_path)
    dose_full = niread(dose_path)
    approx_full = niread(approx_path)
    function extract_3d(arr)
        nd = ndims(arr); if nd == 3; return arr; end; return arr[:, :, :, 1]
    end
    return ct_full, extract_3d(ct_full), extract_3d(spect_full), extract_3d(dose_full), extract_3d(approx_full)
end

function hu_to_density(hu::Real)
    if hu <= 0; return max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)); else; return 1.0f0 + 0.0007f0 * Float32(hu); end
end

function ResBlock(channels::Int)
    return SkipConnection(
        Chain(
            Conv((3, 3, 3), channels => channels, pad=1, relu),
            Conv((3, 3, 3), channels => channels, pad=1)
        ),
        +
    )
end

function build_neural_transport_model()
    width = 16
    branch_A = Conv((3, 3, 3), 1 => width, pad=1, relu)
    branch_ρ = Conv((3, 3, 3), 1 => width, pad=1, relu)
    branch_approx = Conv((3, 3, 3), 1 => width, pad=1, relu)
    parallel_branches = Parallel(+, branch_A, branch_ρ, branch_approx)
    model = Chain(
        parallel_branches,
        ResBlock(width),
        ResBlock(width),
        Conv((3, 3, 3), width => 1, pad=1, tanh)
    )
    return model
end

const rng = Random.default_rng()
Random.seed!(rng, 42)
const NN_model = build_neural_transport_model()
const ps, st_init = Lux.setup(rng, NN_model)

function safe_format_channels(A, ρ, approx)
    return (reshape(A, size(A)..., 1, 1), reshape(ρ, size(ρ)..., 1, 1), reshape(approx, size(approx)..., 1, 1))
end

function predict_dosemap(p_state, theta_opt, A0)
    mass_map = p_state.vol_map .* p_state.ρ_map
    ρ_map = p_state.ρ_map
    approx_dose_map = p_state.approx_dose_map
    dev = Lux.gpu_device()
    u0 = ComponentArray(A_blood=dev(Float32[sum(A0) * 0.05f0]), A_free=A0 .* 0.45f0, A_bound=A0 .* 0.50f0, DOSE=zero(A0))
    tspan = (0.0f0, 300.0f0) 
    st_fixed = p_state.st
    function ude_func_outer(u, p, t)
        total_A0 = sum(A0) + 1f-6
        voxel_uptake_fraction = A0 ./ total_A0
        voxel_in = (f_pop * k_in_pop * u.A_blood) .* voxel_uptake_fraction
        dA_blood = - (k10_pop + λ_phys) * u.A_blood .- (f_pop * k_in_pop * u.A_blood) .+ sum(k_out_pop .* u.A_free)
        dA_free  = voxel_in .- (k_out_pop .* u.A_free) .- (λ_phys .* u.A_free)
        dA_bound = (k3 .* u.A_free .* (1.0f0 .- u.A_bound ./ B_MAX_val)) .- (k4 .* u.A_bound) .- (λ_phys .* u.A_bound)
        A_total = u.A_free .+ u.A_bound
        nn_out, _ = Lux.apply(NN_model, safe_format_channels(A_total, ρ_map, approx_dose_map), p, st_fixed)
        dD_phys = (A_total .* DOSE_CONV .* exp.(reshape(nn_out, size(A_total)))) ./ (mass_map .+ 1f-4)
        return ComponentArray(A_blood=dA_blood, A_free=dA_free, A_bound=dA_bound, DOSE=ifelse.(ρ_map .< 0.1f0, 0.0f0, dD_phys))
    end
    prob = ODEProblem(ude_func_outer, u0, tspan, theta_opt)
    sol = solve(prob, VCABM(), saveat=[300.0f0], reltol=1f-3, abstol=1f-3)
    return sol.u[end].DOSE
end

function run_eval()
    dev = Lux.gpu_device()
    println("Device: $dev")
    model_path = "elsarticle/dosimetry/model_best.jls"
    if !isfile(model_path); println("Model not found at $model_path"); return; end
    θ = deserialize(model_path)
    θ = dev(θ)
    println("Model loaded.")
    
    dataset_dir = "elsarticle/dosimetry/data"
    patients = sort(filter(isdir, readdir(dataset_dir, join=true)))
    
    rng_split = Random.Xoshiro(42)
    shuffle!(rng_split, patients)
    split_idx = Int(round(0.8 * length(patients)))
    val_patients = patients[split_idx+1:end]
    
    out_dir = "elsarticle/dosimetry/vis_results"
    mkpath(out_dir)
    println("Processing $(length(val_patients)) validation patients...")
    
    for (i, patient_dir) in enumerate(val_patients)
        pat_name = basename(patient_dir)
        println("[$i/$(length(val_patients))] Patient: $pat_name")
        data = load_patient_data(patient_dir)
        if data === nothing; println("  Skipped (missing files)"); continue; end
        ct_full, ct_img, spect_img, dosemap_mc, dosemap_approx = data
        mc_max = maximum(dosemap_mc); approx_max = maximum(dosemap_approx)
        scale = mc_m = mc_max / max(approx_max, 1f-6); target_dose = dosemap_mc ./ (scale + 1f-6)
        
        p_x, p_y, p_z = min.(size(ct_img), 64)
        cx, cy, cz = size(ct_img) .÷ 2; hx, hy, hz = (p_x, p_y, p_z) .÷ 2
        x_s = clamp(cx - hx, 1, max(1, size(ct_img, 1) - p_x + 1))
        y_s = clamp(cy - hy, 1, max(1, size(ct_img, 2) - p_y + 1))
        z_s = clamp(cz - hz, 1, max(1, size(ct_img, 3) - p_z + 1))
        xr, yr, zr = x_s:x_s+p_x-1, y_s:y_s+p_y-1, z_s:z_s+p_z-1
        
        p_state = (vol_map=dev(fill(Float32(prod(ct_full.header.pixdim[2:4])), (p_x, p_y, p_z))), ρ_map=dev(hu_to_density.(ct_img[xr,yr,zr])), approx_dose_map=dev(Float32.(dosemap_approx[xr,yr,zr])), st=dev(st_init))
        
        println("  Running inference...")
        predicted_dose = predict_dosemap(p_state, θ, dev(Float32.(spect_img[xr,yr,zr])))
        
        pat_dir = joinpath(out_dir, pat_name); mkpath(pat_dir)
        write(joinpath(pat_dir, "ct.bin"), Array(ct_img[xr,yr,zr]))
        write(joinpath(pat_dir, "pred.bin"), Array(predicted_dose))
        write(joinpath(pat_dir, "orig.bin"), Array(target_dose[xr,yr,zr]))
        write(joinpath(pat_dir, "approx.bin"), Array(dosemap_approx[xr,yr,zr]))
        write(joinpath(pat_dir, "dims.txt"), join([p_x, p_y, p_z], ","))
        println("  Done.")
        CUDA.reclaim(); GC.gc()
    end
end
run_eval()
