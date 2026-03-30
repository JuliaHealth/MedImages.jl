using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using Optimisers
using SciMLSensitivity
using Lux, LuxCUDA
using Zygote
using Random
using ComponentArrays
using NIfTI
using Serialization
using Statistics

# Physical Constants (Fixed)
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

function build_model(width::Int)
    branch_A = Conv((3, 3, 3), 1 => width, pad=1, relu)
    branch_ρ = Conv((3, 3, 3), 1 => width, pad=1, relu)
    branch_approx = Conv((3, 3, 3), 1 => width, pad=1, relu)
    parallel_branches = Parallel(+, branch_A, branch_ρ, branch_approx)
    model = Chain(
        parallel_branches,
        Conv((3, 3, 3), width => width*2, pad=1, relu),
        Conv((3, 3, 3), width*2 => width*2, pad=1, relu), # Extra depth
        Conv((3, 3, 3), width*2 => 1, pad=1, tanh)
    )
    return model
end

function safe_format_channels(A, ρ, approx)
    return (reshape(A, size(A)..., 1, 1), reshape(ρ, size(ρ)..., 1, 1), reshape(approx, size(approx)..., 1, 1))
end

function predict_dosemap(u_model, p_state, theta_opt, A0)
    mass_map = p_state.vol_map .* p_state.ρ_map
    ρ_map = p_state.ρ_map
    approx_dose_map = p_state.approx_dose_map
    dev = Lux.gpu_device()
    u0 = ComponentArray(A_blood=dev(Float32[sum(A0)*0.05f0]), A_free=A0.*0.45f0, A_bound=A0.*0.50f0, DOSE=zero(A0))
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
        nn_out, _ = Lux.apply(u_model, safe_format_channels(A_total, ρ_map, approx_dose_map), p, st_fixed)
        dD_phys = (A_total .* DOSE_CONV .* exp.(reshape(nn_out, size(A_total)))) ./ (mass_map .+ 1f-4)
        return ComponentArray(A_blood=dA_blood, A_free=dA_free, A_bound=dA_bound, DOSE=ifelse.(ρ_map .< 0.1f0, 0.0f0, dD_phys))
    end
    prob = ODEProblem(ude_func_outer, u0, tspan, theta_opt)
    sol = solve(prob, VCABM(), saveat=[300.0f0], reltol=1f-3, abstol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true))
    return sol.u[end].DOSE
end

function run_experiment(lr, width, epochs=5)
    println("\n>>> Experiment: LR=$lr, Width=$width")
    dev = Lux.gpu_device()
    curr_model = build_model(width)
    ps, st = Lux.setup(Random.default_rng(), curr_model)
    θ = dev(ComponentArray(ps))
    opt_state = Optimisers.setup(Optimisers.Adam(lr), θ)
    
    dataset_dir = "/home/user/MedImages.jl/elsarticle/dosimetry/data/"
    patients = sort(filter(isdir, readdir(dataset_dir, join=true)))
    train_pats = patients[1:min(8, length(patients))]
    
    final_loss = 0.0f0
    for epoch in 1:epochs
        epoch_mae = 0.0f0
        epoch_mse = 0.0f0
        count = 0
        for pat in train_pats
            data = load_patient_data(pat)
            if data === nothing; continue; end
            ct_full, ct_img, spect_img, dosemap_mc, dosemap_approx = data
            mc_max = maximum(dosemap_mc); approx_max = maximum(dosemap_approx)
            scale = mc_max / max(approx_max, 1f-6)
            target = dosemap_mc ./ (scale + 1f-6)
            
            cx, cy, cz = size(ct_img) .÷ 2
            xr, yr, zr = cx-15:cx+16, cy-15:cy+16, cz-15:cz+16
            
            p_state = (vol_map=dev(fill(Float32(prod(ct_full.header.pixdim[2:4])), (32,32,32))), ρ_map=dev(hu_to_density.(ct_img[xr,yr,zr])), approx_dose_map=dev(Float32.(dosemap_approx[xr,yr,zr])), st=dev(st))
            spect_patch = dev(Float32.(spect_img[xr,yr,zr]))
            target_patch = dev(Float32.(target[xr,yr,zr]))

            loss_val, gs = Zygote.withgradient(t -> begin
                pred = predict_dosemap(curr_model, p_state, t, spect_patch)
                sum(abs.(pred .- target_patch)) / length(target_patch)
            end, θ)
            
            if !any(isnan.(gs[1]))
                opt_state, θ = Optimisers.update(opt_state, θ, gs[1])
                epoch_mae += loss_val
                # Log MSE for analysis
                pred_final = predict_dosemap(curr_model, p_state, θ, spect_patch)
                epoch_mse += sum((pred_final .- target_patch).^2) / length(target_patch)
                count += 1
            end
            CUDA.reclaim(); GC.gc()
        end
        avg_mae = epoch_mae/count
        avg_mse = epoch_mse/count
        println("  Epoch $epoch MAE: $avg_mae, MSE: $avg_mse")
        final_loss = avg_mae
    end
    return final_loss
end

lrs = [1f-4, 5f-5]
widths = [8, 16]

results = []
for w in widths, lr in lrs
    l = run_experiment(lr, w)
    push!(results, (lr=lr, width=w, loss=l))
end

println("\n--- Summary Results ---")
for r in results
    println("Width: $(r.width), LR: $(r.lr) => Final MAE: $(r.loss)")
end
