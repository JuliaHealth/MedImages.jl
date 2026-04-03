using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations
using Optimization, OptimizationOptimJL, OptimizationOptimisers, Optimisers
using SciMLSensitivity
using Lux, LuxCUDA, CUDA
using Zygote
using Random
using ComponentArrays
using NIfTI
using Serialization
using Statistics

# Physical Constants
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
    ct_f = niread(ct_path); spect_f = niread(spect_path); dose_f = niread(dose_path); approx_f = niread(approx_path)
    extract(arr) = ndims(arr) == 3 ? arr : arr[:,:,:,1]
    return ct_f, extract(ct_f), extract(spect_f), extract(dose_f), extract(approx_f)
end

function hu_to_density(hu::Real)
    hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) : 1.0f0 + 0.0007f0 * Float32(hu)
end

function ResBlock(channels::Int)
    return SkipConnection(Chain(Conv((3, 3, 3), channels => channels, pad=1, relu), Conv((3, 3, 3), channels => channels, pad=1)), +)
end

function build_heavy_model(width::Int, depth::Int)
    layers = []
    # Initial processing
    branch_A = Conv((3, 3, 3), 1 => width, pad=1, relu)
    branch_ρ = Conv((3, 3, 3), 1 => width, pad=1, relu)
    branch_approx = Conv((3, 3, 3), 1 => width, pad=1, relu)
    push!(layers, Parallel(+, branch_A, branch_ρ, branch_approx))
    
    # Residual stack
    for _ in 1:depth
        push!(layers, ResBlock(width))
    end
    
    # Output projection
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

function predict_dosemap(u_model, p_state, theta_opt, A0)
    mass_map = p_state.vol_map .* p_state.ρ_map
    ρ_map = p_state.ρ_map
    approx_dose_map = p_state.approx_dose_map
    dev = Lux.gpu_device()
    u0 = ComponentArray(A_blood=dev(Float32[sum(A0)*0.05f0]), A_free=A0.*0.45f0, A_bound=A0.*0.50f0, DOSE=zero(A0))
    st_fixed = p_state.st
    
    function ude_func_outer(u, p, t)
        total_A0 = sum(A0) + 1f-6
        voxel_in = (f_pop * k_in_pop * u.A_blood) .* (A0 ./ total_A0)
        dA_blood = - (k10_pop + λ_phys) * u.A_blood .- sum(f_pop * k_in_pop * u.A_blood) .+ sum(k_out_pop .* u.A_free)
        dA_free  = voxel_in .- (k_out_pop .* u.A_free) .- (λ_phys .* u.A_free)
        dA_bound = (k3 .* u.A_free .* (1.0f0 .- u.A_bound ./ B_MAX_val)) .- (k4 .* u.A_bound) .- (λ_phys .* u.A_bound)
        A_total = u.A_free .+ u.A_bound
        
        # ADDITIVE POSITIVE MODEL: dD = softplus(Baseline + Residual)
        # This allows radiation to "spread" into voxels where A_total is zero.
        inputs = (reshape(A_total, size(A_total)..., 1, 1), reshape(ρ_map, size(ρ_map)..., 1, 1), reshape(approx_dose_map, size(approx_dose_map)..., 1, 1))
        nn_out, _ = Lux.apply(u_model, inputs, p, st_fixed)
        dD_base = (A_total .* DOSE_CONV) ./ (mass_map .+ 1f-4)
        dD_phys = softplus.(dD_base .+ reshape(nn_out, size(A_total)))
        
        return ComponentArray(A_blood=dA_blood, A_free=dA_free, A_bound=dA_bound, DOSE=ifelse.(ρ_map .< 0.1f0, 0.0f0, dD_phys))
    end
    
    prob = ODEProblem(ude_func_outer, u0, (0.0f0, 300.0f0), theta_opt)
    sol = solve(prob, VCABM(), saveat=[300.0f0], reltol=1f-3, abstol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true))
    return sol.u[end].DOSE
end

function gradient_loss(pred, target)
    gx_p = pred[2:end,:,:] .- pred[1:end-1,:,:]; gx_t = target[2:end,:,:] .- target[1:end-1,:,:]
    return sum(abs.(gx_p .- gx_t)) / length(gx_p)
end

function run_tuning_experiment(lr, width, depth, grad_weight, epochs=5)
    println("\n>>> START EXPERIMENT: LR=$lr, Width=$width, Depth=$depth, GradWeight=$grad_weight")
    dev = Lux.gpu_device()
    curr_model = build_heavy_model(width, depth)
    ps, st = Lux.setup(Random.default_rng(), curr_model)
    θ = dev(ComponentArray(ps))
    opt_state = Optimisers.setup(Optimisers.Adam(lr), θ)
    
    dataset_dir = "/home/user/MedImages.jl/elsarticle/dosimetry/data/"
    patients = sort(filter(isdir, readdir(dataset_dir, join=true)))
    train_pats = patients[1:min(10, length(patients))]
    
    for epoch in 1:epochs
        m_mae = 0.0f0; m_grad = 0.0f0; count = 0
        for pat in train_pats
            data = load_patient_data(pat); if data === nothing; continue; end
            ct_f, ct_i, spect_i, dose_mc, dose_app = data
            scale = maximum(dose_mc) / max(maximum(dose_app), 1f-6); target = dose_mc ./ (scale + 1f-6)
            
            # Use active regions for tuning
            valid = findall(x -> x > 1f-3, target)
            if isempty(valid); continue; end
            
            for patch_idx in 1:1 # 1 patch per pat to cover more patients quickly
                idx = rand(valid); cx, cy, cz = idx.I; p_size = 32; hx = p_size ÷ 2
                xr = clamp(cx-hx, 1, size(target,1)-p_size+1):clamp(cx-hx+p_size-1, p_size, size(target,1))
                yr = clamp(cy-hx, 1, size(target,2)-p_size+1):clamp(cy-hx+p_size-1, p_size, size(target,2))
                zr = clamp(cz-hx, 1, size(target,3)-p_size+1):clamp(cz-hx+p_size-1, p_size, size(target,3))
                
                p_state = (vol_map=dev(fill(Float32(prod(ct_f.header.pixdim[2:4])), (32,32,32))), ρ_map=dev(hu_to_density.(ct_i[xr,yr,zr])), approx_dose_map=dev(Float32.(dose_app[xr,yr,zr])), st=dev(st))
                spect_p = dev(Float32.(spect_i[xr,yr,zr])); target_p = dev(Float32.(target[xr,yr,zr]))

                l, gs = Zygote.withgradient(t -> begin
                    pred = predict_dosemap(curr_model, p_state, t, spect_p)
                    mae = sum(abs.(pred .- target_p)) / length(target_p)
                    grad = gradient_loss(pred, target_p)
                    m_mae += mae; m_grad += grad
                    (1.0f0 - grad_weight) * mae + grad_weight * grad
                end, θ)
                
                if !any(isnan.(gs[1]))
                    opt_state, θ = Optimisers.update(opt_state, θ, gs[1])
                    count += 1
                end
                CUDA.reclaim(); GC.gc()
            end
        end
        println("  Epoch $epoch | Avg MAE: $(m_mae/count) | Avg Grad: $(m_grad/count)")
    end
    # Save best of experiment
    serialize("model_tuning_W$(width)_D$(depth)_LR$(lr).jls", θ)
end

# GRID SEARCH
lrs = [1f-3, 5f-4]
widths = [16, 32]
depths = [3, 5]
grad_weights = [0.3f0, 0.6f0]

for w in widths, d in depths, lr in lrs, gw in grad_weights
    run_tuning_experiment(lr, w, d, gw)
end
