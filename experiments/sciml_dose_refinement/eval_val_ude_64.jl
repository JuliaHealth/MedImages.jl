using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, Lux, LuxCUDA, CUDA, Random, ComponentArrays, NIfTI, Serialization, Statistics, Optimisers, SciMLSensitivity

const λ_phys = Float32(log(2) / 159.5) 
const k10_pop = Float32(log(2) / 40.0) 
const f_pop = 1.0f0  
const k_in_pop = 0.01f0  
const k_out_pop = 0.02f0 
const k3 = 0.05f0 
const k4 = 0.01f0 
const DOSE_CONV = 8.478f-8 

function ResBlockNorm(channels::Int)
    return SkipConnection(Chain(Conv((3, 3, 3), channels => channels, pad=1), GroupNorm(channels, 8, relu), Conv((3, 3, 3), channels => channels, pad=1), GroupNorm(channels, 8)), +)
end

function build_ude_no_approx_64()
    width, depth = 32, 3
    layers = Any[Parallel(+, Conv((3, 3, 3), 1 => width, pad=1, relu), Conv((3, 3, 3), 1 => width, pad=1, relu))]
    for _ in 1:depth; push!(layers, ResBlockNorm(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

function fix_parameters(θ)
    if θ isa ComponentArray
        return NamedTuple(θ)
    else
        return θ
    end
end

function predict_ude(model, θ, st, A0, den_p, vol_p)
    p_s = size(A0, 1)
    dev = Lux.gpu_device()
    CUDA.allowscalar(true)
    u0 = ComponentArray(A_blood=dev(Float32[sum(A0)*0.05f0]), A_free=A0.*0.45f0, A_bound=A0.*0.50f0, DOSE=zero(A0))
    CUDA.allowscalar(false)
    function ude_func(u, p, t)
        A_t = u.A_free .+ u.A_bound
        A_t_std = (A_t .- mean(A_t)) ./ (std(A_t) + 1f-6)
        nn_out, _ = Lux.apply(model, (reshape(A_t_std, p_s, p_s, p_s, 1, 1), reshape(den_p, p_s, p_s, p_s, 1, 1)), p, st)
        dD = softplus.((A_t .* DOSE_CONV) ./ (vol_p .* den_p .+ 1f-4) .+ reshape(nn_out, p_s, p_s, p_s))
        return ComponentArray(A_blood=-(k10_pop+λ_phys)*u.A_blood, A_free=-(k_out_pop+λ_phys)*u.A_free, A_bound=-(k4+λ_phys)*u.A_bound, DOSE=dD)
    end
    prob = ODEProblem(ude_func, u0, (0.0f0, 300.0f0), θ)
    sol = solve(prob, Tsit5(), saveat=[300.0f0], reltol=1f-2, abstol=1f-2)
    return sol.u[end].DOSE
end

function run_eval_val_64()
    dev = Lux.gpu_device(); rng = Random.default_rng()
    
    cp64 = "data/checkpoints/UDE_NO_APPROX_64/model_best_UDE_NO_APPROX_64.jls"
    if !isfile(cp64)
        println("Checkpoint not found at $cp64")
        return
    end

    θ_noapp_raw = dev(deserialize(cp64))
    θ_noapp = fix_parameters(θ_noapp_raw)
    m_noapp = build_ude_no_approx_64(); _, st_noapp = Lux.setup(rng, m_noapp); st_noapp = dev(st_noapp)

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
    
    pearsons = Float64[]
    maes = Float64[]

    println("Evaluating UDE 64x64x64 on validation set...")

    for pat in val_cases
        pat_dir = joinpath(dataset_dir, pat)
        if !isdir(pat_dir)
            continue
        end

        ct_f = niread(joinpath(pat_dir, "ct.nii.gz"))
        sp_f = niread(joinpath(pat_dir, "spect.nii.gz"))
        mc_f = niread(joinpath(pat_dir, "dosemap_mc.nii.gz"))

        ct_i = ndims(ct_f) == 3 ? ct_f : ct_f[:,:,:,1]; sp_i = ndims(sp_f) == 3 ? sp_f : sp_f[:,:,:,1]; mc_i = ndims(mc_f) == 3 ? mc_f : mc_f[:,:,:,1]
        
        # Center crop 64^3
        cx, cy, cz = size(ct_i) .÷ 2
        xr, yr, zr = cx-31:cx+32, cy-31:cy+32, cz-31:cz+32
        
        hu_to_den(hu) = hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) : 1.0f0 + 0.0007f0 * Float32(hu)
        den_p = hu_to_den.(ct_i[xr,yr,zr]); vol_p = Float32(prod(ct_f.header.pixdim[2:4])); A0 = dev(Float32.(sp_i[xr,yr,zr]))

        target = mc_i[xr,yr,zr]
        if maximum(target) < 1e-3
            continue
        end

        ude_p = predict_ude(m_noapp, θ_noapp, st_noapp, A0, dev(den_p), vol_p)
        pred = Array(ude_p)
        
        c_ude = cor(reshape(pred, :), reshape(target, :))
        mae_ude = mean(abs.(pred .- target))
        
        push!(pearsons, c_ude)
        push!(maes, mae_ude)
        
        println("Patient: $pat | Pearson: $(round(c_ude, digits=4)) | MAE: $(round(mae_ude, digits=4))")
    end
    
    println("----------------------------------------")
    println("Mean Pearson: $(round(mean(pearsons), digits=4))")
    println("Mean MAE: $(round(mean(maes), digits=4))")
end
run_eval_val_64()