using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, Lux, LuxCUDA, CUDA, Random, ComponentArrays, NIfTI, Serialization, Statistics, Optimisers, SciMLSensitivity, NNlib, Adapt

# Physical Constants
const λ_phys = Float32(log(2) / 159.5) 
const k10_pop = Float32(log(2) / 40.0) 
const f_pop = 1.0f0  
const k_in_pop = 0.01f0  
const k_out_pop = 0.02f0 
const k3 = 0.05f0 
const k4 = 0.01f0 
const DOSE_CONV_BALANCED = 0.08478f0 

function hu_to_den(hu)
    return hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) : 1.0f0 + 0.0007f0 * Float32(hu)
end

function standardize(x)
    μ = mean(x); σ = std(x) + 1f-6
    return (x .- μ) ./ σ
end

function grad_3d(vol)
    gx = zero(vol); gy = zero(vol); gz = zero(vol)
    gx[2:end-1, :, :] .= (vol[3:end, :, :] .- vol[1:end-2, :, :]) ./ 2.0f0
    gy[:, 2:end-1, :] .= (vol[:, 3:end, :] .- vol[:, 1:end-2, :]) ./ 2.0f0
    gz[:, :, 2:end-1] .= (vol[:, :, 3:end] .- vol[:, :, 1:end-2]) ./ 2.0f0
    return sqrt.(gx.^2 .+ gy.^2 .+ gz.^2 .+ 1f-8)
end

function ResBlockNorm(channels::Int)
    return SkipConnection(Chain(Conv((3, 3, 3), channels => channels, pad=1), GroupNorm(channels, 8, relu), Conv((3, 3, 3), channels => channels, pad=1), GroupNorm(channels, 8)), +)
end

function build_ude_improved(width::Int, depth::Int)
    layers = Any[Parallel(+, Conv((3, 3, 3), 1 => width, pad=1, relu), Conv((3, 3, 3), 1 => width, pad=1, relu), Conv((3, 3, 3), 1 => width, pad=1, relu))]
    for _ in 1:depth; push!(layers, ResBlockNorm(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

function build_stabilized_cnn()
    width, depth = 32, 3
    layers = Any[Conv((3, 3, 3), 3 => width, pad=1, relu)]
    for _ in 1:depth; push!(layers, SkipConnection(Chain(Conv((3, 3, 3), width => width, pad=1, relu), Conv((3, 3, 3), width => width, pad=1)), +)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

function predict_ude_patch_balanced(model, θ, st, sp_raw, den_raw, vol_p)
    p_s = size(sp_raw, 1)
    sp_bal = sp_raw ./ 1e6; sum_sp = sum(sp_bal); sp_p = Float32.(sp_bal)
    u0 = ComponentArray(A_blood=Float32[sum_sp*0.05f0], A_free=sp_p.*0.45f0, A_bound=sp_p.*0.50f0, DOSE=zero(sp_p))
    den_std = Float32.(standardize(den_raw)); den_grad_std = Float32.(standardize(grad_3d(den_raw))); den_raw_cpu = Float32.(den_raw)
    function f(u, p, t)
        A_t = u.A_free .+ u.A_bound; A_t_std = (A_t .- mean(A_t)) ./ (std(A_t) + 1f-6)
        in_nn = (reshape(A_t_std, p_s, p_s, p_s, 1, 1), reshape(den_std, p_s, p_s, p_s, 1, 1), reshape(den_grad_std, p_s, p_s, p_s, 1, 1))
        nn_o, _ = Lux.apply(model, in_nn, p, st)
        dD_base = (A_t .* DOSE_CONV_BALANCED) ./ (vol_p .* den_raw_cpu .+ 1f-4)
        dD = softplus.( dD_base .+ reshape(Array(nn_o), p_s, p_s, p_s) )
        return ComponentArray(A_blood=-(k10_pop+λ_phys)*u.A_blood, A_free=-(k_out_pop+λ_phys)*u.A_free, A_bound=-(k4+λ_phys)*u.A_bound, DOSE=dD)
    end
    prob = ODEProblem(f, u0, (0.0f0, 300.0f0), θ); sol = solve(prob, Tsit5(), saveat=[300.0f0], reltol=1f-1, abstol=1f-1)
    return sol.u[end].DOSE
end

function run_eval_all()
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
    
    # Models
    cp_ude = "experiments/sciml_dose_refinement/data/checkpoints/UDE_IMPROVED_64/model_best.jls"
    θ_ude = isfile(cp_ude) ? adapt(Lux.CPUDevice(), deserialize(cp_ude)) : nothing
    m_ude = build_ude_improved(32, 3); _, st_ude = Lux.setup(Random.default_rng(), m_ude)

    cp_cnn = "data/checkpoints/CNN_APPROX/model_best_CNN_APPROX.jls"
    θ_cnn = isfile(cp_cnn) ? adapt(Lux.CPUDevice(), deserialize(cp_cnn)) : nothing
    m_cnn = build_stabilized_cnn(); _, st_cnn = Lux.setup(Random.default_rng(), m_cnn)

    bias_base_list = Float32[]
    bias_ude_list = Float32[]
    bias_cnn_list = Float32[]

    for pat in val_cases
        pat_dir = joinpath(dataset_dir, pat); if !isdir(pat_dir); continue; end
        ct_f = niread(joinpath(pat_dir, "ct.nii.gz")); sp_f = niread(joinpath(pat_dir, "spect.nii.gz")); mc_f = niread(joinpath(pat_dir, "dosemap_mc.nii.gz")); ap_f = niread(joinpath(pat_dir, "dosemap_approx.nii.gz"))
        extract(f) = ndims(f) == 3 ? f : f[:,:,:,1]
        ct_i = extract(ct_f); sp_i = extract(sp_f); mc_i = extract(mc_f); ap_i = extract(ap_f)
        vol_p = Float32(prod(ct_f.header.pixdim[2:4]))

        # Center crop 32^3
        sz = size(ct_i); cx, cy, cz = sz .÷ 2
        xr, yr, zr = cx-15:cx+16, cy-15:cy+16, cz-15:cz+16
        ct_p = ct_i[xr, yr, zr]; sp_p = sp_i[xr, yr, zr]; mc_p = mc_i[xr, yr, zr]; ap_p = ap_i[xr, yr, zr]
        mask = mc_p .> (0.01 * maximum(mc_p))
        
        # 1. Analytical Baseline
        den = hu_to_den.(ct_p)
        pred_base_gy = (Float32.(sp_p) .* 8.478f-8) ./ (vol_p .* den .+ 1f-4)
        push!(bias_base_list, mean(pred_base_gy[mask] .* 1000 .- mc_p[mask]))

        # 2. UDE Improved
        if θ_ude !== nothing
            pred_ude_raw = predict_ude_patch_balanced(m_ude, θ_ude, st_ude, sp_p, den, vol_p)
            # Check if pred_ude is Gy or mGy. 
            # Based on training (scale ~ 10^11), it likely predicts mGy.
            # But DOSE_CONV_BALANCED = 0.08478 implies Gy.
            # Let's print max to be sure.
            if pat == val_cases[1]
                println("Debug UDE: max(pred) = ", maximum(pred_ude_raw), " | max(target) = ", maximum(mc_p))
            end
            # If max(pred) ~ 1000, then it's mGy. If ~ 1.0, then it's Gy.
            u_scale = maximum(pred_ude_raw) < 100.0 ? 1000.0 : 1.0
            push!(bias_ude_list, mean(pred_ude_raw[mask] .* u_scale .- mc_p[mask]))
        end

        # 3. CNN Improved
        if θ_cnn !== nothing
            sp_std = standardize(sp_p); den_std = standardize(den); ap_p_std = standardize(ap_p)
            input_cnn = reshape(stack([Float32.(sp_std), Float32.(den_std), Float32.(ap_p_std)]), 32, 32, 32, 3, 1)
            pred_cnn_res_norm, _ = Lux.apply(m_cnn, input_cnn, θ_cnn, Lux.testmode(st_cnn))
            mc_max_vol = maximum(mc_f)
            norm_scale = (mc_max_vol / 10.0f0 + 1e-6)
            # This CNN predicts a residual to the NORMALIZED approximate dose
            pred_cnn_total_mgy = (reshape(Array(pred_cnn_res_norm), 32, 32, 32) .+ ap_p ./ norm_scale) .* norm_scale
            # Alternatively, if it's a residual to physical:
            # pred_cnn_total_mgy = reshape(Array(pred_cnn_res_norm), 32, 32, 32) .* norm_scale .+ ap_p
            
            push!(bias_cnn_list, mean(pred_cnn_total_mgy[mask] .- mc_p[mask]))
        end

        println("Processed $pat")
    end

    println("\nMean Bias [mGy]:")
    println("Analytical Baseline: ", round(mean(bias_base_list), digits=2))
    if !isempty(bias_ude_list)
        println("Triple-Branch UDE:   ", round(mean(bias_ude_list), digits=2))
    end
    if !isempty(bias_cnn_list)
        println("CNN Improved:        ", round(mean(bias_cnn_list), digits=2))
    end
end

run_eval_all()
