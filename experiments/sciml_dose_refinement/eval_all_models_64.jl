using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, Lux, LuxCUDA, CUDA, Random, ComponentArrays, NIfTI, Serialization, Statistics, Optimisers, SciMLSensitivity

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

# --- Architectures ---

function ResBlock(channels::Int)
    return SkipConnection(Chain(Conv((3, 3, 3), channels => channels, pad=1, relu), Conv((3, 3, 3), channels => channels, pad=1)), +)
end

function build_ude_no_approx_64()
    width, depth = 32, 3
    branch_A = Conv((3, 3, 3), 1 => width, pad=1, relu); branch_ρ = Conv((3, 3, 3), 1 => width, pad=1, relu)
    layers = Any[Parallel(+, branch_A, branch_ρ)]
    for _ in 1:depth; push!(layers, ResBlock(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

function build_cnn_approx_64()
    width, depth = 32, 3
    layers = Any[Conv((3, 3, 3), 3 => width, pad=1, relu)]
    for _ in 1:depth; push!(layers, ResBlock(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

# --- Parameter Fix Helper ---

function fix_parameters(θ)
    if θ isa ComponentArray
        return NamedTuple(θ)
    else
        return θ
    end
end

# --- Inference ---

function predict_ude(model, θ, st, A0, den_p, vol_p)
    p_s = size(A0, 1)
    dev = Lux.cpu_device()
    u0 = ComponentArray(A_blood=dev(Float32[sum(A0)*0.05f0]), A_free=A0.*0.45f0, A_bound=A0.*0.50f0, DOSE=zero(A0))
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

function run_single_eval_64()
    dev = Lux.cpu_device(); rng = Random.default_rng()
    
    # Load and Fix
    θ_noapp_raw = dev(deserialize("data/checkpoints/UDE_NO_APPROX_64/model_best_UDE_NO_APPROX_64.jls"))
    θ_cnn_raw = dev(deserialize("data/checkpoints/CNN_APPROX_64/model_best_CNN_APPROX_64.jls"))
    
    θ_noapp = fix_parameters(θ_noapp_raw)
    θ_cnn = fix_parameters(θ_cnn_raw)

    m_noapp = build_ude_no_approx_64(); _, st_noapp = Lux.setup(rng, m_noapp); st_noapp = dev(st_noapp)
    m_cnn = build_cnn_approx_64(); _, st_cnn = Lux.setup(rng, m_cnn); st_cnn = dev(st_cnn)

    # Use first available Pat47 for evaluation
    dataset_dir = "data/dosimetry_data"
    candidates = filter(p -> occursin("Pat47", p), readdir(dataset_dir))
    if isempty(candidates)
        println("Error: No Pat47 found in $dataset_dir")
        return
    end
    pat_dir = joinpath(dataset_dir, candidates[1])
    out_dir = "data/vis_results_64"; mkpath(out_dir)

    println("Evaluating 64x64x64 models on $(basename(pat_dir))...")

    ct_f = niread(joinpath(pat_dir, "ct.nii.gz"))
    sp_f = niread(joinpath(pat_dir, "spect.nii.gz"))
    mc_f = niread(joinpath(pat_dir, "dosemap_mc.nii.gz"))
    ap_f = niread(joinpath(pat_dir, "dosemap_approx.nii.gz"))

    ct_i = ndims(ct_f) == 3 ? ct_f : ct_f[:,:,:,1]; sp_i = ndims(sp_f) == 3 ? sp_f : sp_f[:,:,:,1]; mc_i = ndims(mc_f) == 3 ? mc_f : mc_f[:,:,:,1]; ap_i = ndims(ap_f) == 3 ? ap_f : ap_f[:,:,:,1]
    
    # Center crop 64^3
    cx, cy, cz = size(ct_i) .÷ 2
    xr, yr, zr = cx-31:cx+32, cy-31:cy+32, cz-31:cz+32
    
    hu_to_den(hu) = hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) : 1.0f0 + 0.0007f0 * Float32(hu)
    den_p = hu_to_den.(ct_i[xr,yr,zr]); vol_p = Float32(prod(ct_f.header.pixdim[2:4])); A0 = dev(Float32.(sp_i[xr,yr,zr]))

    # Predict UDE
    ude_p = predict_ude(m_noapp, θ_noapp, st_noapp, A0, dev(den_p), vol_p)
    c_ude = cor(reshape(Array(ude_p), :), reshape(mc_i[xr,yr,zr], :))
    println("UDE No-Approx (64^3) Pearson: ", round(c_ude, digits=4))

    # Predict CNN
    sp_std = (sp_i[xr,yr,zr] .- mean(sp_i[xr,yr,zr])) ./ (std(sp_i[xr,yr,zr]) + 1f-6)
    den_std = (den_p .- mean(den_p)) ./ (std(den_p) + 1f-6)
    app_std = (ap_i[xr,yr,zr] .- mean(ap_i[xr,yr,zr])) ./ (std(ap_i[xr,yr,zr]) + 1f-6)
    input_cnn = dev(reshape(stack([Float32.(sp_std), Float32.(den_std), Float32.(app_std)]), 64, 64, 64, 3, 1))
    
    pred_cnn, _ = Lux.apply(m_cnn, input_cnn, θ_cnn, Lux.testmode(st_cnn))
    t_max = maximum(mc_i)
    pred_c_p = softplus.(Array(reshape(pred_cnn, 64, 64, 64)) .+ ap_i[xr,yr,zr] ./ (t_max/10.0f0 + 1f-6))
    c_cnn = cor(reshape(pred_c_p, :), reshape(mc_i[xr,yr,zr], :))
    println("CNN+Approx (64^3) Pearson:    ", round(c_cnn, digits=4))

    # Save results
    pat_out = joinpath(out_dir, basename(pat_dir)); mkpath(pat_out)
    write(joinpath(pat_out, "ude.bin"), Array(ude_p))
    write(joinpath(pat_out, "cnn.bin"), pred_c_p)
    write(joinpath(pat_out, "orig.bin"), Array(mc_i[xr,yr,zr]))
    write(joinpath(pat_out, "approx.bin"), Array(ap_i[xr,yr,zr]))
    println("Results saved to $pat_out")
end
run_single_eval_64()
