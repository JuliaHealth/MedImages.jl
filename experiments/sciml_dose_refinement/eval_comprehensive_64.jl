using Pkg
Pkg.activate(@__DIR__)

using DifferentialEquations, Lux, LuxCUDA, CUDA, Random, ComponentArrays, NIfTI, Serialization, Statistics, Optimisers, SciMLSensitivity, Images, StatsBase

# Physical Constants
const λ_phys = Float32(log(2) / 159.5) 
const k10_pop = Float32(log(2) / 40.0) 
const f_pop = 1.0f0  
const k_in_pop = 0.01f0  
const k_out_pop = 0.02f0 
const k3 = 0.05f0 
const k4 = 0.01f0 
const DOSE_CONV = 8.478f-8 
const DOSE_CONV_BALANCED = 0.08478f0 

# --- Architectures ---

function ResBlockNorm(channels::Int)
    return SkipConnection(Chain(Conv((3, 3, 3), channels => channels, pad=1), GroupNorm(channels, 8, relu), Conv((3, 3, 3), channels => channels, pad=1), GroupNorm(channels, 8)), +)
end

function ResBlock(channels::Int)
    return SkipConnection(Chain(Conv((3, 3, 3), channels => channels, pad=1, relu), Conv((3, 3, 3), channels => channels, pad=1)), +)
end

function build_ude_improved()
    width, depth = 32, 3
    layers = Any[Parallel(+, Conv((3, 3, 3), 1 => width, pad=1, relu), Conv((3, 3, 3), 1 => width, pad=1, relu), Conv((3, 3, 3), 1 => width, pad=1, relu))]
    for _ in 1:depth; push!(layers, ResBlockNorm(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1, init_weight=Lux.zeros32))
    return Chain(layers...)
end

function build_ude_no_approx_64()
    width, depth = 32, 3
    layers = Any[Parallel(+, Conv((3, 3, 3), 1 => width, pad=1, relu), Conv((3, 3, 3), 1 => width, pad=1, relu))]
    for _ in 1:depth; push!(layers, ResBlockNorm(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

function build_cnn_improved_64()
    width, depth = 32, 3
    layers = Any[Conv((3, 3, 3), 3 => width, pad=1, relu)]
    for _ in 1:depth; push!(layers, ResBlock(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1, init_weight=Lux.zeros32))
    return Chain(layers...)
end

function build_cnn_approx_64()
    width, depth = 32, 3
    layers = Any[Conv((3, 3, 3), 3 => width, pad=1, relu)]
    for _ in 1:depth; push!(layers, ResBlockNorm(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1, init_weight=Lux.zeros32))
    return Chain(layers...)
end

function build_pure_cnn_64()
    width, depth = 32, 3
    layers = Any[Conv((3, 3, 3), 2 => width, pad=1, relu)]
    for _ in 1:depth; push!(layers, ResBlock(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1, softplus))
    return Chain(layers...)
end

# --- Helpers ---

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

function save_mip_png(data, path)
    mip = dropdims(maximum(data, dims=2), dims=2)
    mip = rotl90(mip)
    v_max = percentile(vec(mip), 99.5)
    if v_max == 0; v_max = 1.0; end
    mip_norm = clamp.(mip ./ v_max, 0, 1)
    save(path, colorview(Gray, Float64.(mip_norm)))
end

# --- Inference ---

function predict_ude_improved(model, θ, st, sp_raw, den_raw, vol_p)
    p_s = 64; dev = Lux.gpu_device(); st = Lux.testmode(st)
    sp_bal = sp_raw ./ 1e6; sp_p = dev(Float32.(sp_bal))
    u0 = ComponentArray(A_blood=dev(Float32[sum(sp_bal)*0.05f0]), A_free=sp_p.*0.45f0, A_bound=sp_p.*0.50f0, DOSE=zero(sp_p))
    den_std = dev(Float32.(standardize(den_raw))); den_grad_std = dev(Float32.(standardize(grad_3d(den_raw)))); den_raw_dev = dev(Float32.(den_raw))
    function f(u, p, t)
        A_t = u.A_free .+ u.A_bound; A_t_std = (A_t .- mean(A_t)) ./ (std(A_t) + 1f-6)
        in_nn = (reshape(A_t_std, p_s, p_s, p_s, 1, 1), reshape(den_std, p_s, p_s, p_s, 1, 1), reshape(den_grad_std, p_s, p_s, p_s, 1, 1))
        nn_o, _ = Lux.apply(model, in_nn, p, st)
        dD = softplus.( (A_t .* DOSE_CONV_BALANCED) ./ (vol_p .* den_raw_dev .+ 1f-4) .+ reshape(nn_o, p_s, p_s, p_s) )
        return ComponentArray(A_blood=-(k10_pop+λ_phys)*u.A_blood, A_free=-(k_out_pop+λ_phys)*u.A_free, A_bound=-(k4+λ_phys)*u.A_bound, DOSE=dD)
    end
    prob = ODEProblem(f, u0, (0.0f0, 300.0f0), θ)
    sol = solve(prob, Tsit5(), saveat=[300.0f0], reltol=1f-1, abstol=1f-1)
    return sol.u[end].DOSE
end

function predict_ude_noapp(model, θ, st, A0, den_p, vol_p)
    p_s = 64; dev = Lux.gpu_device(); st = Lux.testmode(st)
    u0 = ComponentArray(A_blood=dev(Float32[sum(A0)*0.05f0]), A_free=A0.*0.45f0, A_bound=A0.*0.50f0, DOSE=zero(A0))
    function ude_func(u, p, t)
        A_t = u.A_free .+ u.A_bound; A_t_std = (A_t .- mean(A_t)) ./ (std(A_t) + 1f-6)
        nn_out, _ = Lux.apply(model, (reshape(A_t_std, p_s, p_s, p_s, 1, 1), reshape(den_p, p_s, p_s, p_s, 1, 1)), p, st)
        dD = softplus.((A_t .* DOSE_CONV) ./ (vol_p .* den_p .+ 1f-4) .+ reshape(nn_out, p_s, p_s, p_s))
        return ComponentArray(A_blood=-(k10_pop+λ_phys)*u.A_blood, A_free=-(k_out_pop+λ_phys)*u.A_free, A_bound=-(k4+λ_phys)*u.A_bound, DOSE=dD)
    end
    prob = ODEProblem(ude_func, u0, (0.0f0, 300.0f0), θ)
    sol = solve(prob, Tsit5(), saveat=[300.0f0], reltol=1f-1, abstol=1f-1)
    return sol.u[end].DOSE
end

function run_comprehensive_eval_64()
    dev = Lux.gpu_device(); rng = Random.default_rng()
    
    # Paths
    cp_imp = "experiments/sciml_dose_refinement/data/checkpoints/UDE_IMPROVED_64/model_best.jls"
    cp_noapp = "data/checkpoints/UDE_NO_APPROX_64/model_best_UDE_NO_APPROX_64.jls"
    cp_cnn_imp = "experiments/sciml_dose_refinement/data/checkpoints/CNN_IMPROVED_64/model_best.jls"
    cp_cnn_app = "data/checkpoints/CNN_APPROX_64/model_best_CNN_APPROX_64.jls"
    cp_pure_cnn = "data/checkpoints/pure_cnn/model_pure_cnn.jls"

    # Load and setup
    println("Loading Models...")
    m_imp = build_ude_improved(); θ_imp = dev(ComponentArray(deserialize(cp_imp))); _, st_imp = Lux.setup(rng, m_imp); st_imp = dev(st_imp)
    m_noapp = build_ude_no_approx_64(); θ_noapp = dev(ComponentArray(deserialize(cp_noapp))); _, st_noapp = Lux.setup(rng, m_noapp); st_noapp = dev(st_noapp)
    m_cnn_imp = build_cnn_improved_64(); θ_cnn_imp = dev(ComponentArray(deserialize(cp_cnn_imp))); _, st_cnn_imp = Lux.setup(rng, m_cnn_imp); st_cnn_imp = dev(st_cnn_imp)
    m_cnn_app = build_cnn_approx_64(); θ_cnn_app = dev(ComponentArray(deserialize(cp_cnn_app))); _, st_cnn_app = Lux.setup(rng, m_cnn_app); st_cnn_app = dev(st_cnn_app)
    m_pure = build_pure_cnn_64(); θ_pure = dev(ComponentArray(deserialize(cp_pure_cnn))); _, st_pure = Lux.setup(rng, m_pure); st_pure = dev(st_pure)

    dataset_dir = "data/dosimetry_data"
    val_cases = [
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat48",
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat46",
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_2__Pat54",
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat54",
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_0__Pat44",
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_5__Pat47",
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat51"
    ]
    
    val_out_root = "/home/user/MedImages.jl/val_outputs"
    mkpath(val_out_root)

    println("\n" * "="^120)
    println(rpad("Patient", 35) * " | " * rpad("UDE Imp", 10) * " | " * rpad("UDE NoAp", 10) * " | " * rpad("CNN Imp", 10) * " | " * rpad("CNN+App", 10) * " | " * rpad("Pure CNN", 10) * " | " * rpad("Baseline", 10))
    println("-"^120)

    for pat in val_cases
        pat_dir = joinpath(dataset_dir, pat); if !isdir(pat_dir); continue; end
        ct_f = niread(joinpath(pat_dir, "ct.nii.gz")); sp_f = niread(joinpath(pat_dir, "spect.nii.gz")); mc_f = niread(joinpath(pat_dir, "dosemap_mc.nii.gz")); ap_f = niread(joinpath(pat_dir, "dosemap_approx.nii.gz"))
        extract(f) = ndims(f) == 3 ? f : f[:,:,:,1]
        ct_i = extract(ct_f); sp_i = extract(sp_f); mc_i = extract(mc_f); ap_i = extract(ap_f)
        cx, cy, cz = size(ct_i) .÷ 2; xr, yr, zr = cx-31:cx+32, cy-31:cy+32, cz-31:cz+32
        
        target = mc_i[xr,yr,zr]; hu_to_den(hu) = hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) : 1.0f0 + 0.0007f0 * Float32(hu)
        den_raw = hu_to_den.(ct_i[xr,yr,zr]); vol_p = Float32(prod(ct_f.header.pixdim[2:4])); sp_p = Float32.(sp_i[xr,yr,zr])
        
        pat_out_dir = joinpath(val_out_root, pat); mkpath(pat_out_dir)

        # 1. UDE Improved
        p_imp = Array(predict_ude_improved(m_imp, θ_imp, st_imp, sp_p, den_raw, vol_p))
        c_imp = cor(reshape(p_imp, :), reshape(target, :))
        save_mip_png(p_imp, joinpath(pat_out_dir, "ude_improved.png"))
        
        # 2. UDE No Approx
        p_noapp = Array(predict_ude_noapp(m_noapp, θ_noapp, st_noapp, dev(sp_p), dev(den_raw), vol_p))
        c_noapp = cor(reshape(p_noapp, :), reshape(target, :))
        save_mip_png(p_noapp, joinpath(pat_out_dir, "ude_no_approx.png"))

        # 3. CNN Improved (3-branch)
        sp_std = standardize(sp_p); den_std = standardize(den_raw); den_grad_std = standardize(grad_3d(den_raw))
        in_cnn_imp = dev(reshape(stack([Float32.(sp_std), Float32.(den_std), Float32.(den_grad_std)]), 64, 64, 64, 3, 1))
        p_cnn_imp, _ = Lux.apply(m_cnn_imp, in_cnn_imp, θ_cnn_imp, Lux.testmode(st_cnn_imp))
        p_cnn_imp_r = Array(reshape(p_cnn_imp, 64, 64, 64))
        c_cnn_imp = cor(reshape(p_cnn_imp_r, :), reshape(target, :))
        save_mip_png(p_cnn_imp_r, joinpath(pat_out_dir, "cnn_improved.png"))
        
        # 4. CNN + Approx
        app_std = standardize(ap_i[xr,yr,zr])
        in_cnn_app = dev(reshape(stack([Float32.(sp_std), Float32.(den_std), Float32.(app_std)]), 64, 64, 64, 3, 1))
        p_cnn_app, _ = Lux.apply(m_cnn_app, in_cnn_app, θ_cnn_app, Lux.testmode(st_cnn_app))
        p_cnn_app_phys = softplus.(Array(reshape(p_cnn_app, 64, 64, 64)) .+ ap_i[xr,yr,zr] ./ (maximum(mc_i)/10.0f0 + 1f-6))
        c_cnn_app = cor(reshape(p_cnn_app_phys, :), reshape(target, :))
        save_mip_png(p_cnn_app_phys, joinpath(pat_out_dir, "cnn_approx.png"))
        
        # 5. Pure CNN (Uses RAW inputs as per training script)
        in_pure = dev(reshape(stack([Float32.(sp_p), Float32.(den_raw)]), 64, 64, 64, 2, 1))
        p_pure, _ = Lux.apply(m_pure, in_pure, θ_pure, Lux.testmode(st_pure))
        p_pure_r = Array(reshape(p_pure, 64, 64, 64))
        c_pure = cor(reshape(p_pure_r, :), reshape(target, :))
        save_mip_png(p_pure_r, joinpath(pat_out_dir, "pure_cnn.png"))
        
        # 6. Baseline
        c_base = cor(reshape(ap_i[xr,yr,zr], :), reshape(target, :))
        save_mip_png(target, joinpath(pat_out_dir, "ground_truth.png"))

        # SAVE RAW DATA FOR PAT44
        if occursin("Pat44", pat)
            println(">>> Saving raw comparison data for Pat44 (.bin files)...")
            write(joinpath(pat_out_dir, "ct.bin"), Array(ct_i[xr,yr,zr]))
            write(joinpath(pat_out_dir, "mc.bin"), Array(target))
            write(joinpath(pat_out_dir, "baseline.bin"), Array(ap_i[xr,yr,zr]))
            write(joinpath(pat_out_dir, "ude_imp.bin"), p_imp)
            write(joinpath(pat_out_dir, "ude_noapp.bin"), p_noapp)
            write(joinpath(pat_out_dir, "cnn_imp.bin"), p_cnn_imp_r)
            write(joinpath(pat_out_dir, "cnn_app.bin"), p_cnn_app_phys)
            write(joinpath(pat_out_dir, "pure_cnn.bin"), p_pure_r)
        end

        println(rpad(pat[end-5:end], 35) * " | " * rpad(round(c_imp, digits=4), 10) * " | " * rpad(round(c_noapp, digits=4), 10) * " | " * rpad(round(c_cnn_imp, digits=4), 10) * " | " * rpad(round(c_cnn_app, digits=4), 10) * " | " * rpad(round(c_pure, digits=4), 10) * " | " * rpad(round(c_base, digits=4), 10))
    end
    println("="^120)
end

run_comprehensive_eval_64()
