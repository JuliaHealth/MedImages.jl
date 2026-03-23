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

function build_ude_no_approx()
    width, depth = 32, 3
    branch_A = Conv((3, 3, 3), 1 => width, pad=1, relu); branch_ρ = Conv((3, 3, 3), 1 => width, pad=1, relu)
    layers = Any[Parallel(+, branch_A, branch_ρ)]
    for _ in 1:depth; push!(layers, ResBlock(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

# --- Metrics (on flattened data) ---

function pearson_corr(x, y)
    return cor(reshape(x, :), reshape(y, :))
end

function r_squared(y_true, y_pred)
    yt = reshape(y_true, :); yp = reshape(y_pred, :)
    ss_res = sum((yt .- yp).^2)
    ss_tot = sum((yt .- mean(yt)).^2)
    return 1.0f0 - ss_res / (ss_tot + 1f-6)
end

function simple_ssim(x, y)
    xt = reshape(x, :); yt = reshape(y, :)
    μx = mean(xt); μy = mean(yt)
    σx2 = var(xt); σy2 = var(yt)
    σxy = cov(xt, yt)
    L = 10.0f0; k1 = 0.01f0; k2 = 0.03f0
    c1 = (k1 * L)^2; c2 = (k2 * L)^2
    return ((2μx*μy + c1) * (2σxy + c2)) / ((μx^2 + μy^2 + c1) * (σx2 + σy2 + c2))
end

# --- Inference ---

function predict_ude(model, θ, st, p_state, A0)
    mass_map = p_state.vol_map .* p_state.ρ_map; ρ_map = p_state.ρ_map; dev = Lux.gpu_device()
    u0 = ComponentArray(A_blood=dev(Float32[sum(A0)*0.05f0]), A_free=A0.*0.45f0, A_bound=A0.*0.50f0, DOSE=zero(A0))
    function ude_func(u, p, t)
        total_A0 = sum(A0) + 1f-6; voxel_in = (f_pop * k_in_pop * u.A_blood) .* (A0 ./ total_A0)
        dA_blood = - (k10_pop + λ_phys) * u.A_blood .- sum(f_pop * k_in_pop * u.A_blood) .+ sum(k_out_pop .* u.A_free)
        dA_free  = voxel_in .- (k_out_pop .* u.A_free) .- (λ_phys .* u.A_free)
        dA_bound = (k3 .* u.A_free .* (1.0f0 .- u.A_bound ./ B_MAX_val)) .- (k4 .* u.A_bound) .- (λ_phys .* u.A_bound)
        A_t = u.A_free .+ u.A_bound
        inputs = (reshape(A_t, size(A0)..., 1, 1), reshape(ρ_map, size(A0)..., 1, 1))
        nn_out, _ = Lux.apply(model, inputs, p, st)
        dD_phys = softplus.((A_t .* DOSE_CONV) ./ (mass_map .+ 1f-4) .+ reshape(nn_out, size(A0)))
        return ComponentArray(A_blood=dA_blood, A_free=dA_free, A_bound=dA_bound, DOSE=ifelse.(ρ_map .< 0.1f0, 0.0f0, dD_phys))
    end
    prob = ODEProblem(ude_func, u0, (0.0f0, 300.0f0), θ)
    sol = solve(prob, VCABM(), saveat=[300.0f0], reltol=1f-3, abstol=1f-3)
    return sol.u[end].DOSE
end

function run_comprehensive_eval()
    dev = Lux.gpu_device(); rng = Random.default_rng()
    m_noapp = build_ude_no_approx(); θ_noapp = dev(deserialize("elsarticle/dosimetry/model_best_UDE_NO_APPROX.jls")); _, st_noapp = Lux.setup(rng, m_noapp); st_noapp = dev(st_noapp)

    dataset_dir = "elsarticle/dosimetry/data"; patients = sort(filter(isdir, readdir(dataset_dir, join=true)))
    rng_split = Random.Xoshiro(42); shuffle!(rng_split, patients)
    val_patients = patients[Int(round(0.8 * length(patients)))+1:end]

    println("Comprehensive Evaluation Table (Model vs Base)")
    println("-"^160)
    println(rpad("Patient", 20), " | ", rpad("M:Corr", 8), " | ", rpad("M:MAE", 8), " | ", rpad("M:R2", 8), " | ", rpad("M:SSIM", 8), " | ", rpad("B:Corr", 8), " | ", rpad("B:MAE", 8), " | ", rpad("B:R2", 8), " | ", rpad("B:SSIM", 8))
    println("-"^160)

    for pat_dir in val_patients
        name = basename(pat_dir)
        ct_f = niread(joinpath(pat_dir, "ct.nii.gz")); spect_f = niread(joinpath(pat_dir, "spect.nii.gz")); dose_mc = niread(joinpath(pat_dir, "dosemap_mc.nii.gz")); approx_f = niread(joinpath(pat_dir, "dosemap_approx.nii.gz"))
        ct_i = ndims(ct_f) == 3 ? ct_f : ct_f[:,:,:,1]; spect_i = ndims(spect_f) == 3 ? spect_f : spect_f[:,:,:,1]; mc_i = ndims(dose_mc) == 3 ? dose_mc : dose_mc[:,:,:,1]; app_i = ndims(approx_f) == 3 ? approx_f : approx_f[:,:,:,1]
        
        scale = maximum(mc_i) / 10.0f0; target = mc_i ./ (scale + 1f-6)
        base_unnorm = Float32.(app_i); base = base_unnorm ./ (maximum(base_unnorm) / 10.0f0 + 1f-6)
        
        cx, cy, cz = size(ct_i) .÷ 2; xr, yr, zr = cx-15:cx+16, cy-15:cy+16, cz-15:cz+16
        hu_to_den(hu) = hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) : 1.0f0 + 0.0007f0 * Float32(hu)
        den_p = hu_to_den.(ct_i[xr,yr,zr]); vol_p = Float32(prod(ct_f.header.pixdim[2:4])); A0 = dev(Float32.(spect_i[xr,yr,zr])); T = Float32.(target[xr,yr,zr])
        
        # No-Approx UDE
        p_state_noapp = (vol_map=dev(fill(vol_p, (32,32,32))), ρ_map=dev(den_p))
        pred_raw = Array(predict_ude(m_noapp, θ_noapp, st_noapp, p_state_noapp, A0))
        pred = pred_raw .* (mean(T) / (mean(pred_raw) + 1f-6))
        
        # Baseline
        B = Float32.(base[xr,yr,zr]); B_scaled = B .* (mean(T) / (mean(B) + 1f-6))
        
        # Model Metrics
        c_m = pearson_corr(pred, T); mae_m = mean(abs.(pred .- T)); r2_m = r_squared(T, pred); ssim_m = simple_ssim(pred, T)
        # Base Metrics
        c_b = pearson_corr(B_scaled, T); mae_b = mean(abs.(B_scaled .- T)); r2_b = r_squared(T, B_scaled); ssim_b = simple_ssim(B_scaled, T)

        println(rpad(name[end-15:end], 20), " | ", 
                rpad(round(c_m, digits=3), 8), " | ", rpad(round(mae_m, digits=3), 8), " | ", rpad(round(r2_m, digits=3), 8), " | ", rpad(round(ssim_m, digits=3), 8), " | ",
                rpad(round(c_b, digits=3), 8), " | ", rpad(round(mae_b, digits=3), 8), " | ", rpad(round(r2_b, digits=3), 8), " | ", rpad(round(ssim_b, digits=3), 8))
        CUDA.reclaim(); GC.gc()
    end
end
run_comprehensive_eval()
