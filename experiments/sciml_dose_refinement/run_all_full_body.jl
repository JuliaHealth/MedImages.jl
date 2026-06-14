using Pkg
Pkg.activate(@__DIR__)

using DifferentialEquations, Lux, LuxCUDA, CUDA, Random, ComponentArrays, NIfTI, Serialization, Statistics, Optimisers, SciMLSensitivity, ProgressMeter

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

function get_3d_cosine_window(s)
    w = 1.0 .- cos.(range(0, 2π, length=s))
    w = w ./ (maximum(w) + 1e-6)
    win = w .* w'
    win3d = reshape(win, s, s, 1) .* reshape(w, 1, 1, s)
    return Float32.(win3d)
end

function fix_parameters(θ)
    if θ isa ComponentArray; return NamedTuple(θ); else return θ; end
end

# --- Patch Predictors ---

function predict_ude_imp_patch(model, θ, st, sp_raw, den_raw, vol_p)
    p_s = size(sp_raw, 1); dev = Lux.gpu_device()
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
    sol = solve(prob, Euler(), dt=30.0f0, saveat=[300.0f0])
    return sol.u[end].DOSE
end

function predict_ude_noapp_patch(model, θ, st, sp_raw, den_raw, vol_p)
    p_s = size(sp_raw, 1); dev = Lux.gpu_device()
    sp_p = dev(Float32.(sp_raw))
    u0 = ComponentArray(A_blood=dev(Float32[sum(sp_raw)*0.05f0]), A_free=sp_p.*0.45f0, A_bound=sp_p.*0.50f0, DOSE=zero(sp_p))
    den_p = dev(Float32.(standardize(den_raw)))
    function f(u, p, t)
        A_t = u.A_free .+ u.A_bound; A_t_std = (A_t .- mean(A_t)) ./ (std(A_t) + 1f-6)
        nn_o, _ = Lux.apply(model, (reshape(A_t_std, p_s, p_s, p_s, 1, 1), reshape(den_p, p_s, p_s, p_s, 1, 1)), p, st)
        dD = softplus.((A_t .* DOSE_CONV) ./ (vol_p .* den_p .+ 1f-4) .+ reshape(nn_o, p_s, p_s, p_s))
        return ComponentArray(A_blood=-(k10_pop+λ_phys)*u.A_blood, A_free=-(k_out_pop+λ_phys)*u.A_free, A_bound=-(k4+λ_phys)*u.A_bound, DOSE=dD)
    end
    prob = ODEProblem(f, u0, (0.0f0, 300.0f0), θ)
    sol = solve(prob, Euler(), dt=30.0f0, saveat=[300.0f0])
    return sol.u[end].DOSE
end

# --- Main Sliding Window Loop ---

function run_comprehensive_full_body_inference()
    dev = Lux.gpu_device(); rng = Random.default_rng()
    
    # Checkpoints
    cp_imp = "experiments/sciml_dose_refinement/data/checkpoints/UDE_IMPROVED_64/model_best.jls"
    cp_noapp = "data/checkpoints/UDE_NO_APPROX_64/model_best_UDE_NO_APPROX_64.jls"
    cp_cnn_imp = "experiments/sciml_dose_refinement/data/checkpoints/CNN_IMPROVED_64/model_best.jls"
    cp_cnn_app = "data/checkpoints/CNN_APPROX_64/model_best_CNN_APPROX_64.jls"
    cp_pure_cnn = "data/checkpoints/pure_cnn/model_pure_cnn.jls"

    # Models & Params
    println("Loading Models...")
    m_imp = build_ude_improved(); θ_imp = fix_parameters(dev(deserialize(cp_imp))); _, st_imp = Lux.setup(rng, m_imp); st_imp = dev(st_imp)
    m_noapp = build_ude_no_approx_64(); θ_noapp = fix_parameters(dev(deserialize(cp_noapp))); _, st_noapp = Lux.setup(rng, m_noapp); st_noapp = dev(st_noapp)
    m_cnn_imp = build_cnn_improved_64(); θ_cnn_imp = fix_parameters(dev(deserialize(cp_cnn_imp))); _, st_cnn_imp = Lux.setup(rng, m_cnn_imp); st_cnn_imp = dev(st_cnn_imp)
    m_cnn_app = build_cnn_approx_64(); θ_cnn_app = fix_parameters(dev(deserialize(cp_cnn_app))); _, st_cnn_app = Lux.setup(rng, m_cnn_app); st_cnn_app = dev(st_cnn_app)
    m_pure = build_pure_cnn_64(); θ_pure = fix_parameters(dev(deserialize(cp_pure_cnn))); _, st_pure = Lux.setup(rng, m_pure); st_pure = dev(st_pure)

    val_cases = [
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat48",
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat46",
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_2__Pat54",
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat54",
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_0__Pat44",
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_5__Pat47",
        "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat51"
    ]

    for pat_name in val_cases
        out_dir = joinpath("val_outputs", pat_name)
        if isfile(joinpath(out_dir, "full_mc.bin"))
            println("Skipping $pat_name, already reconstructed.")
            continue
        end
        println("\n>>> Processing $pat_name")

        # Patient Data
        pat_dir = joinpath("data/dosimetry_data", pat_name)
        ct_f = niread(joinpath(pat_dir, "ct.nii.gz"))
        sp_f = niread(joinpath(pat_dir, "spect.nii.gz"))
        mc_f = niread(joinpath(pat_dir, "dosemap_mc.nii.gz"))
        ap_f = niread(joinpath(pat_dir, "dosemap_approx.nii.gz"))
        
        extract(f) = ndims(f) == 3 ? f : f[:,:,:,1]
        ct_i = extract(ct_f); sp_i = extract(sp_f); mc_i = extract(mc_f); ap_i = extract(ap_f)
        vol_p = Float32(prod(ct_f.header.pixdim[2:4]))
        
        nx, ny, nz = size(sp_i)
        stride = 48; p_s = 64
        x_centers = unique(vcat(collect(1:stride:nx-p_s+1), nx-p_s+1))
        y_centers = unique(vcat(collect(1:stride:ny-p_s+1), ny-p_s+1))
        z_centers = unique(vcat(collect(1:stride:nz-p_s+1), nz-p_s+1))
        
        total_patches = length(x_centers) * length(y_centers) * length(z_centers)
        window = get_3d_cosine_window(p_s)
        window_dev = dev(window)

        # Buffers for all variants
        res_imp = zeros(Float32, nx, ny, nz); res_noapp = zeros(Float32, nx, ny, nz)
        res_cnn_imp = zeros(Float32, nx, ny, nz); res_cnn_app = zeros(Float32, nx, ny, nz)
        res_pure = zeros(Float32, nx, ny, nz); counts = zeros(Float32, nx, ny, nz)

        prog = Progress(total_patches, desc="Reconstructing $pat_name...")

        for i in x_centers, j in y_centers, k in z_centers
            xr, yr, zr = i:i+p_s-1, j:j+p_s-1, k:k+p_s-1
            sp_patch = Float32.(sp_i[xr, yr, zr])
            if sum(sp_patch) < 1e-2; next!(prog); continue; end
            
            den_patch = hu_to_den.(ct_i[xr, yr, zr])
            
            # 1. UDE Imp
            p_imp = predict_ude_imp_patch(m_imp, θ_imp, st_imp, sp_patch, den_patch, vol_p)
            res_imp[xr,yr,zr] .+= Array(p_imp .* window_dev)
            
            # 2. UDE NoApp
            p_noapp = predict_ude_noapp_patch(m_noapp, θ_noapp, st_noapp, sp_patch, den_patch, vol_p)
            res_noapp[xr,yr,zr] .+= Array(p_noapp .* window_dev)
            
            # 3. CNN Imp
            sp_std = dev(Float32.(standardize(sp_patch))); den_std = dev(Float32.(standardize(den_patch))); grad_std = dev(Float32.(standardize(grad_3d(den_patch))))
            in_cnn_imp = dev(reshape(stack([Array(sp_std), Array(den_std), Array(grad_std)]), p_s, p_s, p_s, 3, 1))
            p_cnn_imp, _ = Lux.apply(m_cnn_imp, in_cnn_imp, θ_cnn_imp, Lux.testmode(st_cnn_imp))
            res_cnn_imp[xr,yr,zr] .+= Array(reshape(p_cnn_imp, p_s, p_s, p_s) .* window_dev)
            
            # 4. CNN App
            app_std = dev(Float32.(standardize(ap_i[xr,yr,zr])))
            in_cnn_app = dev(reshape(stack([Array(sp_std), Array(den_std), Array(app_std)]), p_s, p_s, p_s, 3, 1))
            p_cnn_app, _ = Lux.apply(m_cnn_app, in_cnn_app, θ_cnn_app, Lux.testmode(st_cnn_app))
            p_cnn_app_phys = softplus.(reshape(p_cnn_app, p_s, p_s, p_s) .+ dev(Float32.(ap_i[xr,yr,zr])) ./ (maximum(mc_i)/10.0f0 + 1f-6))
            res_cnn_app[xr,yr,zr] .+= Array(p_cnn_app_phys .* window_dev)
            
            # 5. Pure CNN
            in_pure = dev(reshape(stack([sp_patch, den_patch]), p_s, p_s, p_s, 2, 1))
            p_pure, _ = Lux.apply(m_pure, in_pure, θ_pure, Lux.testmode(st_pure))
            res_pure[xr,yr,zr] .+= Array(reshape(p_pure, p_s, p_s, p_s) .* window_dev)
            
            counts[xr,yr,zr] .+= window
            next!(prog)
            if i == x_centers[1] && j == y_centers[1] && k == z_centers[1]; GC.gc(); CUDA.reclaim(); end
        end

        # Normalize
        res_imp ./= (counts .+ 1e-6); res_noapp ./= (counts .+ 1e-6)
        res_cnn_imp ./= (counts .+ 1e-6); res_cnn_app ./= (counts .+ 1e-6)
        res_pure ./= (counts .+ 1e-6)
        
        # Save BIN files for fast plotting
        mkpath(out_dir)
        write(joinpath(out_dir, "full_ct.bin"), Array(ct_i))
        write(joinpath(out_dir, "full_mc.bin"), Array(mc_i))
        write(joinpath(out_dir, "full_baseline.bin"), Array(ap_i))
        write(joinpath(out_dir, "full_ude_imp.bin"), res_imp)
        write(joinpath(out_dir, "full_ude_noapp.bin"), res_noapp)
        write(joinpath(out_dir, "full_cnn_imp.bin"), res_cnn_imp)
        write(joinpath(out_dir, "full_cnn_app.bin"), res_cnn_app)
        write(joinpath(out_dir, "full_pure_cnn.bin"), res_pure)
        
        println("Full body results saved to $out_dir")
    end
end

run_comprehensive_full_body_inference()
