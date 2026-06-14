using Pkg
Pkg.activate("/home/user/MedImages.jl")

using Lux, LuxCUDA, CUDA, Random, ComponentArrays, NIfTI, Serialization, Statistics, OrdinaryDiffEq

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

function grad_3d_fix(vol)
    nx, ny, nz = size(vol)
    gx = zeros(Float32, nx, ny, nz)
    gy = zeros(Float32, nx, ny, nz)
    gz = zeros(Float32, nx, ny, nz)
    gx[2:nx-1, :, :] .= (vol[3:nx, :, :] .- vol[1:nx-2, :, :]) ./ 2.0f0
    gy[:, 2:ny-1, :] .= (vol[:, 3:ny, :] .- vol[:, 1:ny-2, :]) ./ 2.0f0
    gz[:, :, 2:nz-1] .= (vol[:, :, 3:nz] .- vol[:, :, 1:nz-2]) ./ 2.0f0
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

function build_cnn_improved(width::Int, depth::Int)
    layers = Any[Conv((3, 3, 3), 3 => width, pad=1, relu)]
    for _ in 1:depth; push!(layers, ResBlockNorm(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

function softplus(x)
    return log(1f0 + exp(x))
end

function solve_euler_manual(f, u0, tspan, p, dt)
    u = copy(u0)
    t = tspan[1]
    while t < tspan[2]
        du = f(u, p, t)
        u.A_blood .+= du.A_blood .* dt
        u.A_free .+= du.A_free .* dt
        u.A_bound .+= du.A_bound .* dt
        u.DOSE .+= du.DOSE .* dt
        t += dt
    end
    return u
end

function predict_ude_patch_balanced(model, θ, st, sp_raw, den_raw, vol_p)
    p_s = size(sp_raw, 1)
    dev = Lux.gpu_device()
    sp_bal = sp_raw ./ 1e6
    sp_p = dev(Float32.(sp_bal))
    sum_sp = sum(sp_bal)
    u0 = ComponentArray(A_blood=dev(Float32[sum_sp*0.05f0]), A_free=sp_p.*0.45f0, A_bound=sp_p.*0.50f0, DOSE=zero(sp_p))
    den_std = dev(Float32.(standardize(den_raw)))
    den_grad_std = dev(Float32.(standardize(grad_3d_fix(den_raw))))
    den_raw_dev = dev(Float32.(den_raw))

    function f(u, p, t)
        A_t = u.A_free .+ u.A_bound
        A_t_std = (A_t .- mean(A_t)) ./ (std(A_t) + 1f-6)
        in_nn = (reshape(A_t_std, p_s, p_s, p_s, 1, 1), reshape(den_std, p_s, p_s, p_s, 1, 1), reshape(den_grad_std, p_s, p_s, p_s, 1, 1))
        nn_o, _ = Lux.apply(model, in_nn, p, st)
        dD_phys = (A_t .* DOSE_CONV_BALANCED) ./ (vol_p .* den_raw_dev .+ 1f-4)
        dD = softplus.( dD_phys .+ reshape(nn_o, p_s, p_s, p_s) )
        return (A_blood=-(k10_pop+λ_phys)*u.A_blood, A_free=-(k_out_pop+λ_phys)*u.A_free, A_bound=-(k4+λ_phys)*u.A_bound, DOSE=dD)
    end
    
    u_final = solve_euler_manual(f, u0, (0.0f0, 300.0f0), θ, 15.0f0)
    return u_final.DOSE
end

function predict_cnn_improved(model, θ, st, sp_raw, den_raw)
    p_s = size(sp_raw, 1)
    dev = Lux.gpu_device()
    sp_std = dev(Float32.(standardize(sp_raw)))
    den_std = dev(Float32.(standardize(den_raw)))
    den_grad_std = dev(Float32.(standardize(grad_3d_fix(den_raw))))
    
    in_nn = dev(reshape(stack([sp_std, den_std, den_grad_std]), p_s, p_s, p_s, 3, 1))
    out, _ = Lux.apply(model, in_nn, θ, st)
    return out
end

function get_3d_cosine_window(s)
    w = 1.0 .- cos.(range(0, 2π, length=s))
    w = w ./ maximum(w)
    win = w .* w'
    win3d = reshape(win, s, s, 1) .* reshape(w, 1, 1, s)
    return Float32.(win3d)
end

function sliding_window_inference(model, θ, st, spect, ct, vol_p, m_type; patch_size=64, stride=32)
    dev = Lux.gpu_device()
    nx, ny, nz = size(spect)
    final_dose = zeros(Float32, nx, ny, nz)
    counts = zeros(Float32, nx, ny, nz)
    window = get_3d_cosine_window(patch_size)
    window_dev = dev(window)

    x_centers = unique(vcat(collect(1:stride:nx-patch_size+1), nx-patch_size+1))
    y_centers = unique(vcat(collect(1:stride:ny-patch_size+1), ny-patch_size+1))
    z_centers = unique(vcat(collect(1:stride:nz-patch_size+1), nz-patch_size+1))
    
    for i in x_centers, j in y_centers, k in z_centers
        xr, yr, zr = i:i+patch_size-1, j:j+patch_size-1, k:k+patch_size-1
        sp_patch = Float32.(spect[xr, yr, zr])
        if sum(sp_patch) < 1e-2; continue; end
        den_patch = hu_to_den.(ct[xr, yr, zr])
        
        if m_type == :ude
            pred = predict_ude_patch_balanced(model, θ, st, sp_patch, den_patch, vol_p)
            final_dose[xr, yr, zr] .+= Array(pred .* window_dev)
        elseif m_type == :cnn
            pred = predict_cnn_improved(model, θ, st, sp_patch, den_patch)
            final_dose[xr, yr, zr] .+= Array(reshape(pred, patch_size, patch_size, patch_size) .* window_dev)
        end
        counts[xr, yr, zr] .+= window
    end
    return final_dose ./ (counts .+ 1e-6)
end

function main()
    dev = Lux.gpu_device(); rng = Random.default_rng()
    
    # Models
    m_ude = build_ude_improved(32, 3)
    ps_ude, st_ude = Lux.setup(rng, m_ude)
    θ_ude = ComponentArray(ps_ude) |> dev; st_ude = st_ude |> dev

    m_cnn = build_cnn_improved(32, 3)
    ps_cnn, st_cnn = Lux.setup(rng, m_cnn)
    θ_cnn = ComponentArray(ps_cnn) |> dev; st_cnn = st_cnn |> dev

    val_cases = String[]
    open("experiments/sciml_dose_refinement/splits.txt", "r") do f
        in_val = false
        for line in eachline(f)
            if occursin("VALIDATION:", line); in_val = true; continue; end
            if occursin("TRAINING:", line); in_val = false; continue; end
            if in_val && strip(line) != ""; push!(val_cases, strip(line)); end
        end
    end
    
    # Sort for deterministic output
    sort!(val_cases)

    dataset_dir = "data/dosimetry_data"
    results = []

    println("Warmup...")
    dummy_sp = zeros(Float32, 64, 64, 64); dummy_sp[32,32,32] = 1.0f0
    dummy_ct = zeros(Float32, 64, 64, 64)
    predict_ude_patch_balanced(m_ude, θ_ude, st_ude, dummy_sp, dummy_ct, 1.0f0)
    predict_cnn_improved(m_cnn, θ_cnn, st_cnn, dummy_sp, dummy_ct)

    println("Starting Full-Set Benchmark (Julia Models)...")
    
    for (idx, pat) in enumerate(val_cases)
        pat_dir = joinpath(dataset_dir, pat)
        if !isdir(pat_dir); continue; end
        
        println("\n>>> Case $idx/$(length(val_cases)): $pat")
        
        ct_f = niread(joinpath(pat_dir, "ct.nii.gz"))
        sp_f = niread(joinpath(pat_dir, "spect.nii.gz"))
        extract(f) = ndims(f) == 3 ? f : f[:,:,:,1]
        ct_i = extract(ct_f); sp_i = extract(sp_f)
        vol_p = Float32(prod(ct_f.header.pixdim[2:4]))
        
        # 1. Analytical
        t1 = time()
        den = hu_to_den.(ct_i)
        _ = (Float32.(sp_i) .* 8.478f-8) ./ (vol_p .* den .+ 1f-4)
        t_static = time() - t1
        
        # 2. CNN Improved
        t1 = time()
        sliding_window_inference(m_cnn, θ_cnn, st_cnn, sp_i, ct_i, vol_p, :cnn)
        t_cnn = time() - t1
        
        # 3. UDE Improved
        t1 = time()
        sliding_window_inference(m_ude, θ_ude, st_ude, sp_i, ct_i, vol_p, :ude)
        t_ude = time() - t1
        
        push!(results, (pat, t_static, t_cnn, t_ude))
        @info "Results for $pat: Static=$(round(t_static, digits=2))s, CNN=$(round(t_cnn, digits=2))s, UDE=$(round(t_ude, digits=2))s"
        
        # Limit to 5 cases if it's too slow, but user asked for ALL.
        # Given 100s for UDE, 50 cases = 5000s = 1.4h. 
        # I will run it and the user can wait or I'll report progress.
    end

    println("\n" * "="^60)
    println("Summary Table (Seconds)")
    println("Patient | Static | CNN | UDE")
    for r in results
        println("$(r[1]) | $(round(r[2],2)) | $(round(r[3],2)) | $(round(r[4],2))")
    end
    
    m_static = mean([r[2] for r in results])
    m_cnn = mean([r[3] for r in results])
    m_ude = mean([r[4] for r in results])
    
    println("\nMEAN TIMES:")
    println("Analytical Baseline: $m_static s")
    println("CNN Improved:        $m_cnn s")
    println("UDE Improved:        $m_ude s")
end

main()
