using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, Lux, LuxCUDA, CUDA, Random, ComponentArrays, NIfTI, Serialization, Statistics, Optimisers, SciMLSensitivity
using ProgressMeter
using PyCall
using NNlib

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

function build_ude_no_approx_32()
    width, depth = 32, 3
    layers = Any[Parallel(+, Conv((3, 3, 3), 1 => width, pad=1, relu), Conv((3, 3, 3), 1 => width, pad=1, relu))]
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

function standardize(x)
    μ = mean(x); σ = std(x) + 1f-6
    return (x .- μ) ./ σ
end

# --- Optimized Inference ---

function predict_ude_patch(model, θ, st, A0, den_raw, vol_p, patch_size, model_type)
    dev = Lux.gpu_device()
    CUDA.allowscalar(true)
    u0 = ComponentArray(A_blood=dev(Float32[sum(A0)*0.05f0]), A_free=A0.*0.45f0, A_bound=A0.*0.50f0, DOSE=zero(A0))
    CUDA.allowscalar(false)
    
    den_raw_dev = dev(Float32.(den_raw))
    den_std = model_type == "UDE_NOAPP_64" ? dev(Float32.(standardize(den_raw))) : den_raw_dev

    function ude_func_fast(u, p, t)
        A_t = u.A_free .+ u.A_bound
        local in_nn
        if model_type == "UDE_NOAPP_64"
            A_t_std = (A_t .- mean(A_t)) ./ (std(A_t) + 1f-6)
            in_nn = (reshape(A_t_std, patch_size, patch_size, patch_size, 1, 1), reshape(den_std, patch_size, patch_size, patch_size, 1, 1))
        else
            in_nn = (reshape(A_t, patch_size, patch_size, patch_size, 1, 1), reshape(den_raw_dev, patch_size, patch_size, patch_size, 1, 1))
        end
        nn_o, _ = Lux.apply(model, in_nn, p, st)
        dD_base = (A_t .* DOSE_CONV) ./ (vol_p .* den_raw_dev .+ 1f-4)
        dD_phys = softplus.(dD_base .+ reshape(nn_o, patch_size, patch_size, patch_size))
        dA_blood = -(k10_pop + λ_phys) * u.A_blood
        dA_free = -(k_out_pop + λ_phys) * u.A_free
        dA_bound = -(k4 + λ_phys) * u.A_bound
        return ComponentArray(A_blood=dA_blood, A_free=dA_free, A_bound=dA_bound, DOSE=dD_phys)
    end
    
    prob = ODEProblem(ude_func_fast, u0, (0.0f0, 300.0f0), θ)
    sol = solve(prob, Euler(), dt=15.0f0, saveat=[300.0f0])
    return sol.u[end].DOSE
end

function sliding_window_inference(model, θ, st, spect, ct, vol_p, model_type; patch_size=32, stride=16)
    dev = Lux.gpu_device()
    sz = size(spect)
    output = zeros(Float32, sz)
    counts = zeros(Float32, sz)
    
    hu_to_den(hu) = hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) : 1.0f0 + 0.0007f0 * Float32(hu)
    den_full = hu_to_den.(ct)
    baseline_full = (spect .* DOSE_CONV) ./ (vol_p .* den_full .+ 1f-4)

    r1 = [1:stride:sz[1]-patch_size+1; sz[1]-patch_size+1] |> unique |> sort
    r2 = [1:stride:sz[2]-patch_size+1; sz[2]-patch_size+1] |> unique |> sort
    r3 = [1:stride:sz[3]-patch_size+1; sz[3]-patch_size+1] |> unique |> sort
    
    total_patches = length(r1) * length(r2) * length(r3)
    p = Progress(total_patches, dt=1.0, desc="Inference ($model_type)...")

    x = range(-π/2, π/2, length=patch_size)
    w1d = cos.(x)
    window = Float32.(reshape(w1d, :, 1, 1) .* reshape(w1d, 1, :, 1) .* reshape(w1d, 1, 1, :))

    for i in r1, j in r2, k in r3
        xr, yr, zr = i:i+patch_size-1, j:j+patch_size-1, k:k+patch_size-1
        if maximum(spect[xr, yr, zr]) < 1e-3
            output[xr, yr, zr] .+= baseline_full[xr, yr, zr] .* window
            counts[xr, yr, zr] .+= window
            next!(p)
            continue
        end
        A0_p = dev(Float32.(spect[xr, yr, zr]))
        local pred_patch
        try
            pred_patch = Array(predict_ude_patch(model, θ, st, A0_p, den_full[xr, yr, zr], vol_p, patch_size, model_type))
            output[xr, yr, zr] .+= pred_patch .* window
            counts[xr, yr, zr] .+= window
        catch e
            @warn "Error in patch [$i, $j, $k]: $e"
            output[xr, yr, zr] .+= baseline_full[xr, yr, zr] .* window
            counts[xr, yr, zr] .+= window
        end
        next!(p)
        if (i + j + k) % 20 == 0; CUDA.reclaim(); GC.gc(); end
    end
    final_res = output ./ (counts .+ 1f-8)
    v = var(final_res)
    if v < 0.001; @warn "Warning: Output variance too low ($v)."; end
    return final_res
end

function get_body_mask(ct_path, pat_name)
    mask_dir = "data/body_masks/$pat_name"
    mask_file = joinpath(mask_dir, "body.nii.gz")
    if isfile(mask_file); return Float32.(niread(mask_file)); end
    mkpath(mask_dir)
    println("  Generating body mask with TotalSegmentator...")
    ts_bin = "/home/user/miniforge/envs/merlin_anatomical/bin/TotalSegmentator"
    # Using the 'body' task for artifact removal
    cmd = `$ts_bin -i $ct_path -o $mask_dir -ta body --fast`
    try
        run(cmd)
    catch e
        @error "TotalSegmentator failed for $pat_name: $e"
        return nothing
    end
    if isfile(mask_file)
        return Float32.(niread(mask_file))
    else
        @error "Body mask file not found after TotalSegmentator run for $pat_name"
        return nothing
    end
end

function run_ude_only_val()
    dev = Lux.gpu_device(); rng = Random.default_rng()
    models = Dict()
    cp64 = "data/checkpoints/UDE_NO_APPROX_64/model_best_UDE_NO_APPROX_64.jls"
    if isfile(cp64)
        m64 = build_ude_no_approx_64()
        models["UDE_NOAPP_64"] = (m64, fix_parameters(dev(deserialize(cp64))), dev(Lux.setup(rng, m64)[2]))
        println("Loaded 64x64 model.")
    else
        @error "64x64 checkpoint NOT found at $cp64"; return
    end

    val_cases = ["FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat48"]

    out_root = "val_outputs"; mkpath(out_root)
    metrics_file = open(joinpath(out_root, "metrics_ude.txt"), "a")

    for pat_name in val_cases
        pat_dir = joinpath("data/dosimetry_data", pat_name)
        if !isdir(pat_dir); continue; end
        println("\n>>> Processing Patient: $pat_name")
        ct_path = joinpath(pat_dir, "ct.nii.gz")
        ct_f = niread(ct_path); sp_f = niread(joinpath(pat_dir, "spect.nii.gz"))
        mc_f = niread(joinpath(pat_dir, "dosemap_mc.nii.gz"))
        ct = Float32.(ct_f); spect = Float32.(sp_f); gt = Float32.(mc_f)
        vol_p = Float32(prod(ct_f.header.pixdim[2:4])); pat_out = joinpath(out_root, pat_name); mkpath(pat_out)

        body_mask = get_body_mask(ct_path, pat_name)
        if body_mask === nothing; continue; end

        for m_name in sort(collect(keys(models)))
            out_path = joinpath(pat_out, "$(lowercase(m_name)).nii.gz")
            if isfile(out_path); continue; end
            m, θ, st = models[m_name]
            p_size = 64; stride = p_size ÷ 2
            res = sliding_window_inference(m, θ, st, spect, ct, vol_p, m_name; patch_size=p_size, stride=stride)
            res = res .* body_mask
            niwrite(out_path, NIVolume(ct_f.header, ct_f.extensions, res))
            r = cor(reshape(res, :), reshape(gt, :))
            println("  $m_name Correlation: $r")
            println(metrics_file, "$pat_name | $m_name | $r")
            flush(metrics_file)
        end
        # Copy comparisons
        src_mc = joinpath(pat_dir, "dosemap_mc.nii.gz")
        dst_mc = joinpath(pat_out, "monte_carlo.nii.gz")
        if isfile(src_mc) && realpath(abspath(src_mc)) != (isfile(dst_mc) ? realpath(abspath(dst_mc)) : "")
            cp(src_mc, dst_mc, force=true)
        end

        src_ap = joinpath(pat_dir, "dosemap_approx.nii.gz")
        dst_ap = joinpath(pat_out, "analytical_baseline.nii.gz")
        if isfile(src_ap) && realpath(abspath(src_ap)) != (isfile(dst_ap) ? realpath(abspath(dst_ap)) : "")
            cp(src_ap, dst_ap, force=true)
        end
    end
    close(metrics_file)
end

run_ude_only_val()
