using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, Lux, LuxCUDA, CUDA, Random, ComponentArrays, NIfTI, Serialization, Statistics, Optimisers, SciMLSensitivity
using ProgressMeter

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

function fix_parameters(θ)
    if θ isa ComponentArray; return NamedTuple(θ); else return θ; end
end

function predict_ude_patch_balanced(model, θ, st, sp_raw, den_raw, vol_p)
    p_s = size(sp_raw, 1)
    dev = Lux.gpu_device()
    sp_bal = sp_raw ./ 1e6
    sp_p = dev(Float32.(sp_bal))
    sum_sp = sum(sp_bal)
    u0 = ComponentArray(A_blood=dev(Float32[sum_sp*0.05f0]), A_free=sp_p.*0.45f0, A_bound=sp_p.*0.50f0, DOSE=zero(sp_p))
    den_std = dev(Float32.(standardize(den_raw)))
    den_grad_std = dev(Float32.(standardize(grad_3d(den_raw))))
    den_raw_dev = dev(Float32.(den_raw))

    function f(u, p, t)
        A_t = u.A_free .+ u.A_bound
        A_t_std = (A_t .- mean(A_t)) ./ (std(A_t) + 1f-6)
        in_nn = (reshape(A_t_std, p_s, p_s, p_s, 1, 1), reshape(den_std, p_s, p_s, p_s, 1, 1), reshape(den_grad_std, p_s, p_s, p_s, 1, 1))
        nn_o, _ = Lux.apply(model, in_nn, p, st)
        dD_phys = (A_t .* DOSE_CONV_BALANCED) ./ (vol_p .* den_raw_dev .+ 1f-4)
        dD = softplus.( dD_phys .+ reshape(nn_o, p_s, p_s, p_s) )
        return ComponentArray(A_blood=-(k10_pop+λ_phys)*u.A_blood, A_free=-(k_out_pop+λ_phys)*u.A_free, A_bound=-(k4+λ_phys)*u.A_bound, DOSE=dD)
    end
    prob = ODEProblem(f, u0, (0.0f0, 300.0f0), θ)
    sol = solve(prob, Euler(), dt=15.0f0, saveat=[300.0f0])
    return sol.u[end].DOSE
end

function get_3d_cosine_window(s)
    w = 1.0 .- cos.(range(0, 2π, length=s))
    w = w ./ maximum(w)
    win = w .* w'
    win3d = reshape(win, s, s, 1) .* reshape(w, 1, 1, s)
    return Float32.(win3d)
end

function sliding_window_inference(model, θ, st, spect, ct, vol_p; patch_size=64, stride=32)
    dev = Lux.gpu_device()
    nx, ny, nz = size(spect)
    final_dose = zeros(Float32, nx, ny, nz)
    counts = zeros(Float32, nx, ny, nz)
    window = get_3d_cosine_window(patch_size)
    window_dev = dev(window)

    x_centers = unique(vcat(collect(1:stride:nx-patch_size+1), nx-patch_size+1))
    y_centers = unique(vcat(collect(1:stride:ny-patch_size+1), ny-patch_size+1))
    z_centers = unique(vcat(collect(1:stride:nz-patch_size+1), nz-patch_size+1))
    
    total_patches = length(x_centers) * length(y_centers) * length(z_centers)
    p = Progress(total_patches, desc="Reconstructing Single Case...")

    for i in x_centers, j in y_centers, k in z_centers
        xr, yr, zr = i:i+patch_size-1, j:j+patch_size-1, k:k+patch_size-1
        sp_patch = Float32.(spect[xr, yr, zr])
        if sum(sp_patch) < 1e-2; next!(p); continue; end
        den_patch = hu_to_den.(ct[xr, yr, zr])
        pred = predict_ude_patch_balanced(model, θ, st, sp_patch, den_patch, vol_p)
        final_dose[xr, yr, zr] .+= Array(pred .* window_dev)
        counts[xr, yr, zr] .+= window
        next!(p)
    end
    return final_dose ./ (counts .+ 1e-6)
end

function run_single_case_inference()
    dev = Lux.gpu_device(); rng = Random.default_rng()
    cp = "experiments/sciml_dose_refinement/data/checkpoints/UDE_IMPROVED_64/model_best.jls"
    if !isfile(cp); println("Checkpoint not found."); return; end
    θ = fix_parameters(dev(deserialize(cp)))
    m = build_ude_improved(32, 3); _, st = Lux.setup(rng, m); st = dev(st)

    pat_name = "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat48"
    dataset_dir = "data/dosimetry_data"
    pat_dir = joinpath(dataset_dir, pat_name)
    
    ct_f = niread(joinpath(pat_dir, "ct.nii.gz"))
    sp_f = niread(joinpath(pat_dir, "spect.nii.gz"))
    mc_f = niread(joinpath(pat_dir, "dosemap_mc.nii.gz"))
    
    extract(f) = ndims(f) == 3 ? f : f[:,:,:,1]
    ct_i = extract(ct_f); sp_i = extract(sp_f); mc_i = extract(mc_f)
    vol_p = Float32(prod(ct_f.header.pixdim[2:4]))

    println("\n>>> Running Parallel Inference for: $pat_name")
    full_pred = sliding_window_inference(m, θ, st, sp_i, ct_i, vol_p)
    
    mask = mc_i .> (0.01 * maximum(mc_i))
    c = cor(reshape(full_pred[mask], :), reshape(mc_i[mask], :))
    m_abs = mean(abs.(full_pred[mask] .- mc_i[mask]))
    
    println("  Results for $pat_name:")
    println("  Pearson: $(round(c, digits=4)) | MAE: $(round(m_abs, digits=4))")
    
    out_dir = "val_outputs_parallel"; mkpath(out_dir)
    ni = NIVolume(full_pred); ni.header.pixdim = ct_f.header.pixdim
    niwrite(joinpath(out_dir, "$(pat_name)_parallel.nii.gz"), ni)
end

run_single_case_inference()
