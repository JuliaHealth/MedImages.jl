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
    push!(layers, Conv((3, 3, 3), width => 1, pad=1, init_weight=Lux.zeros32))
    return Chain(layers...)
end

function fix_parameters(θ)
    if θ isa ComponentArray; return NamedTuple(θ); else return θ; end
end

function predict_ude_balanced(model, θ, st, sp_raw, den_raw, vol_p)
    p_s = 64
    dev = Lux.gpu_device()
    # Unit Balancing: Activity / 1e6
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
    sol = solve(prob, Tsit5(), saveat=[300.0f0], reltol=1e-1, abstol=1e-1)
    return sol.u[end].DOSE
end

function evaluate_ude_balanced()
    dev = Lux.gpu_device(); rng = Random.default_rng()
    cp = "experiments/sciml_dose_refinement/data/checkpoints/UDE_IMPROVED_64/model_best.jls"
    if !isfile(cp); println("UDE_IMPROVED checkpoint not found."); return; end
    
    θ = fix_parameters(dev(deserialize(cp)))
    model = build_ude_improved(32, 3)
    _, st = Lux.setup(rng, model); st = dev(st)

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
    pearsons = Float64[]; maes = Float64[]

    println("\nEvaluating UDE_IMPROVED (BALANCED) on validation set...")
    for pat in val_cases
        pat_dir = joinpath(dataset_dir, pat); if !isdir(pat_dir); continue; end
        ct_f = niread(joinpath(pat_dir, "ct.nii.gz")); sp_f = niread(joinpath(pat_dir, "spect.nii.gz")); mc_f = niread(joinpath(pat_dir, "dosemap_mc.nii.gz"))
        
        extract(f) = ndims(f) == 3 ? f : f[:,:,:,1]
        ct_i = extract(ct_f); sp_i = extract(sp_f); mc_i = extract(mc_f)
        cx, cy, cz = size(ct_i) .÷ 2; xr, yr, zr = cx-31:cx+32, cy-31:cy+32, cz-31:cz+32
        
        target = mc_i[xr,yr,zr]; if maximum(target) < 1e-3; continue; end
        
        den_raw = hu_to_den.(ct_i[xr,yr,zr]); vol_p = Float32(prod(ct_f.header.pixdim[2:4]))
        sp_p = Float32.(sp_i[xr,yr,zr])

        pred = Array(predict_ude_balanced(model, θ, st, sp_p, den_raw, vol_p))

        if any(isnan, pred)
            println("  $pat | WARNING: NaNs in prediction")
            continue
        end

        c = cor(reshape(pred, :), reshape(target, :))
        m = mean(abs.(pred .- target))
        if isnan(c); c = 0.0; end
        push!(pearsons, c); push!(maes, m)
        println("  $pat | Pearson: $(round(c, digits=4)) | MAE (Gy): $(round(m, digits=4))")
    end
    println("\n--- Result for UDE_IMPROVED ---")
    println("Mean Pearson: $(round(mean(pearsons), digits=4))")
    println("Mean MAE (Gy): $(round(mean(maes), digits=4))")
end

evaluate_ude_balanced()
