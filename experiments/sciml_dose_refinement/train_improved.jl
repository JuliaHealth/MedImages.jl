using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, Lux, LuxCUDA, CUDA, Optimisers, Zygote, Random, ComponentArrays, NIfTI, Serialization, Statistics, SciMLSensitivity

const λ_phys = Float32(log(2) / 159.5) 
const k10_pop = Float32(log(2) / 40.0) 
const f_pop = 1.0f0  
const k_in_pop = 0.01f0  
const k_out_pop = 0.02f0 
const k3 = 0.05f0 
const k4 = 0.01f0 
const DOSE_CONV_BALANCED = 0.08478f0 

function load_patient_data(patient_dir::String)
    ct_p = joinpath(patient_dir, "ct.nii.gz"); sp_p = joinpath(patient_dir, "spect.nii.gz"); ds_p = joinpath(patient_dir, "dosemap_mc.nii.gz")
    if !isfile(ct_p) || !isfile(sp_p) || !isfile(ds_p); return nothing; end
    ct_f = niread(ct_p); sp_f = niread(sp_p); ds_f = niread(ds_p)
    extract(arr) = ndims(arr) == 3 ? arr : arr[:,:,:,1]
    return ct_f, extract(ct_f), extract(sp_f), extract(ds_f)
end

function hu_to_density(hu::Real)
    hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) : 1.0f0 + 0.0007f0 * Float32(hu)
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

function train_balanced_ude_resume(max_epochs::Int=500)
    println("\n>>> UDE_IMPROVED: RESUMING BALANCED UNIT TRAINING (LR 5e-5)")
    dataset_dir = "/home/user/MedImages.jl/data/dosimetry_data"
    patients = sort(filter(isdir, readdir(dataset_dir, join=true)))
    
    width, depth = 32, 3; p_s = 64
    model = build_ude_improved(width, depth)
    dev = Lux.gpu_device()
    
    cp = "experiments/sciml_dose_refinement/data/checkpoints/UDE_IMPROVED_64/model_best.jls"
    if isfile(cp)
        println("Loading existing checkpoint from $cp")
        θ = dev(ComponentArray(deserialize(cp)))
        # Recover best loss from log or just set a safe default
        best_loss = 2.09f0 
    else
        ps, _ = Lux.setup(Random.default_rng(), model)
        θ = dev(ComponentArray(ps))
        best_loss = Inf32
    end
    
    _, st = Lux.setup(Random.default_rng(), model)
    st = dev(st)
    
    # Lower LR for fine-tuning
    opt_state = Optimisers.setup(Optimisers.Adam(5f-5), θ)

    for epoch in 1:max_epochs
        epoch_loss = 0.0f0; count = 0
        # Shuffle patients each epoch
        rng_p = Random.Xoshiro(epoch + 42)
        shuffled_pats = shuffle(rng_p, patients)
        
        for pat_dir in shuffled_pats
            ct_p = joinpath(pat_dir, "ct.nii.gz"); sp_p = joinpath(pat_dir, "spect.nii.gz"); ds_p = joinpath(pat_dir, "dosemap_mc.nii.gz")
            if !isfile(ct_p) || !isfile(sp_p) || !isfile(ds_p); continue; end
            
            ct_f = niread(ct_p); sp_f = niread(sp_p); ds_f = niread(ds_p)
            extract(arr) = ndims(arr) == 3 ? arr : arr[:,:,:,1]
            ct_i = extract(ct_f); sp_i = extract(sp_f); ds_mc = extract(ds_f)
            
            sp_bal = sp_i ./ 1e6
            target_log = log1p.(ds_mc)
            valid = findall(x -> x > log1p(10.0), target_log); if isempty(valid); continue; end
            
            for patch_idx in 1:1
                idx = rand(valid); cx, cy, cz = idx.I; hx = p_s ÷ 2
                xr = clamp(cx-hx, 1, size(ds_mc,1)-p_s+1):clamp(cx-hx+p_s-1, p_s, size(ds_mc,1))
                yr = clamp(cy-hx, 1, size(ds_mc,2)-p_s+1):clamp(cy-hx+p_s-1, p_s, size(ds_mc,2))
                zr = clamp(cz-hx, 1, size(ds_mc,3)-p_s+1):clamp(cz-hx+p_s-1, p_s, size(ds_mc,3))
                
                target_p_log = dev(Float32.(target_log[xr,yr,zr]))
                sp_patch = dev(Float32.(sp_bal[xr,yr,zr]))
                den_raw = hu_to_density.(ct_i[xr,yr,zr])
                den_std = dev(Float32.(standardize(den_raw)))
                den_grad_std = dev(Float32.(standardize(grad_3d(den_raw))))
                
                den_raw_dev = dev(Float32.(den_raw)); vol_p = Float32(prod(ct_f.header.pixdim[2:4]))

                loss_val, gs = Zygote.withgradient(t -> begin
                    sum_sp = sum(sp_patch)
                    u0 = ComponentArray(A_blood=dev(Float32[sum_sp*0.05f0]), A_free=sp_patch.*0.45f0, A_bound=sp_patch.*0.50f0, DOSE=zero(sp_patch))
                    
                    function f(u,p,time)
                        A_t = u.A_free .+ u.A_bound
                        A_t_std = (A_t .- mean(A_t)) ./ (std(A_t) + 1f-6)
                        in_nn = (reshape(A_t_std, p_s, p_s, p_s, 1, 1), reshape(den_std, p_s, p_s, p_s, 1, 1), reshape(den_grad_std, p_s, p_s, p_s, 1, 1))
                        nn_o, _ = Lux.apply(model, in_nn, p, st)
                        dD_phys = (A_t .* DOSE_CONV_BALANCED) ./ (vol_p .* den_raw_dev .+ 1f-4)
                        dD = softplus.( dD_phys .+ reshape(nn_o, p_s, p_s, p_s) )
                        return ComponentArray(A_blood=-(k10_pop+λ_phys)*u.A_blood, A_free=-(k_out_pop+λ_phys)*u.A_free, A_bound=-(k4+λ_phys)*u.A_bound, DOSE=dD)
                    end
                    prob = ODEProblem(f, u0, (0.0f0, 300.0f0), t)
                    sol = solve(prob, Tsit5(), saveat=[300.0f0], reltol=1e-1, abstol=1e-1, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true))
                    
                    pred_log = log1p.(sol.u[end].DOSE)
                    sum(abs2, pred_log .- target_p_log) / (p_s^3)
                end, θ)
                
                if !any(isnan.(gs[1]))
                    g = gs[1]
                    gnorm = sqrt(sum(abs2, g))
                    if gnorm > 0.05f0; g = g .* (0.05f0 / gnorm); end # Even stricter clipping for fine-tuning
                    opt_state, θ = Optimisers.update(opt_state, θ, g)
                    epoch_loss += loss_val; count += 1
                end
                CUDA.reclaim(); GC.gc()
            end
        end
        avg_l = epoch_loss/count
        println("  Epoch $(epoch+47) | Resumed Balanced Log-MSE: $avg_l")
        if avg_l < best_loss && !isnan(avg_l)
            best_loss = avg_l
            serialize("experiments/sciml_dose_refinement/data/checkpoints/UDE_IMPROVED_64/model_best.jls", θ)
        end
    end
end

train_balanced_ude_resume(453)
