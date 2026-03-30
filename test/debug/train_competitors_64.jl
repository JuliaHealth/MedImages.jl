using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, Lux, LuxCUDA, CUDA, Optimisers, Zygote, Random, ComponentArrays, NIfTI, Serialization, Statistics, SciMLSensitivity

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

function load_patient_data(patient_dir::String)
    ct_p = joinpath(patient_dir, "ct.nii.gz"); sp_p = joinpath(patient_dir, "spect.nii.gz"); ds_p = joinpath(patient_dir, "dosemap_mc.nii.gz"); ap_p = joinpath(patient_dir, "dosemap_approx.nii.gz")
    if !isfile(ct_p) || !isfile(sp_p) || !isfile(ds_p) || !isfile(ap_p); return nothing; end
    ct_f = niread(ct_p); sp_f = niread(sp_p); ds_f = niread(ds_p); ap_f = niread(ap_p)
    extract(arr) = ndims(arr) == 3 ? arr : arr[:,:,:,1]
    return ct_f, extract(ct_f), extract(sp_f), extract(ds_f), extract(ap_f)
end

function hu_to_density(hu::Real)
    hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) : 1.0f0 + 0.0007f0 * Float32(hu)
end

function standardize(x)
    μ = mean(x); σ = std(x) + 1f-6
    return (x .- μ) ./ σ
end

function ResBlockNorm(channels::Int)
    return SkipConnection(Chain(Conv((3, 3, 3), channels => channels, pad=1), GroupNorm(channels, 8, relu), Conv((3, 3, 3), channels => channels, pad=1), GroupNorm(channels, 8)), +)
end

function build_ude_no_approx(width::Int, depth::Int)
    layers = Any[Parallel(+, Conv((3, 3, 3), 1 => width, pad=1, relu), Conv((3, 3, 3), 1 => width, pad=1, relu))]
    for _ in 1:depth; push!(layers, ResBlockNorm(width)); end # Use Norm for UDE too if 64x64x64
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

function build_stabilized_cnn(width::Int, depth::Int)
    layers = Any[Conv((3, 3, 3), 3 => width, pad=1, relu)]
    for _ in 1:depth; push!(layers, ResBlockNorm(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1, init_weight=Lux.zeros32))
    return Chain(layers...)
end

function huber_loss(x, y, δ=0.1f0)
    diff = abs.(x .- y)
    return sum(ifelse.(diff .<= δ, 0.5f0 .* diff.^2, δ .* (diff .- 0.5f0 .* δ))) / length(loss)
end

function gradient_loss(pred, target)
    gx_p = pred[2:end,:,:] .- pred[1:end-1,:,:]; gx_t = target[2:end,:,:] .- target[1:end-1,:,:]
    return sum(abs.(gx_p .- gx_t)) / (length(gx_p) + 1e-6)
end

function train_model_64(mode::String, max_epochs::Int=500, patience::Int=50)
    println("\n>>> 64x64x64 TRAINING MODE: $mode")
    dataset_dir = "/home/user/MedImages.jl/elsarticle/dosimetry/data/"
    patients = sort(filter(isdir, readdir(dataset_dir, join=true)))
    rng_p = Random.Xoshiro(42); shuffle!(rng_p, patients)
    train_pats = patients[1:min(12, length(patients))]
    
    width, depth = 32, 3
    p_s = 64
    model = (mode == "CNN_APPROX") ? build_stabilized_cnn(width, depth) : build_ude_no_approx(width, depth)
    ps, st = Lux.setup(Random.default_rng(), model)
    dev = Lux.gpu_device(); θ = dev(ComponentArray(ps)); st = dev(st)
    opt_state = Optimisers.setup(Optimisers.Adam(1f-4), θ)
    best_loss = Inf32; epochs_without_impr = 0

    for epoch in 1:max_epochs
        epoch_loss = 0.0f0; count = 0
        for pat in train_pats
            data = load_patient_data(pat); if data === nothing; continue; end
            ct_f, ct_i, sp_i, ds_mc, ap_i = data
            t_min = minimum(ds_mc); t_max = maximum(ds_mc); target = (ds_mc .- t_min) ./ (t_max - t_min + 1f-6)
            valid = findall(x -> x > 0.1, target); if isempty(valid); continue; end
            
            for patch_idx in 1:1
                idx = rand(valid); cx, cy, cz = idx.I; hx = p_s ÷ 2
                xr = clamp(cx-hx, 1, size(target,1)-p_s+1):clamp(cx-hx+p_s-1, p_s, size(target,1))
                yr = clamp(cy-hx, 1, size(target,2)-p_s+1):clamp(cy-hx+p_s-1, p_s, size(target,2))
                zr = clamp(cz-hx, 1, size(target,3)-p_s+1):clamp(cz-hx+p_s-1, p_s, size(target,3))
                
                target_p = dev(Float32.(target[xr,yr,zr]))
                sp_std = dev(Float32.(standardize(sp_i[xr,yr,zr])))
                den_raw = hu_to_density.(ct_i[xr,yr,zr])
                den_std = dev(Float32.(standardize(den_raw)))
                app_raw = dev(Float32.(ap_i[xr,yr,zr]))
                app_std = dev(Float32.(standardize(ap_i[xr,yr,zr])))
                
                vol_p = Float32(prod(ct_f.header.pixdim[2:4]))
                t_max_f = Float32(t_max)
                den_raw_dev = dev(Float32.(den_raw))

                loss_val, gs = Zygote.withgradient(t -> begin
                    if mode == "CNN_APPROX"
                        input = reshape(stack([sp_std, den_std, app_std]), p_s, p_s, p_s, 3, 1)
                        nn_out, _ = Lux.apply(model, input, t, st)
                        pred_r = softplus.(reshape(nn_out, p_s, p_s, p_s) .+ app_raw ./ (t_max_f + 1f-6))
                    else
                        sp_p = dev(Float32.(sp_i[xr,yr,zr]))
                        u0 = ComponentArray(A_blood=dev(Float32[sum(sp_p)*0.05f0]), A_free=sp_p.*0.45f0, A_bound=sp_p.*0.50f0, DOSE=zero(sp_p))
                        function f(u,p,time)
                            A_t = u.A_free .+ u.A_bound
                            A_t_std = (A_t .- mean(A_t)) ./ (std(A_t) + 1f-6)
                            in_nn = (reshape(A_t_std, p_s, p_s, p_s, 1, 1), reshape(den_std, p_s, p_s, p_s, 1, 1))
                            nn_o, _ = Lux.apply(model, in_nn, p, st)
                            dD = softplus.( (A_t .* DOSE_CONV) ./ (vol_p .* den_raw_dev .+ 1f-4) .+ reshape(nn_o, p_s, p_s, p_s) )
                            return ComponentArray(A_blood=-(k10_pop+λ_phys)*u.A_blood, A_free=-(k_out_pop+λ_phys)*u.A_free, A_bound=-(k4+λ_phys)*u.A_bound, DOSE=dD)
                        end
                        prob = ODEProblem(f, u0, (0.0f0, 300.0f0), t)
                        # HEAVY CHECKPOINTING for 64x64x64
                        sol = solve(prob, VCABM(), saveat=[300.0f0], reltol=1f-3, abstol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true))
                        pred_r = (sol.u[end].DOSE .- minimum(sol.u[end].DOSE)) ./ (maximum(sol.u[end].DOSE) + 1f-6)
                    end
                    mae = sum(abs.(pred_r .- target_p)) / (p_s^3)
                    mae + 0.3f0 * gradient_loss(pred_r, target_p)
                end, θ)
                
                if !any(isnan.(gs[1]))
                    opt_state, θ = Optimisers.update(opt_state, θ, gs[1])
                    epoch_loss += loss_val; count += 1
                end
                CUDA.reclaim(); GC.gc()
            end
        end
        
        avg_l = epoch_loss/count
        if avg_l < best_loss; best_loss = avg_l; epochs_without_impr = 0; serialize("elsarticle/dosimetry/model_best_$(mode)_64.jls", θ) else epochs_without_impr += 1 end
        if epoch % 5 == 0 || epoch == 1; println("  Epoch $epoch | Loss: $avg_l | Patience: $epochs_without_impr/$patience") end
        if epochs_without_impr >= patience; println("  Early stopping at epoch $epoch"); break end
    end
end

train_model_64("CNN_APPROX", 500, 50)
train_model_64("UDE_NO_APPROX", 500, 50)
