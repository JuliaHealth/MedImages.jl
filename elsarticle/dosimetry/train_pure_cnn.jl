using Pkg
Pkg.activate("/home/user/MedImages.jl")

using Lux, LuxCUDA, CUDA, Optimisers, Zygote, Random, ComponentArrays, NIfTI, Serialization, Statistics

# Physical Constants
const DOSE_CONV = 8.478f-8 

function load_patient_data(patient_dir::String)
    ct_path = joinpath(patient_dir, "ct.nii.gz")
    spect_path = joinpath(patient_dir, "spect.nii.gz")
    dose_path = joinpath(patient_dir, "dosemap_mc.nii.gz")
    approx_path = joinpath(patient_dir, "dosemap_approx.nii.gz")
    if !isfile(ct_path) || !isfile(spect_path) || !isfile(dose_path); return nothing; end
    ct_f = niread(ct_path); spect_f = niread(spect_path); dose_f = niread(dose_path); approx_f = niread(approx_path)
    extract(arr) = ndims(arr) == 3 ? arr : arr[:,:,:,1]
    return ct_f, extract(ct_f), extract(spect_f), extract(dose_f), extract(approx_f)
end

function hu_to_density(hu::Real)
    hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) : 1.0f0 + 0.0007f0 * Float32(hu)
end

function ResBlock(channels::Int)
    return SkipConnection(Chain(Conv((3, 3, 3), channels => channels, pad=1, relu), Conv((3, 3, 3), channels => channels, pad=1)), +)
end

function build_pure_cnn(width::Int, depth::Int)
    layers = []
    # 2 channels input: Activity, Density
    push!(layers, Conv((3, 3, 3), 2 => width, pad=1, relu))
    for _ in 1:depth; push!(layers, ResBlock(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1, softplus))
    return Chain(layers...)
end

function gradient_loss(pred, target)
    gx_p = pred[2:end,:,:] .- pred[1:end-1,:,:]; gx_t = target[2:end,:,:] .- target[1:end-1,:,:]
    return sum(abs.(gx_p .- gx_t)) / length(gx_p)
end

function train_pure_cnn(dataset_dir::String, epochs::Int=15)
    println("Training Pure 3D CNN (Direct Dose Mapping)...")
    patients = sort(filter(isdir, readdir(dataset_dir, join=true)))
    rng = Random.default_rng(); Random.seed!(rng, 42)
    
    width, depth = 32, 3
    model = build_pure_cnn(width, depth)
    ps, st = Lux.setup(rng, model)
    dev = Lux.gpu_device()
    θ = dev(ComponentArray(ps))
    st = dev(st)
    opt_state = Optimisers.setup(Optimisers.Adam(1f-3), θ)

    for epoch in 1:epochs
        epoch_mae = 0.0f0; count = 0
        for pat in patients[1:min(15, length(patients))] # Training subset
            data = load_patient_data(pat); if data === nothing; continue; end
            ct_f, ct_i, spect_i, dose_mc, _ = data
            target = dose_mc ./ (maximum(dose_mc) / 10.0f0 + 1f-6) # Normalize target roughly to 0-10 range for CNN stability
            valid = findall(x -> x > 1f-3, target)
            if isempty(valid); continue; end
            
            for patch_idx in 1:2
                idx = rand(valid); cx, cy, cz = idx.I; p_size = 32; hx = p_size ÷ 2
                xr = clamp(cx-hx, 1, size(target,1)-p_size+1):clamp(cx-hx+p_size-1, p_size, size(target,1))
                yr = clamp(cy-hx, 1, size(target,2)-p_size+1):clamp(cy-hx+p_size-1, p_size, size(target,2))
                zr = clamp(cz-hx, 1, size(target,3)-p_size+1):clamp(cz-hx+p_size-1, p_size, size(target,3))
                
                # Inputs: [Activity, Density]
                act_p = Float32.(spect_i[xr,yr,zr]); den_p = hu_to_density.(ct_i[xr,yr,zr])
                input_p = dev(reshape(stack([act_p, den_p]), p_size, p_size, p_size, 2, 1))
                target_p = dev(Float32.(target[xr,yr,zr]))

                l, gs = Zygote.withgradient(t -> begin
                    pred, _ = Lux.apply(model, input_p, t, st)
                    pred_r = reshape(pred, p_size, p_size, p_size)
                    mae = sum(abs.(pred_r .- target_p)) / length(target_p)
                    grad = gradient_loss(pred_r, target_p)
                    0.7f0 * mae + 0.3f0 * grad
                end, θ)
                
                if !any(isnan.(gs[1]))
                    opt_state, θ = Optimisers.update(opt_state, θ, gs[1])
                    epoch_mae += l; count += 1
                end
                CUDA.reclaim(); GC.gc()
            end
        end
        println("  Epoch $epoch Loss: $(epoch_mae/count)")
    end
    serialize("elsarticle/dosimetry/model_pure_cnn.jls", θ)
end

train_pure_cnn("/home/user/MedImages.jl/elsarticle/dosimetry/data/", 15)
