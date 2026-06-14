using Pkg
Pkg.activate("experiments/sciml_dose_refinement")
using Lux, LuxCUDA, CUDA, Random, ComponentArrays, NIfTI, Serialization, Statistics, StatsBase

function ResBlock(channels::Int)
    return SkipConnection(Chain(Conv((3, 3, 3), channels => channels, pad=1, relu), Conv((3, 3, 3), channels => channels, pad=1)), +)
end
function build_cnn_improved_64()
    width, depth = 32, 3
    layers = Any[Conv((3, 3, 3), 3 => width, pad=1, relu)]
    for _ in 1:depth; push!(layers, ResBlock(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1, init_weight=Lux.zeros32))
    return Chain(layers...)
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

function debug_cnn()
    dev = Lux.gpu_device(); rng = Random.default_rng()
    cp = "experiments/sciml_dose_refinement/data/checkpoints/CNN_IMPROVED_64/model_best.jls"
    m = build_cnn_improved_64(); θ = dev(ComponentArray(deserialize(cp))); _, st = Lux.setup(rng, m); st = dev(st)
    
    pat_dir = "data/dosimetry_data/FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_0__Pat44"
    sp_i = niread(joinpath(pat_dir, "spect.nii.gz"))[:,:,:,1]
    ct_i = niread(joinpath(pat_dir, "ct.nii.gz"))[:,:,:,1]
    mc_i = niread(joinpath(pat_dir, "dosemap_mc.nii.gz"))[:,:,:,1]
    
    cx, cy, cz = size(sp_i) .÷ 2; xr, yr, zr = cx-31:cx+32, cy-31:cy+32, cz-31:cz+32
    target = mc_i[xr,yr,zr]
    sp_p = Float32.(sp_i[xr,yr,zr]); den_p = Float32.(ct_i[xr,yr,zr]); grad_p = grad_3d(den_p)
    
    # Try 1: Standardized raw
    in1 = dev(reshape(stack([standardize(sp_p), standardize(den_p), standardize(grad_p)]), 64,64,64,3,1))
    p1, _ = Lux.apply(m, in1, θ, Lux.testmode(st)); c1 = cor(vec(Array(p1)), vec(target))
    println("Try 1 (Std Raw): ", c1)
    
    # Try 2: Raw
    in2 = dev(reshape(stack([sp_p, den_p, grad_p]), 64,64,64,3,1))
    p2, _ = Lux.apply(m, in2, θ, Lux.testmode(st)); c2 = cor(vec(Array(p2)), vec(target))
    println("Try 2 (Raw): ", c2)
    
    # Try 3: Std Log
    in3 = dev(reshape(stack([standardize(log1p.(sp_p)), standardize(den_p), standardize(grad_p)]), 64,64,64,3,1))
    p3, _ = Lux.apply(m, in3, θ, Lux.testmode(st)); c3 = cor(vec(Array(p3)), vec(target))
    println("Try 3 (Std Log): ", c3)

    # Try 4: Log output?
    println("Try 1 exp1m: ", cor(vec(exp1m.(Array(p1))), vec(target)))
end
debug_cnn()
