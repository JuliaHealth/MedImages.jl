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

function build_ude_heavy()
    width, depth = 16, 3
    branch_A = Conv((3, 3, 3), 1 => width, pad=1, relu); branch_ρ = Conv((3, 3, 3), 1 => width, pad=1, relu); branch_approx = Conv((3, 3, 3), 1 => width, pad=1, relu)
    layers = Any[Parallel(+, branch_A, branch_ρ, branch_approx)]
    for _ in 1:depth; push!(layers, ResBlock(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

function build_ude_no_approx()
    width, depth = 32, 3
    branch_A = Conv((3, 3, 3), 1 => width, pad=1, relu); branch_ρ = Conv((3, 3, 3), 1 => width, pad=1, relu)
    layers = Any[Parallel(+, branch_A, branch_ρ)]
    for _ in 1:depth; push!(layers, ResBlock(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

function build_stabilized_cnn()
    width, depth = 32, 3
    layers = Any[Conv((3, 3, 3), 3 => width, pad=1, relu)]
    for _ in 1:depth; push!(layers, ResBlock(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

# --- Metrics ---

function pearson_corr(x, y)
    return cor(reshape(x, :), reshape(y, :))
end

# --- Inference ---

function predict_ude(model, θ, st, p_state, A0, use_approx)
    mass_map = p_state.vol_map .* p_state.ρ_map; ρ_map = p_state.ρ_map; dev = Lux.cpu_device()
    u0 = ComponentArray(A_blood=dev(Float32[sum(A0)*0.05f0]), A_free=A0.*0.45f0, A_bound=A0.*0.50f0, DOSE=zero(A0))
    function ude_func(u, p, t)
        total_A0 = sum(A0) + 1f-6; voxel_in = (f_pop * k_in_pop * u.A_blood) .* (A0 ./ total_A0)
        dA_blood = - (k10_pop + λ_phys) * u.A_blood .- (f_pop * k_in_pop * u.A_blood) .+ sum(k_out_pop .* u.A_free)
        dA_free  = voxel_in .- (k_out_pop .* u.A_free) .- (λ_phys .* u.A_free)
        dA_bound = (k3 .* u.A_free .* (1.0f0 .- u.A_bound ./ B_MAX_val)) .- (k4 .* u.A_bound) .- (λ_phys .* u.A_bound)
        A_t = u.A_free .+ u.A_bound
        A_std = (A_t .- mean(A_t)) ./ (std(A_t) + 1f-6)
        inputs = use_approx ? (reshape(A_std, size(A0)..., 1, 1), reshape(ρ_map, size(A0)..., 1, 1), reshape(p_state.approx, size(A0)..., 1, 1)) : (reshape(A_std, size(A0)..., 1, 1), reshape(ρ_map, size(A0)..., 1, 1))
        nn_out, _ = Lux.apply(model, inputs, p, st)
        dD_phys = softplus.((A_t .* DOSE_CONV) ./ (mass_map .+ 1f-4) .+ reshape(nn_out, size(A0)))
        return ComponentArray(A_blood=dA_blood, A_free=dA_free, A_bound=dA_bound, DOSE=ifelse.(ρ_map .< 0.1f0, 0.0f0, dD_phys))
    end
    prob = ODEProblem(ude_func, u0, (0.0f0, 300.0f0), θ)
    sol = solve(prob, Tsit5(), saveat=[300.0f0], reltol=1f-2, abstol=1f-2)
    res = sol.u[end].DOSE
    # println("    DEBUG: A0 mean=$(mean(A0)), res mean=$(mean(res)), res max=$(maximum(res))")
    return res
end

function run_comprehensive_eval()
    dev = Lux.cpu_device(); rng = Random.default_rng()
    # Load Models
    m_heavy = build_ude_heavy(); θ_heavy = dev(deserialize("data/checkpoints/heavy/model_heavy.jls")); _, st_heavy = Lux.setup(rng, m_heavy); st_heavy = dev(st_heavy)
    m_noapp = build_ude_no_approx(); θ_noapp = dev(deserialize("data/checkpoints/UDE_NO_APPROX/model_best.jls")); _, st_noapp = Lux.setup(rng, m_noapp); st_noapp = dev(st_noapp)
    m_cnn = build_stabilized_cnn(); θ_cnn = dev(deserialize("data/checkpoints/CNN_APPROX/model_best_CNN_APPROX.jls")); _, st_cnn = Lux.setup(rng, m_cnn); st_cnn = dev(st_cnn)

    dataset_dir = "data/dosimetry_data"; patients = sort(filter(isdir, readdir(dataset_dir, join=true)))
    if isempty(patients)
        println("Error: No patients found in $dataset_dir")
        return
    end
    
    pat47 = filter(p -> occursin("Tc_1__Pat47", p), patients)
    rng_split = Random.Xoshiro(42); shuffle!(rng_split, patients)
    val_patients = patients[Int(round(0.8 * length(patients)))+1:end]
    if !isempty(pat47) && !(pat47[1] in val_patients)
        push!(val_patients, pat47[1])
    end
    if isempty(val_patients); val_patients = patients; end

    println("\nComprehensive Comparative Analysis (Pearson Correlation)")
    println("-"^110)
    header = rpad("Patient", 25) * " | " * rpad("UDE Heavy", 12) * " | " * rpad("UDE No-Approx", 14) * " | " * rpad("CNN+Approx", 12) * " | " * rpad("Baseline", 12)
    println(header)
    println("-"^110)

    for pat_dir in val_patients
        name = basename(pat_dir)
        ct_p = joinpath(pat_dir, "ct.nii.gz"); sp_p = joinpath(pat_dir, "spect.nii.gz"); mc_p = joinpath(pat_dir, "dosemap_mc.nii.gz"); ap_p = joinpath(pat_dir, "dosemap_approx.nii.gz")
        
        if !isfile(ct_p) || !isfile(sp_p) || !isfile(mc_p) || !isfile(ap_p)
            continue
        end

        ct_f = niread(ct_p); sp_f = niread(sp_p); dose_mc = niread(mc_p); approx_f = niread(ap_p)
        ct_i = ndims(ct_f) == 3 ? ct_f : ct_f[:,:,:,1]; sp_i = ndims(sp_f) == 3 ? sp_f : sp_f[:,:,:,1]; mc_i = ndims(dose_mc) == 3 ? dose_mc : dose_mc[:,:,:,1]; app_i = ndims(approx_f) == 3 ? approx_f : approx_f[:,:,:,1]
        
        # Center crop 32^3 with bounds check
        sz = size(ct_i)
        cx, cy, cz = sz .÷ 2
        xr = max(1, cx-15):min(sz[1], cx+16)
        yr = max(1, cy-15):min(sz[2], cy+16)
        zr = max(1, cz-15):min(sz[3], cz+16)
        
        # Ensure exactly 32^3
        if length(xr) < 32 || length(yr) < 32 || length(zr) < 32
            continue
        end
        hu_to_den(hu) = hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) : 1.0f0 + 0.0007f0 * Float32(hu)
        den_p = hu_to_den.(ct_i[xr,yr,zr]); vol_p = Float32(prod(ct_f.header.pixdim[2:4])); A0 = dev(Float32.(sp_i[xr,yr,zr])); T = mc_i[xr,yr,zr]
        
        # Heavy UDE
        p_state_heavy = (vol_map=dev(fill(vol_p, (32,32,32))), ρ_map=dev(den_p), approx=dev(Float32.(app_i[xr,yr,zr])))
        c_heavy = round(cor(reshape(Array(predict_ude(m_heavy, θ_heavy, st_heavy, p_state_heavy, A0, true)), :), reshape(T, :)), digits=4)
        # No-Approx UDE
        p_state_noapp = (vol_map=dev(fill(vol_p, (32,32,32))), ρ_map=dev(den_p))
        c_noapp = round(cor(reshape(Array(predict_ude(m_noapp, θ_noapp, st_noapp, p_state_noapp, A0, false)), :), reshape(T, :)), digits=4)
        # CNN+Approx
        sp_std = (sp_i[xr,yr,zr] .- mean(sp_i[xr,yr,zr])) ./ (std(sp_i[xr,yr,zr]) + 1f-6)
        den_std = (den_p .- mean(den_p)) ./ (std(den_p) + 1f-6)
        app_std = (app_i[xr,yr,zr] .- mean(app_i[xr,yr,zr])) ./ (std(app_i[xr,yr,zr]) + 1f-6)
        input_cnn = dev(reshape(stack([Float32.(sp_std), Float32.(den_std), Float32.(app_std)]), 32, 32, 32, 3, 1))
        pred_cnn, _ = Lux.apply(m_cnn, input_cnn, θ_cnn, Lux.testmode(st_cnn))
        t_max = maximum(mc_i)
        pred_cnn_phys = softplus.(Array(reshape(pred_cnn, 32, 32, 32)) .+ app_i[xr,yr,zr] ./ (t_max/10.0f0 + 1f-6))
        c_cnn_ap = round(cor(reshape(pred_cnn_phys, :), reshape(T, :)), digits=4)
        # Baseline
        c_base = round(cor(reshape(app_i[xr,yr,zr], :), reshape(T, :)), digits=4)

        println(rpad(name[1:min(end,23)], 25), " | ", rpad(c_heavy, 12), " | ", rpad(c_noapp, 14), " | ", rpad(c_cnn_ap, 12), " | ", rpad(c_base, 12))
        GC.gc()
    end
end
run_comprehensive_eval()
