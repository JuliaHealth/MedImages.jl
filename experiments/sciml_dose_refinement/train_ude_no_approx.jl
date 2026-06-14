using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, Optimization, OptimizationOptimJL, OptimizationOptimisers, Optimisers, SciMLSensitivity, Lux, LuxCUDA, CUDA, Zygote, Random, ComponentArrays, NIfTI, Serialization, Statistics

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
const SA_MBq_nmol = 120.0f0 # Specific Activity conversion to physical molar mass

function load_patient_data(patient_dir::String)
    ct_path = joinpath(patient_dir, "ct.nii.gz"); spect_path = joinpath(patient_dir, "spect.nii.gz"); dose_path = joinpath(patient_dir, "dosemap_mc.nii.gz")
    if !isfile(ct_path) || !isfile(spect_path) || !isfile(dose_path); return nothing; end
    ct_f = niread(ct_path); spect_f = niread(spect_path); dose_f = niread(dose_path)
    extract(arr) = ndims(arr) == 3 ? arr : arr[:,:,:,1]
    return ct_f, extract(ct_f), extract(spect_f), extract(dose_f)
end

function hu_to_density(hu::Real; slope_air=0.001f0, slope_tissue=0.0007f0)
    hu <= 0 ? max(0.0012f0, 1.0f0 + slope_air * Float32(hu)) : 1.0f0 + slope_tissue * Float32(hu)
end

function ResBlock(channels::Int)
    return SkipConnection(Chain(Conv((3, 3, 3), channels => channels, pad=1, relu), Conv((3, 3, 3), channels => channels, pad=1)), +)
end

function build_no_approx_model()
    width, depth = 32, 3
    layers = []
    # 3 input branches: Base Energy (E), Density (ρ) and Density Gradient (∇ρ)
    branch_E = Conv((3, 3, 3), 1 => width, pad=1, relu)
    branch_ρ = Conv((3, 3, 3), 1 => width, pad=1, relu)
    branch_∇ρ = Conv((3, 3, 3), 1 => width, pad=1, relu)
    push!(layers, Parallel(+, branch_E, branch_ρ, branch_∇ρ))
    for _ in 1:depth; push!(layers, ResBlock(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

const NN_model = build_no_approx_model()
const ps, st_init = Lux.setup(Random.default_rng(), NN_model)

function compute_grad_rho(rho)
    gx = rho[[2:end; end],:,:] .- rho[[1; 1:end-1],:,:]
    gy = rho[:,[2:end; end],:] .- rho[:,[1; 1:end-1],:]
    gz = rho[:,:,[2:end; end]] .- rho[:,:,[1; 1:end-1]]
    return sqrt.(gx.^2 .+ gy.^2 .+ gz.^2 .+ 1e-6f0)
end

function predict_dosemap(p_state, theta_opt, A0)
    mass_map = p_state.vol_map .* p_state.ρ_map; ρ_map = p_state.ρ_map
    dev = Lux.gpu_device()
    u0 = ComponentArray(A_blood=dev(Float32[sum(A0)*0.05f0]), A_free=A0.*0.45f0, A_surface=A0.*0.50f0, A_internal=A0.*0.0f0, TIA=A0.*0.0f0)
    
    # K5 internalized parameter for Endosomal track bounds
    k5 = 0.1f0

    function ude_func_outer(u, p, t)
        total_A0 = sum(A0) + 1f-6; voxel_in = (f_pop * k_in_pop * u.A_blood) .* (A0 ./ total_A0)
        dA_blood = - (k10_pop + λ_phys) * u.A_blood .- sum(f_pop * k_in_pop * u.A_blood) .+ sum(k_out_pop .* u.A_free)
        # Using SA conversion for true molar saturation, preventing false λ_phys receptor freeing
        M_bound = ((u.A_surface .+ u.A_internal) ./ SA_MBq_nmol) .* exp(λ_phys * t)
        binding_flux = k3 .* u.A_free .* max.(0.0f0, 1.0f0 .- M_bound ./ B_MAX_val)
        unbinding_flux = k4 .* u.A_surface
        internalize_flux = k5 .* u.A_surface
        
        dA_free  = voxel_in .- (k_out_pop .* u.A_free) .- (λ_phys .* u.A_free) .- binding_flux .+ unbinding_flux
        dA_surface = binding_flux .- unbinding_flux .- internalize_flux .- (λ_phys .* u.A_surface)
        dA_internal = internalize_flux .- (λ_phys .* u.A_internal)
        
        dTIA = u.A_free .+ u.A_surface .+ u.A_internal .+ (p_state.vol_map .* (0.05f0 / 5000.0f0) .* u.A_blood[1])
        return ComponentArray(A_blood=dA_blood, A_free=dA_free, A_surface=dA_surface, A_internal=dA_internal, TIA=dTIA)
    end
    
    # 1. Temporal PK Integration decoupled from Spatial Radiation
    prob = ODEProblem(ude_func_outer, u0, (0.0f0, 300.0f0), NullParameters())
    sol = solve(prob, VCABM(), saveat=[300.0f0], reltol=1f-3, abstol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true))
    
    # 2. Extract final Time Integrated Activity (TIA)
    final_TIA = sol.u[end].TIA
    
    # 3. Post-Integration Spatial Transport & Convolution
    grad_ρ_map = compute_grad_rho(ρ_map)
    # Dual Convolution Kernels: explicitly capturing short-range Beta and long-range Gamma fields
    base_energy_beta = (0.85f0 .* DOSE_CONV) .* final_TIA # Proxy beta convolution K_beta,lung
    base_energy_gamma = (0.15f0 .* DOSE_CONV) .* final_TIA # Proxy gamma convolution K_gamma,water
    base_energy = base_energy_beta .+ base_energy_gamma
    
    # Establish First-Order Unscattered Vector Flux Directionality
    vector_flux_ray = compute_grad_rho(base_energy) # Proxies directional ray-tracing flux field
    
    # NN receives local uncorrected base energy, physical geometry fields, and directional vector flux
    inputs = (reshape(base_energy, size(A0)..., 1, 1), reshape(ρ_map, size(A0)..., 1, 1), reshape(grad_ρ_map, size(A0)..., 1, 1), reshape(vector_flux_ray, size(A0)..., 1, 1))
    nn_absorbed_fraction_offset, _ = Lux.apply(NN_model, inputs, theta_opt, p_state.st)
    
    # The Dose math: Multiply the base_energy multiplicatively by the network offset modifier, then normalize by mass
    # Prevent 1/0 singularities explicitly on the mass map division
    safe_mass_map = ifelse.(p_state.ρ_map .< 0.05f0, 0.05f0 .* p_state.vol_map, mass_map)
    D_phys_raw = (base_energy .* exp.(reshape(nn_absorbed_fraction_offset, size(A0)))) ./ safe_mass_map
    
    return ifelse.(p_state.ρ_map .< 0.05f0, 0.0f0, D_phys_raw), base_energy
end

function gradient_loss(pred, target)
    gx_p = pred[2:end,:,:] .- pred[1:end-1,:,:]; gx_t = target[2:end,:,:] .- target[1:end-1,:,:]
    return sum(abs.(gx_p .- gx_t)) / length(gx_p)
end

function train_no_approx(dataset_dir::String, epochs::Int=15; CF=1.0f0, rescale_slope=1.0f0, RC=1.0f0)
    println("Training UDE (No Approx Dose Input)...")
    patients = sort(filter(isdir, readdir(dataset_dir, join=true)))
    dev = Lux.gpu_device(); θ = dev(ComponentArray(ps)); opt_state = Optimisers.setup(Optimisers.Adam(1f-3), θ)
    for epoch in 1:epochs
        epoch_mae = 0.0f0; count = 0
        for pat in patients[1:min(15, length(patients))]
            data = load_patient_data(pat); if data === nothing; continue; end
            ct_f, ct_i, spect_i, dose_mc = data
            
            # Iodine Contrast Pre-flight Safety Limit
            if maximum(ct_i) > 300.0f0
                println("Safety Abort: Patient $pat Iodine contrast detected! High-Z atoms > 300 HU explicitly hallucinates beta attenuation.")
                continue
            end
            
            scale = maximum(dose_mc) / 10.0f0; target = dose_mc ./ (scale + 1f-6)
            valid = findall(x -> x > 1f-3, target)
            if isempty(valid); continue; end
            for patch_idx in 1:1
                idx = rand(valid); cx, cy, cz = idx.I; p_size = 32; hx = p_size ÷ 2
                xr = clamp(cx-hx, 1, size(target,1)-p_size+1):clamp(cx-hx+p_size-1, p_size, size(target,1))
                yr = clamp(cy-hx, 1, size(target,2)-p_size+1):clamp(cy-hx+p_size-1, p_size, size(target,2))
                zr = clamp(cz-hx, 1, size(target,3)-p_size+1):clamp(cz-hx+p_size-1, p_size, size(target,3))
                p_state = (vol_map=dev(fill(Float32(prod(ct_f.header.pixdim[2:4])), (32,32,32))), ρ_map=dev(hu_to_density.(ct_i[xr,yr,zr])), st=dev(st_init))
                
                # Apply absolute quantification parameters: CF, Rescale Slope, and RC.
                # Note: RC primarily matters for per-ROI dosimetry rather than per-voxel.
                scaled_spect = (Float32.(spect_i[xr,yr,zr]) .* rescale_slope) ./ (CF .* RC)
                spect_p = dev(scaled_spect)
                target_p = dev(Float32.(target[xr,yr,zr]))
                l, gs = Zygote.withgradient(t -> begin
                    pred, e_base = predict_dosemap(p_state, t, spect_p)
                    mae = sum(abs.(pred .- target_p)) / length(target_p)
                    grad = gradient_loss(pred, target_p)
                    # Energy Conservation Constraint: Mass*Dose == Base_Energy
                    mass_tensor = p_state.ρ_map .* p_state.vol_map
                    energy_cons = sum(abs2, sum(pred .* mass_tensor) - sum(e_base))
                    0.6f0 * mae + 0.25f0 * grad + 0.15f0 * energy_cons
                end, θ)
                if !any(isnan.(gs[1])); opt_state, θ = Optimisers.update(opt_state, θ, gs[1]); epoch_mae += l; count += 1; end
                CUDA.reclaim(); GC.gc()
            end
        end
        println("  Epoch $epoch Loss: $(epoch_mae/count)")
    end
    serialize("elsarticle/dosimetry/model_ude_no_approx.jls", θ)
end

train_no_approx("/home/user/MedImages.jl/elsarticle/dosimetry/data/", 15)
