using Pkg
Pkg.activate(".")

using DifferentialEquations
using Optimization
using OptimizationOptimJL
using OptimizationOptimJL
using OptimizationOptimisers
using Optimisers
using SciMLSensitivity
using Lux, LuxCUDA
using Zygote
using Random
using ComponentArrays
using NIfTI

# Physical Constants
const λ_phys = Float32(log(2) / 159.5) # h^-1, Physical decay constant
const k10_pop = Float32(log(2) / 2.0)  # h^-1, Pop excretion rate
const f_pop = 1.0f0  # Assumed organ blood flow scalar
const k_in_pop = 0.00967f0  # h^-1
const k_out_pop = 0.0247f0  # h^-1
const k3 = 0.1f0 # min^-1, assumed 
const k4 = 0.1f0 # min^-1, assumed
const B_MAX_val = 100.0f0 # assumed constant or voxel-wise later
const CF_assumed = 1.0f0 # Camera Calib
const RC_assumed = 1.0f0 # Recovery
const E_bar = 0.147f0 # Average energy in MeV per decay
const μ_repair = 0.46f0 # h^-1, Lea-Catcheside DNA repair constant

# Simple DataLoader for MedImage
function load_patient_data(patient_dir::String)
    ct_path = joinpath(patient_dir, "SPECT_DATA", "CT.nii.gz")
    spect_path = joinpath(patient_dir, "SPECT_DATA", "NM_Vendor.nii.gz")
    dose_path = joinpath(patient_dir, "SPECT_DATA", "Dosemap.nii.gz")
    
    if !isfile(ct_path) || !isfile(spect_path) || !isfile(dose_path)
        println("Missing files in $patient_dir")
        return nothing
    end
    
    ct_full = niread(ct_path)
    spect_full = niread(spect_path)
    dose_full = niread(dose_path)
    
    function extract_3d(arr)
        nd = ndims(arr)
        if nd == 3
            return arr
        elseif nd == 4
            return arr[:, :, :, 1]
        elseif nd == 5
            return arr[:, :, :, 1, 1]
        else
            return arr[:, :, :, 1, 1, 1]
        end
    end
    
    # We must explicitly return the Image headers/data sizes, so keeping the struct wrapper isn't necessary 
    # since `train_on_dataset` expects array formats but uses `ct_img.header` to get pixdim.
    # To keep headers available, we return the original object but slicing an `NIVolume` returns a `Array`.
    # Actually wait. If we extract_3d, we return an Array. But `train_on_dataset` calls `prod(ct_img.header.pixdim[2:4])`.
    
    return ct_full, extract_3d(ct_full), extract_3d(spect_full), extract_3d(dose_full)
end

# 1. CT HU to Density Mapping
function hu_to_density(hu::Real)
    if hu <= 0
        return 1.0f0 + 0.001f0 * Float32(hu)
    else
        return 1.0f0 + 0.0007f0 * Float32(hu)
    end
end

# 2. Neural Network Corrector Architecture 𝒩_θ
# Input: (A_total, ρ, ∇ρ) -> Output: [Dose Residual]
# Lux explicitly tracks pure parameters (ps) and state (st) cleanly without Mutating errors
# To bypass Mutating array errors in Zygote over CPU tensors natively, we use a Parallel layer construct in Lux
function build_neural_transport_model()
    # Parallel layer processes each channel independently with 3D Convolutions
    # and then adds them together before the final solver layers 
    # This completely circumvents `cat` or `stack` mutating array gradient errors
    
    branch_A = Conv((3, 3, 3), 1 => 4, pad=1, relu)
    branch_ρ = Conv((3, 3, 3), 1 => 4, pad=1, relu)
    branch_∇ρ = Conv((3, 3, 3), 1 => 4, pad=1, relu)
    
    parallel_branches = Parallel(+, branch_A, branch_ρ, branch_∇ρ)
    
    model = Chain(
        parallel_branches,
        Conv((3, 3, 3), 4 => 8, pad=1, relu),
        Conv((3, 3, 3), 8 => 1, pad=1)
    )
    return model
end

# Set up the Random number generator and initialize Lux model
const rng = Random.default_rng()
Random.seed!(rng, 42)
const NN_model = build_neural_transport_model()
const ps, st = Lux.setup(rng, NN_model)
const NN_params_flat = ComponentArray(ps)

# Helper to format tensors into a 5D Batch for Lux Parallel layers
function safe_format_channels(A, ρ, ∇ρ)
    A_5d = reshape(A, size(A)..., 1, 1)
    ρ_5d = reshape(ρ, size(ρ)..., 1, 1)
    ∇ρ_5d = reshape(∇ρ, size(∇ρ)..., 1, 1)
    
    return (A_5d, ρ_5d, ∇ρ_5d)
end

# Gradient of Density Approximation (Simple differences for demonstration)
function compute_density_gradient(ρ_map::Array{Float32, 3})
    # Placeholder: In practice use Sobel filters or Enzyme/Zygote compatible gradients
    # For now, return zero tensors of same size or simple finite diffs
    ∇ρ = zeros(Float32, size(ρ_map))
    # ... actual gradient logic ...
    return ∇ρ
end

function predict_dosemap(p_state, theta_opt)
    mass_map = p_state.vol_map .* p_state.ρ_map
    ρ_map = p_state.ρ_map
    ∇ρ_map = p_state.∇ρ_map
    
    A0_obs = p_state.A0_obs
    A0 = A0_obs .* (RC_assumed / CF_assumed)
    
    dev = Lux.gpu_device()
    A_blood = dev(Float32[sum(A0) * 0.1f0])
    A_free = A0 .* 0.45f0
    A_bound = A0 .* 0.45f0
    BED = zero(A0)
    
    u0 = ComponentArray(A_blood=A_blood, A_free=A_free, A_bound=A_bound, BED=BED)
    tspan = (0.0f0, 300.0f0) # Realistic ~12.5 days integration
    
    st_fixed = p_state.st
    
    function ude_func(u, p, t)
        # p is explicitly JUST theta
        theta_local = p
        
        A_blood_local = u.A_blood
        A_free_local = u.A_free
        A_bound_local = u.A_bound
        BED_local = u.BED
        
        sum_uptake = sum(f_pop .* k_in_pop .* A_blood_local)
        sum_washout = sum(k_out_pop .* A_free_local)
        
        dA_blood = - (k10_pop * 1.0f0 + sum_uptake + λ_phys) * A_blood_local .+ sum_washout
        
        dA_free  = (f_pop .* k_in_pop .* A_blood_local) .- (k_out_pop .* A_free_local) .- (λ_phys .* A_free_local)
        dA_bound = (k3 .* A_free_local .* (1.0f0 .- A_bound_local ./ B_MAX_val)) .- (k4 .* A_bound_local) .- (λ_phys .* A_bound_local)
        
        # 4. Neural Transport
        A_total = A_free_local .+ A_bound_local
        
        inputs = safe_format_channels(A_total, ρ_map, ∇ρ_map)
        dD_physical_residual, _ = Lux.apply(NN_model, inputs, theta_local, st_fixed)
        dD_physical_residual = reshape(dD_physical_residual, size(A_total))
        
        dD_base = A_total .* (E_bar * 10.0f0) 
        dD_phys = (dD_base .+ dD_physical_residual) ./ (mass_map .+ 1f-5)
        
        dBED = dD_phys .- μ_repair .* BED_local
        
        return ComponentArray(A_blood=dA_blood, A_free=dA_free, A_bound=dA_bound, BED=dBED)
    end
    
    prob = ODEProblem(ude_func, u0, tspan, theta_opt)
    
    # We use Euler for this fast dummy test to avoid massive AD compilation
    # in an actual run you'd use Tsit5() and a longer tspan with saveat
    sol = solve(prob, Tsit5(), saveat=[300.0f0], reltol=1f-1, abstol=1f-1, 
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    
    # Extract final BED
    final_bed = sol.u[end].BED
    return final_bed
end

function loss_function(p_state, theta_opt)
    predicted_bed = predict_dosemap(p_state, theta_opt)
    target_bed = p_state.target_dosemap
    
    # Use Mean Absolute Error to prevent Float32 overflow on sum of squares
    diff = abs.(predicted_bed .- target_bed)
    mae_loss = sum(diff) / length(target_bed)
    
    return mae_loss
end


function train_on_dataset(dataset_dir::String, epochs::Int=5)
    println("Initializing dataset training over: ", dataset_dir)
    # Get all patient subdirectories
    patients = filter(isdir, readdir(dataset_dir, join=true))
    
    dev = Lux.gpu_device()
    println("Using device: ", dev)
    
    # Initialize trainable parameter copy from the network
    θ = dev(copy(NN_params_flat))
    opt_state = Optimisers.setup(Optimisers.Adam(1f-4), θ)
    
    for epoch in 1:epochs
        println("=== Epoch $epoch ===")
        epoch_loss = 0.0f0
        valid_patients = 0
        
        for patient_dir in patients
            println("  Processing patient: ", basename(patient_dir))
            
            data = load_patient_data(patient_dir)
            if data === nothing 
                continue 
            end
            
            ct_full, ct_img, spect_img, dose_img = data
            
            # --- RANDOM PATCH LOGIC ---
            min_x = min(size(ct_img, 1), size(spect_img, 1), size(dose_img, 1))
            min_y = min(size(ct_img, 2), size(spect_img, 2), size(dose_img, 2))
            min_z = min(size(ct_img, 3), size(spect_img, 3), size(dose_img, 3))
            p_size = 64
            
            x_max = max(1, min_x - p_size)
            y_max = max(1, min_y - p_size)
            z_max = max(1, min_z - p_size)
            
            # Cache dose to RAM to prevent NIfTI mmap thrashing
            dose_ram = Float32.(dose_img)
            replace!(dose_ram, NaN32 => 0.0f0)
            
            valid_indices = findall(x -> x > 1f-4, dose_ram)
            if isempty(valid_indices)
                println("Skipping patient: entirely empty validation cache.")
                continue
            end
            
            idx = rand(valid_indices)
            cx, cy, cz = idx.I
            
            p_size = 64
            hx = p_size ÷ 2
            
            # Clamp starting coordinates to guarantee window fits bounds
            x_start = clamp(cx - hx, 1, max(1, min_x - p_size + 1))
            y_start = clamp(cy - hx, 1, max(1, min_y - p_size + 1))
            z_start = clamp(cz - hx, 1, max(1, min_z - p_size + 1))
            
            x_end = min(x_start + p_size - 1, min_x)
            y_end = min(y_start + p_size - 1, min_y)
            z_end = min(z_start + p_size - 1, min_z)
            
            ct_crop = Float32.(ct_img[x_start:x_end, y_start:y_end, z_start:z_end])
            spect_crop = Float32.(spect_img[x_start:x_end, y_start:y_end, z_start:z_end])
            dose_crop = dose_ram[x_start:x_end, y_start:y_end, z_start:z_end]
            
            replace!(ct_crop, NaN32 => 0.0f0)
            replace!(spect_crop, NaN32 => 0.0f0)
            
            vol_voxel = prod(ct_full.header.pixdim[2:4])
            vol_map = fill(Float32(vol_voxel), size(ct_crop))
            ρ_map = hu_to_density.(ct_crop)
            ∇ρ_map = compute_density_gradient(ρ_map)
            
            p_state = (
                vol_map = dev(vol_map),
                ρ_map = dev(ρ_map),
                ∇ρ_map = dev(∇ρ_map),
                A0_obs = dev(spect_crop),
                target_dosemap = dev(dose_crop),
                st = dev(st)
            )
            
            # --- Compute AutoZygote Gradient ---
            loss, gs = Zygote.withgradient(θ_val -> loss_function(p_state, θ_val), θ)
            
            println("    Loss: ", loss)
            epoch_loss += loss
            valid_patients += 1
            
            # --- Update Model Parameters ---
            opt_state, θ = Optimisers.update(opt_state, θ, gs[1])
        end
        
        if valid_patients > 0
            println("  Average Epoch Loss: ", epoch_loss / valid_patients)
        end
    end
    
    println("Finished native Lux network dataset training.")
    return θ
end

# Test run on the entire target dataset folder
test_dir = joinpath("test_data", "dataset_Lu")
if isdir(test_dir)
    train_on_dataset(test_dir, 5)
end
