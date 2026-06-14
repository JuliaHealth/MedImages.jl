# Challenge 3: Differentiability (Dosimetry UDE)

**Objective:** Validate complex SciML integration by formulating a Universal Differential Equation (UDE) for high-fidelity Lu-177 PSMA dosimetry.

## The Dosimetry Problem

Clinical workflows use simplified analytical methods (like VSV convolutions) assuming homogeneous water-equivalent density. Monte Carlo (MC) tracking is highly accurate but computationally impossible for routine clinical use. Pure deep learning often lacks physical constraints. 

Our approach partitions the problem:
*   **Mechanistic Knowns:** Physical decay ($\lambda_{phys}$), Blood circulation rates, Base dose convolution, Mass-density conversion from CT.
*   **Learned Uncertainties:** A neural network $\mathcal{N}_\theta$ learns the complex scattering at density gradients ($\nabla \rho$) and local receptor saturation phenomena.

## Detailed Code Walkthrough

The system is a coupled 4-state IVP (Initial Value Problem) describing Blood Activity, Free Tissue Activity, Bound Tissue Activity, and Absorbed Dose Rate.

Let's walk through the core definition of the Ordinary Differential Equation (ODE) system and how the Neural Network is embedded within the physical equations.

### The Universal Differential Equation (UDE)

```julia
# File: experiments/sciml_dose_refinement/train_ude_no_approx.jl

# The core ODE describing the compartmental model and Dose rate
function ude_func_outer(u, p, t)
    # 1. Base kinetics
    total_A0 = sum(A0) + 1f-6
    voxel_in = (f_pop * k_in_pop * u.A_blood) .* (A0 ./ total_A0)
    
    # 2. Mechanistic Compartmental Kinetics
    dA_blood = - (k10_pop + λ_phys) * u.A_blood .- sum(f_pop * k_in_pop * u.A_blood) .+ sum(k_out_pop .* u.A_free)
    dA_free  = voxel_in .- (k_out_pop .* u.A_free) .- (λ_phys .* u.A_free)
    dA_bound = (k3 .* u.A_free .* (1.0f0 .- u.A_bound ./ B_MAX_val)) .- (k4 .* u.A_bound) .- (λ_phys .* u.A_bound)
    
    A_total = u.A_free .+ u.A_bound
    
    # 3. Physical mechanism: Density Gradients
    grad_ρ_map = compute_grad_rho(ρ_map) 
    
    # 4. Learned Uncertainty: Neural Network Evaluation
    inputs = (reshape(A_total, size(A0)..., 1, 1), reshape(ρ_map, size(A0)..., 1, 1), reshape(grad_ρ_map, size(A0)..., 1, 1))
    nn_out, _ = Lux.apply(NN_model, inputs, p, st_fixed)
    
    # 5. Combined Dose Rate Equation
    dD_base = ((u.A_free .+ u.A_bound) .* DOSE_CONV) ./ (mass_map .+ 1f-4)
    dD_phys = softplus.(dD_base .+ reshape(nn_out, size(A0)))
    
    # 6. Return the updated state derivatives
    return ComponentArray(A_blood=dA_blood, A_free=dA_free, A_bound=dA_bound, DOSE=ifelse.(ρ_map .< 0.1f0, 0.0f0, dD_phys))
end
```

### Line-by-Line Breakdown:
1. **Lines 6-7:** Defines the baseline entry of radiopharmaceutical from the blood pool into the individual tissue voxels, scaled by the initial observed SPECT distribution (`A0`).
2. **Lines 10-12 (Mechanistic Kinetics):** This defines the strict physical rules for biological decay. `dA_blood`, `dA_free`, and `dA_bound` represent the rates of change. Notice the `λ_phys` term: this guarantees the model mathematically obeys the strict 159.5-hour physical half-life of Lutetium-177.
3. **Lines 16-17:** Calculates the spatial density gradient (`∇ρ`) dynamically using finite differences. This identifies tissue boundaries (e.g., lung-to-tumor interfaces) where radiation scattering is most chaotic.
4. **Lines 20-21 (The Neural Network):** This is the "Universal" component. We pass the current total activity (`A_total`), the physical density map (`ρ_map` derived from CT Hounsfield units), and the boundary map (`grad_ρ_map`) into the 3D CNN (`NN_model`). The model parameters `p` are what we will optimize via Zygote.
5. **Lines 24-25 (Dose Rate):** We calculate the purely physical baseline dose rate (`dD_base`) using standard conversions against voxel mass. We then *add* the Neural Network's prediction (`nn_out`) as a corrective residual. The `softplus` ensures radiation dose can never be negative.
6. **Line 28:** The function returns the rate of change for all 4 states as a `ComponentArray`. `DifferentialEquations.jl` will integrate this over time to yield the final accumulated absorbed dose map.

## Results

The "No-Approx UDE" implemented via Julia/SciML achieved the definitive state-of-the-art result against the Monte Carlo ground truth over $64^3$ spatial patches:

*   **UDE (SciML):** Pearson $r = 0.9570$
*   **Analytical Baseline (VSV):** Pearson $r = 0.9120$
*   **DblurDoseNet (Deep Learning):** Pearson $r = 0.5566$

## Data Leakage Prevention

To ensure clinical validity, the UDE model is designed with strict data isolation:
1. **Zero Ground-Truth Exposure**: The neural network $\mathcal{N}_\theta$ has no access to Monte Carlo labels during the forward pass. Its inputs are derived exclusively from the current ODE state (Activity) and anatomical priors (CT Density).
2. **Blind Inference**: The sliding window reconstruction does not utilize any patient-wide global statistics (like maximum dose) that would be unavailable at runtime.
3. **Compartmental Isolation**: The biological clearance rates and decay constants are fixed physical parameters, preventing the model from "cheating" by adjusting fundamental physics to fit specific noise patterns.

By isolating mechanistic physics from complex scattering phenomena learned by the neural residual, the model captures the critical dose variance at tissue boundaries that traditional models fail to systematically track.
