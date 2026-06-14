using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, Lux, LuxCUDA, CUDA, Random, ComponentArrays, NIfTI, Serialization, Statistics

const λ_phys = Float32(log(2) / 159.5) 
const k10_pop = Float32(log(2) / 40.0) 
const f_pop = 1.0f0  
const k_in_pop = 0.01f0  
const k_out_pop = 0.02f0 
const k3 = 0.05f0 
const k4 = 0.01f0 
const DOSE_CONV = 8.478f-8 

function evaluate_ode_baseline()
    pat = "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat48"
    pat_dir = "data/dosimetry_data/$pat"
    
    ct_f = niread(joinpath(pat_dir, "ct.nii.gz"))
    sp_f = niread(joinpath(pat_dir, "spect.nii.gz"))
    mc_f = niread(joinpath(pat_dir, "dosemap_mc.nii.gz"))
    
    extract(f) = ndims(f) == 3 ? f : f[:,:,:,1]
    ct_i = extract(ct_f); sp_i = extract(sp_f); mc_i = extract(mc_f)
    cx, cy, cz = size(ct_i) .÷ 2; xr, yr, zr = cx-31:cx+32, cy-31:cy+32, cz-31:cz+32
    
    target = mc_i[xr,yr,zr]
    hu_to_den(hu) = hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) : 1.0f0 + 0.0007f0 * Float32(hu)
    den_raw = hu_to_den.(ct_i[xr,yr,zr])
    vol_p = Float32(prod(ct_f.header.pixdim[2:4]))
    sp_p = Float32.(sp_i[xr,yr,zr])
    
    # ODE Setup
    u0 = ComponentArray(A_blood=[sum(sp_p)*0.05f0], A_free=sp_p.*0.45f0, A_bound=sp_p.*0.50f0, DOSE=zero(sp_p))
    
    function f(u,p,t)
        A_t = u.A_free .+ u.A_bound
        # Pure physical term, no NN
        dD = (A_t .* DOSE_CONV) ./ (vol_p .* den_raw .+ 1f-4)
        dA_blood = -(k10_pop + λ_phys) .* u.A_blood
        dA_free = -(k_out_pop + λ_phys) .* u.A_free
        dA_bound = -(k4 + λ_phys) .* u.A_bound
        return ComponentArray(A_blood=dA_blood, A_free=dA_free, A_bound=dA_bound, DOSE=dD)
    end
    
    prob = ODEProblem(f, u0, (0.0f0, 300.0f0))
    sol = solve(prob, Tsit5(), saveat=[300.0f0])
    pred = sol.u[end].DOSE
    
    println("ODE Baseline Pearson: ", cor(reshape(pred, :), reshape(target, :)))
    println("ODE Baseline Mean: ", mean(pred))
    
    # Static Baseline
    static_pred = (sp_p .* DOSE_CONV) ./ (vol_p .* den_raw .+ 1f-4)
    println("Static Baseline Pearson: ", cor(reshape(static_pred, :), reshape(target, :)))
end

evaluate_ode_baseline()
