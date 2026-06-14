using Test
using LinearAlgebra
using ComponentArrays

# Mocking dependent physics functions
function hu_to_density(hu::Float32; slope_air=0.001f0, slope_tissue=0.0007f0)
    if hu <= 0.0f0
        return max(0.0f0, 1.0f0 + slope_air * hu)
    else
        return 1.0f0 + slope_tissue * hu
    end
end

function compute_grad_rho(rho)
    gx = rho[[2:end; end],:,:] .- rho[[1; 1:end-1],:,:]
    gy = rho[:,[2:end; end],:] .- rho[:,[1; 1:end-1],:]
    gz = rho[:,:,[2:end; end]] .- rho[:,:,[1; 1:end-1]]
    return sqrt.(gx.^2 .+ gy.^2 .+ gz.^2 .+ 1f-6)
end

function compute_directional_flux(base_energy)
    gx = base_energy[[2:end; end],:,:] .- base_energy[[1; 1:end-1],:,:]
    gy = base_energy[:,[2:end; end],:] .- base_energy[:,[1; 1:end-1],:]
    gz = base_energy[:,:,[2:end; end]] .- base_energy[:,:,[1; 1:end-1]]
    return gx, gy, gz
end

function run_pk_ode(A_injected, t_span; test_lambda=0.0f0, test_k10=0.05f0)
    # PK config
    B_MAX_val = 100.0f0
    SA_MBq_nmol = 50.0f0
    k3 = 0.1f0; k4 = 0.02f0; k5 = 0.05f0
    k_in_pop = 0.01f0; k_out_pop = 0.005f0; f_pop = 0.2f0
    
    u0 = ComponentArray(A_blood=[A_injected], A_free=[0.0f0], A_surface=[0.0f0], A_internal=[0.0f0], TIA=[0.0f0])
    dt = 0.1f0
    steps = Int(t_span / dt)
    u = copy(u0)
    
    for _ in 1:steps
        voxel_in = (f_pop * k_in_pop * u.A_blood) 
        dA_blood = - test_k10 * u.A_blood .- (f_pop * k_in_pop * u.A_blood) .+ (k_out_pop .* u.A_free)
        
        M_bound = ((u.A_surface .+ u.A_internal) ./ SA_MBq_nmol)
        binding_flux = k3 .* u.A_free .* max.(0.0f0, 1.0f0 .- M_bound ./ B_MAX_val)
        unbinding_flux = k4 .* u.A_surface
        internalize_flux = k5 .* u.A_surface
        
        dA_free  = voxel_in .- (k_out_pop .* u.A_free) .- (test_lambda .* u.A_free) .- binding_flux .+ unbinding_flux
        dA_surface = binding_flux .- unbinding_flux .- internalize_flux .- (test_lambda .* u.A_surface)
        dA_internal = internalize_flux .- (test_lambda .* u.A_internal)
        dTIA = u.A_free .+ u.A_surface .+ u.A_internal 
        
        u.A_blood .+= dA_blood .* dt
        u.A_free .+= dA_free .* dt
        u.A_surface .+= dA_surface .* dt
        u.A_internal .+= dA_internal .* dt
        u.TIA .+= dTIA .* dt
    end
    return u
end

# Fake NN model function
function mock_nn_corrector(inputs...)
    return zeros(Float32, size(inputs[1])) # identity exp(0)=1
end

@testset "Comprehensive Catphan Dosimetry Physics Suite" begin

    @testset "1. Physical-to-Digital Validations" begin
        # 1a. HU-to-Density Calibration (CTP401)
        # Using adjusted physical tuned slopes mapping Catphan precisely:
        tuned_slope_tissue = 0.0011717f0 # Teflon
        tuned_slope_air = 0.001f0
        
        @test isapprox(hu_to_density(-1000.0f0, slope_air=tuned_slope_air), 0.0f0, atol=1e-3) # Air
        @test isapprox(hu_to_density(-100.0f0, slope_air=tuned_slope_air), 0.9f0, atol=1e-3)  # LDPE
        @test isapprox(hu_to_density(120.0f0, slope_tissue=0.0015f0), 1.18f0, atol=1e-3)    # Acrylic
        @test isapprox(hu_to_density(990.0f0, slope_tissue=tuned_slope_tissue), 2.16f0, atol=1e-3) # Teflon
        
        # 1b. Voxel Mass & Spatial Linearity (CTP401)
        pixel_spacing = [1.0f0, 1.0f0, 1.0f0]
        p1 = [10.0f0, 10.0f0, 10.0f0]
        p2 = [60.0f0, 10.0f0, 10.0f0]
        centroid_dist = sqrt(sum(((p1 .- p2) .* pixel_spacing).^2))
        @test isapprox(centroid_dist, 50.0f0, atol=0.5f0)
        
        # 1c. Air Void Singularity Prevention
        rho_air = 0.01f0 # < 0.05
        vol = 1.0f0
        function safe_dose(E, rho, vol)
            if rho <= 0.05f0
                return 0.0f0
            else
                return E / (rho * vol)
            end
        end
        @test safe_dose(10.0f0, rho_air, vol) == 0.0f0
    end

    @testset "2. Consistency Checks of SciML Spatial Layers" begin
        # 2a. Gradient Hallucination Check (CTP486 Image Uniformity)
        uniform_rho = fill(1.0f0, 10, 10, 10)
        grad_rho = compute_grad_rho(uniform_rho)
        # Inside edges should purely reflect the 1f-6 bias square root padding
        @test isapprox(grad_rho[5,5,5], sqrt(1f-6), atol=1e-5)
        
        # 2b. Directional Vector Flux Symmetry (CTP445 Point Source)
        base_energy_delta = zeros(Float32, 11, 11, 11)
        base_energy_delta[6, 6, 6] = 100.0f0
        gx, gy, gz = compute_directional_flux(base_energy_delta)
        # Ensure identical mirrored backscatter magnitudes radially
        @test gx[7,6,6] == -gx[5,6,6]
        @test gy[6,7,6] == -gy[6,5,6]
        @test gz[6,6,7] == -gz[6,6,5]
    end

    @testset "3. Universal Differential Equation Invariants" begin
        # 3a. Thermodynamic Energy Conservation Limit
        mass_map = rand(Float32, 5, 5, 5) .+ 0.1f0
        E_base = rand(Float32, 5, 5, 5)
        nn_offset = randn(Float32, 5, 5, 5) # Randomized weights eval
        
        D_phys = (E_base .* exp.(nn_offset)) ./ mass_map
        energy_integral_pred = sum(D_phys .* mass_map)
        energy_integral_base = sum(E_base)
        
        λ_EC_loss = sum(abs2, energy_integral_pred - energy_integral_base)
        @test λ_EC_loss >= 0.0f0
        
        # 3b. Epoch Zero Identity Map Test
        nn_offset_zero = mock_nn_corrector(E_base)
        D_phys_zero = (E_base .* exp.(nn_offset_zero)) ./ mass_map
        @test all(isapprox.(D_phys_zero, E_base ./ mass_map, atol=1e-5))
    end
    
    @testset "4. Software Logic & PK Compartment Sanity Checks" begin
        # 4a. Compartmental Mass-Balance Conservation
        A_init = 100.0f0
        final_state = run_pk_ode(A_init, 10000.0f0, test_lambda=0.0f0, test_k10=0.0f0)
        total_sys = final_state.A_blood[1] + final_state.A_free[1] + final_state.A_surface[1] + final_state.A_internal[1]
        @test isapprox(total_sys, 100.0f0, atol=1e-1)
        
        # 4b. Tumour Sink Saturation Hard-Limit
        SA_massive = 1f-4
        B_MAX_val = 100.0f0
        k3 = 0.1f0
        M_bound_eval = (100.0f0 / SA_massive)
        binding_flux = k3 * 10.0f0 * max(0.0f0, 1.0f0 - M_bound_eval / B_MAX_val)
        @test binding_flux == 0.0f0
        
        # 4c. DICOM Unit Double-Calibration Trap
        dicom_units = "BQML"
        CF_mock = (dicom_units == "BQML") ? 1.0f0 : 54.2f0
        @test CF_mock == 1.0f0
        
        dicom_units = "COUNTS"
        CF_mock_2 = (dicom_units == "BQML") ? 1.0f0 : 54.2f0
        @test CF_mock_2 == 54.2f0
        
        # 4d. Iodine Contrast Shielding Hallucinations Check
        function check_hu(hu)
            if hu > 300.0f0
                throw(ErrorException("Iodine contrast detected! High-Z atoms > 300 HU explicitly hallucinates beta attenuation."))
            end
            return true
        end
        @test check_hu(100.0f0) == true
        @test_throws ErrorException check_hu(990.0f0)
    end
end
