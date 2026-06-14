using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, Lux, LuxCUDA, CUDA, Random, ComponentArrays, SciMLSensitivity, Optimisers, Enzyme

function build_minimal_cnn()
    return Chain(Conv((3, 3, 3), 1 => 4, relu, pad=1), Conv((3, 3, 3), 4 => 1, pad=1))
end

function debug_enzyme_minimal()
    dev = Lux.cpu_device()
    rng = Random.default_rng()
    p_s = 8 # Very small patch size to ensure quick compilation
    
    m_ude = build_minimal_cnn()
    ps, st = Lux.setup(rng, m_ude)
    θ = dev(ComponentArray(ps))
    st = dev(st)
    
    A0 = dev(randn(Float32, p_s, p_s, p_s))
    target = dev(randn(Float32, p_s, p_s, p_s))
    
    u0 = ComponentArray(A_free=A0, DOSE=zero(A0))
    
    function f(u, p, t)
        # Reshape for CNN: (W, H, D, C, N)
        in_nn = reshape(u.A_free, p_s, p_s, p_s, 1, 1)
        nn_o, _ = Lux.apply(m_ude, in_nn, p, st)
        
        # Simple dynamics
        dA_free = -0.01f0 .* u.A_free
        dD = relu.(0.05f0 .* u.A_free .+ reshape(nn_o, p_s, p_s, p_s))
        
        return ComponentArray(A_free=dA_free, DOSE=dD)
    end
    
    prob = ODEProblem(f, u0, (0.0f0, 10.0f0), θ)
    
    println("Testing ZygoteVJP fallback for reference...")
    using Zygote
    loss_z, gs_z = Zygote.withgradient(θ) do p
        sol = solve(ODEProblem(f, u0, (0.0f0, 10.0f0), p), Tsit5(), saveat=[10.0f0], reltol=1e-1, abstol=1e-1, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
        sum((sol.u[end].DOSE .- target).^2)
    end
    println("Zygote Loss: ", loss_z)
    
    println("\nTesting EnzymeVJP...")
    try
        loss_e, gs_e = Zygote.withgradient(θ) do p
            sol = solve(ODEProblem(f, u0, (0.0f0, 10.0f0), p), Tsit5(), saveat=[10.0f0], reltol=1e-1, abstol=1e-1, sensealg=InterpolatingAdjoint(autojacvec=EnzymeVJP()))
            sum((sol.u[end].DOSE .- target).^2)
        end
        println("Enzyme Loss: ", loss_e)
        println("Success! Minimal setup works with Enzyme.")
    catch e
        println("Enzyme failed:")
        showerror(stdout, e, catch_backtrace())
    end
end

debug_enzyme_minimal()