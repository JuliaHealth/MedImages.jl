using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, Lux, Random, SciMLSensitivity, Optimisers, Zygote, Enzyme, ComponentArrays

function build_minimal_cnn_3d()
    return Chain(Conv((3, 3, 3), 1 => 2, tanh, pad=1), Conv((3, 3, 3), 2 => 1, pad=1))
end

function debug_enzyme_step2()
    println("--- Step 2: Small 3D Neural ODE on CPU ---")
    rng = Random.default_rng()
    
    p_s = 4 # Small 3D patch
    
    m_ude = build_minimal_cnn_3d()
    ps, st = Lux.setup(rng, m_ude)
    θ = ComponentArray(ps)
    
    A0 = randn(Float32, p_s, p_s, p_s)
    target = randn(Float32, p_s, p_s, p_s)
    
    u0 = ComponentArray(A_free=A0, DOSE=zero(A0))
    
    function f(u, p, t)
        in_nn = reshape(u.A_free, p_s, p_s, p_s, 1, 1)
        nn_o, _ = Lux.apply(m_ude, in_nn, p, st)
        
        dA_free = -0.01f0 .* u.A_free
        dD = 0.05f0 .* u.A_free .+ reshape(nn_o, p_s, p_s, p_s)
        
        return ComponentArray(A_free=dA_free, DOSE=dD)
    end
    
    prob = ODEProblem(f, u0, (0.0f0, 2.0f0), θ)
    
    function loss_fn(p)
        sol = solve(prob, Tsit5(), p=p, saveat=[2.0f0], sensealg=InterpolatingAdjoint(autojacvec=EnzymeVJP()))
        return sum((sol.u[end].DOSE .- target).^2)
    end
    
    println("Testing forward pass...")
    l = loss_fn(θ)
    println("Loss: ", l)
    
    println("Testing Enzyme VJP backward pass...")
    try
        loss, gs = Zygote.withgradient(loss_fn, θ)
        println("Enzyme Loss: ", loss)
        println("Gradients computed successfully! Norm: ", sum(abs2, gs[1]))
    catch e
        println("Enzyme failed:")
        showerror(stdout, e, catch_backtrace())
    end
end

debug_enzyme_step2()
