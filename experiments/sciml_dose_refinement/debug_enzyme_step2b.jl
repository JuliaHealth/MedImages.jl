using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, Lux, Random, SciMLSensitivity, Optimisers, Zygote, Enzyme

function build_minimal_cnn_3d()
    return Chain(Conv((3, 3, 3), 1 => 2, tanh, pad=1), Conv((3, 3, 3), 2 => 1, pad=1))
end

function debug_enzyme_step2b()
    println("--- Step 2b: Small 3D Neural ODE on CPU (Optimisers.destructure) ---")
    rng = Random.default_rng()
    
    p_s = 4 # Small 3D patch
    
    m_ude = build_minimal_cnn_3d()
    ps, st = Lux.setup(rng, m_ude)
    θ, re = Optimisers.destructure(ps)
    
    A0 = randn(Float32, p_s, p_s, p_s)
    target = randn(Float32, p_s, p_s, p_s)
    
    # We need to augment u0 to hold A_free and DOSE
    u0 = vcat(vec(A0), zeros(Float32, p_s^3))
    
    function f(u, p, t)
        # u is [A_free... ; DOSE...]
        A_free = reshape(u[1:p_s^3], p_s, p_s, p_s)
        in_nn = reshape(A_free, p_s, p_s, p_s, 1, 1)
        nn_o, _ = Lux.apply(m_ude, in_nn, re(p), st)
        
        dA_free = -0.01f0 .* A_free
        dD = 0.05f0 .* A_free .+ reshape(nn_o, p_s, p_s, p_s)
        
        return vcat(vec(dA_free), vec(dD))
    end
    
    prob = ODEProblem(f, u0, (0.0f0, 2.0f0), θ)
    
    function loss_fn(p)
        sol = solve(prob, Tsit5(), p=p, saveat=[2.0f0], sensealg=InterpolatingAdjoint(autojacvec=EnzymeVJP()))
        final_dose = sol.u[end][p_s^3+1:end]
        return sum((final_dose .- vec(target)).^2)
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

debug_enzyme_step2b()
