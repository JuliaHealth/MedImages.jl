using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, Lux, Random, SciMLSensitivity, Optimisers, Zygote, Enzyme, ComponentArrays

function debug_enzyme_step1()
    println("--- Step 1: Simple 1D Neural ODE on CPU ---")
    rng = Random.default_rng()
    
    # 1. Very simple model: 1 -> 4 -> 1
    model = Chain(Dense(1 => 4, tanh), Dense(4 => 1))
    ps, st = Lux.setup(rng, model)
    
    # Convert ps to Flat vector for simplicity first, or just leave as NamedTuple. 
    # Let's use ComponentArrays since that's what SciML uses.
    θ = ComponentArray(ps)
    
    u0 = [1.0f0]
    tspan = (0.0f0, 1.0f0)
    
    function f(u, p, t)
        nn_out, _ = Lux.apply(model, u, p, st)
        return -u .+ nn_out
    end
    
    prob = ODEProblem(f, u0, tspan, θ)
    target = [0.5f0]
    
    function loss_fn(p)
        sol = solve(prob, Tsit5(), p=p, saveat=[1.0f0], sensealg=InterpolatingAdjoint(autojacvec=EnzymeVJP()))
        return sum((sol.u[end] .- target).^2)
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

debug_enzyme_step1()
