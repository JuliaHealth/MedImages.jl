using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, Lux, LuxCUDA, CUDA, Random, SciMLSensitivity, Optimisers, Zygote, Enzyme

function debug_enzyme_step4b()
    println("--- Step 4b: Simple Dense NN on GPU (Optimisers.destructure) ---")
    dev = Lux.gpu_device()
    rng = Random.default_rng()
    
    # 1. Simple Dense model
    model = Chain(Dense(1 => 4, tanh), Dense(4 => 1))
    ps, st = Lux.setup(rng, model)
    θ_cpu, re = Optimisers.destructure(ps)
    θ = dev(θ_cpu)
    st = dev(st)
    
    u0 = dev(Float32[1.0])
    target = dev(Float32[0.5])
    tspan = (0.0f0, 1.0f0)
    
    function f(u, p, t)
        nn_out, _ = Lux.apply(model, u, re(p), st)
        return -u .+ nn_out
    end
    
    prob = ODEProblem(f, u0, tspan, θ)
    
    function loss_fn(p)
        sol = solve(prob, Tsit5(), p=p, saveat=[1.0f0], sensealg=InterpolatingAdjoint(autojacvec=EnzymeVJP()))
        return sum((sol.u[end] .- target).^2)
    end
    
    println("Testing forward pass...")
    l = loss_fn(θ)
    println("Loss: ", l)
    
    println("Testing Enzyme VJP backward pass on GPU (Dense)...")
    try
        loss, gs = Zygote.withgradient(loss_fn, θ)
        println("Enzyme Loss: ", loss)
        println("Gradients computed successfully! Norm: ", sum(abs2, gs[1]))
    catch e
        println("Enzyme failed:")
        showerror(stdout, e, catch_backtrace())
    end
end

debug_enzyme_step4b()
