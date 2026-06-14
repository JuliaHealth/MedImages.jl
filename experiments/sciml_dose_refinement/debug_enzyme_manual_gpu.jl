using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, CUDA, Enzyme, Zygote, SciMLSensitivity

function debug_enzyme_manual()
    println("--- Manual Neural ODE on GPU with Enzyme ---")
    
    # Simple linear layer: u' = W * u
    W = CUDA.rand(Float32, 4, 4)
    u0 = CUDA.rand(Float32, 4)
    
    function f(u, p, t)
        return p * u
    end
    
    prob = ODEProblem(f, u0, (0.0f0, 1.0f0), W)
    
    function loss_fn(p)
        # Using BacksolveAdjoint as it requires less memory caching
        sol = solve(prob, Tsit5(), p=p, saveat=[1.0f0], sensealg=BacksolveAdjoint(autojacvec=EnzymeVJP()))
        return sum(sol.u[end].^2)
    end
    
    println("Testing forward pass...")
    l = loss_fn(W)
    println("Loss: ", l)
    
    println("Testing Enzyme VJP backward pass on GPU (Manual MatMul)...")
    try
        loss, gs = Zygote.withgradient(loss_fn, W)
        println("Enzyme Loss: ", loss)
        println("Gradients computed successfully! Norm: ", sum(abs2, gs[1]))
    catch e
        println("Enzyme failed:")
        showerror(stdout, e, catch_backtrace())
    end
end

debug_enzyme_manual()
