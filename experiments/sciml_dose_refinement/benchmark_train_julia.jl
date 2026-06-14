using Pkg
Pkg.activate(".")

using DifferentialEquations, Lux, LuxCUDA, CUDA, BenchmarkTools, Random, ComponentArrays, SciMLSensitivity, Optimisers, Zygote, Enzyme

function ResBlockNorm(channels::Int)
    return SkipConnection(Chain(Conv((3, 3, 3), channels => channels, pad=1), GroupNorm(channels, 8, relu), Conv((3, 3, 3), channels => channels, pad=1), GroupNorm(channels, 8)), +)
end

function build_ude_improved(width::Int, depth::Int)
    layers = Any[Parallel(+, Conv((3, 3, 3), 1 => width, pad=1, relu), Conv((3, 3, 3), 1 => width, pad=1, relu), Conv((3, 3, 3), 1 => width, pad=1, relu))]
    for _ in 1:depth; push!(layers, ResBlockNorm(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

function benchmark_julia_train()
    dev = Lux.gpu_device(); rng = Random.default_rng()
    p_s = 32
    
    CUDA.reclaim()
    
    m_ude = build_ude_improved(32, 3); ps_u, st_ude = Lux.setup(rng, m_ude); θ_u = dev(ComponentArray(ps_u)); st_ude = dev(st_ude)
    A0 = dev(randn(Float32, p_s, p_s, p_s)); den = dev(randn(Float32, p_s, p_s, p_s)); den_grad = dev(randn(Float32, p_s, p_s, p_s))
    target = dev(randn(Float32, p_s, p_s, p_s))
    
    u0 = ComponentArray(A_blood=dev(Float32[1.0]), A_free=A0, A_bound=A0, DOSE=zero(A0))
    vol_p = 1.0f0
    
    function f(u,p,t)
        A_t = u.A_free .+ u.A_bound
        A_t_std = (A_t .- mean(A_t)) ./ (std(A_t) + 1f-6)
        in_nn = (reshape(A_t_std, p_s, p_s, p_s, 1, 1), reshape(den, p_s, p_s, p_s, 1, 1), reshape(den_grad, p_s, p_s, p_s, 1, 1))
        nn_o, _ = Lux.apply(m_ude, in_nn, p, st_ude)
        dD_phys = (A_t .* 0.08478f0) ./ (vol_p .* den .+ 1f-4)
        dD = softplus.(dD_phys .+ reshape(nn_o, p_s, p_s, p_s))
        return ComponentArray(A_blood=zero(u.A_blood), A_free=zero(u.A_free), A_bound=zero(u.A_bound), DOSE=dD)
    end
    
    opt_state = Optimisers.setup(Optimisers.Adam(1f-4), θ_u)
    
    println("Benchmarking Julia SciML Training Step...")
    # Warmup
    for _ in 1:2
        loss, gs = Zygote.withgradient(θ_u) do p
            prob = ODEProblem(f, u0, (0.0f0, 300.0f0), p)
            sol = solve(prob, Euler(), dt=15.0f0, saveat=[300.0f0], sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()))
            sum((sol.u[end].DOSE .- target).^2) / p_s^3
        end
        Optimisers.update!(opt_state, θ_u, gs[1])
    end
    CUDA.reclaim()
    
    t0 = time()
    n_iters = 5
    for _ in 1:n_iters
        loss, gs = Zygote.withgradient(θ_u) do p
            prob = ODEProblem(f, u0, (0.0f0, 300.0f0), p)
            sol = solve(prob, Euler(), dt=15.0f0, saveat=[300.0f0], sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()))
            sum((sol.u[end].DOSE .- target).^2) / p_s^3
        end
        Optimisers.update!(opt_state, θ_u, gs[1])
    end
    CUDA.@sync nothing
    t1 = time()
    
    println("  Julia Training Step Time: ", (t1 - t0) / n_iters, " s")
end

benchmark_julia_train()