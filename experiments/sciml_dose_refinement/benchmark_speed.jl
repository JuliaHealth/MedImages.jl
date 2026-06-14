using Pkg
Pkg.activate(".")

using DifferentialEquations, Lux, LuxCUDA, CUDA, BenchmarkTools, Random, ComponentArrays, SciMLSensitivity

function ResBlockNorm(channels::Int)
    return SkipConnection(Chain(Conv((3, 3, 3), channels => channels, pad=1), GroupNorm(channels, 8, relu), Conv((3, 3, 3), channels => channels, pad=1), GroupNorm(channels, 8)), +)
end

function build_ude_improved(width::Int, depth::Int)
    layers = Any[Parallel(+, Conv((3, 3, 3), 1 => width, pad=1, relu), Conv((3, 3, 3), 1 => width, pad=1, relu), Conv((3, 3, 3), 1 => width, pad=1, relu))]
    for _ in 1:depth; push!(layers, ResBlockNorm(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

function build_cnn_improved(width::Int, depth::Int)
    layers = Any[Conv((3, 3, 3), 3 => width, pad=1, relu)]
    for _ in 1:depth; push!(layers, ResBlockNorm(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

function benchmark_all()
    dev = Lux.gpu_device(); rng = Random.default_rng()
    p_s = 32 # Use 32x32x32 to avoid OOM on fragmented GPU
    
    # --- UDE Benchmark ---
    m_ude = build_ude_improved(32, 3); ps_u, st_ude = Lux.setup(rng, m_ude); θ_u = dev(ComponentArray(ps_u)); st_ude = dev(st_ude)
    A0 = dev(randn(Float32, p_s, p_s, p_s)); den = dev(randn(Float32, p_s, p_s, p_s)); den_grad = dev(randn(Float32, p_s, p_s, p_s))
    
    u0 = ComponentArray(A_blood=dev(Float32[1.0]), A_free=A0, A_bound=A0, DOSE=zero(A0))
    function f(u,p,t)
        A_t = u.A_free .+ u.A_bound
        in_nn = (reshape(A_t, p_s, p_s, p_s, 1, 1), reshape(den, p_s, p_s, p_s, 1, 1), reshape(den_grad, p_s, p_s, p_s, 1, 1))
        nn_o, _ = Lux.apply(m_ude, in_nn, p, st_ude)
        return ComponentArray(A_blood=zero(u.A_blood), A_free=zero(u.A_free), A_bound=zero(u.A_bound), DOSE=reshape(nn_o, p_s, p_s, p_s))
    end
    prob = ODEProblem(f, u0, (0.0f0, 300.0f0), θ_u)
    
    println("Benchmarking UDE Improved Inference ($p_s^3, 300h integration with Euler)...")
    # Using Euler as in final inference
    t_ude = @belapsed CUDA.@sync solve($prob, Euler(), dt=15.0f0, saveat=[300.0f0])
    println("  UDE Time: ", round(t_ude, digits=4), " s")

    # --- CNN Benchmark ---
    m_cnn = build_cnn_improved(32, 3); ps_c, st_cnn = Lux.setup(rng, m_cnn); θ_c = dev(ComponentArray(ps_c)); st_cnn = dev(st_cnn)
    input_cnn = dev(randn(Float32, p_s, p_s, p_s, 3, 1))
    
    println("Benchmarking CNN Improved Inference ($p_s^3 Forward Pass)...")
    t_cnn = @belapsed CUDA.@sync Lux.apply($m_cnn, $input_cnn, $θ_c, $st_cnn)
    println("  CNN Time: ", round(t_cnn, digits=4), " s")
end

benchmark_all()
