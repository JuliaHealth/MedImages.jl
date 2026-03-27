using Pkg
Pkg.activate("/home/user/MedImages.jl")

using DifferentialEquations, Lux, LuxCUDA, CUDA, BenchmarkTools, Random, ComponentArrays, SciMLSensitivity

# Architectures (Minimal for benchmarking)
function ResBlock(channels::Int)
    return SkipConnection(Chain(Conv((3, 3, 3), channels => channels, pad=1, relu), Conv((3, 3, 3), channels => channels, pad=1)), +)
end

function build_ude(width, depth)
    layers = Any[Parallel(+, Conv((3, 3, 3), 1 => width, pad=1, relu), Conv((3, 3, 3), 1 => width, pad=1, relu))]
    for _ in 1:depth; push!(layers, ResBlock(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

function build_cnn(width, depth)
    layers = Any[Conv((3, 3, 3), 3 => width, pad=1, relu)]
    for _ in 1:depth; push!(layers, ResBlock(width)); end
    push!(layers, Conv((3, 3, 3), width => 1, pad=1))
    return Chain(layers...)
end

function benchmark_all()
    dev = Lux.gpu_device(); rng = Random.default_rng()
    p_s = 32
    
    # --- UDE Benchmark ---
    m_ude = build_ude(32, 3); ps_u, st_ude = Lux.setup(rng, m_ude); θ_u = dev(ComponentArray(ps_u)); st_ude = dev(st_ude)
    A0 = dev(randn(Float32, p_s, p_s, p_s)); den = dev(randn(Float32, p_s, p_s, p_s))
    
    u0 = ComponentArray(A_blood=dev(Float32[1.0]), A_free=A0, A_bound=A0, DOSE=zero(A0))
    function f(u,p,t)
        A_t = u.A_free .+ u.A_bound
        in_nn = (reshape(A_t, p_s, p_s, p_s, 1, 1), reshape(den, p_s, p_s, p_s, 1, 1))
        nn_o, _ = Lux.apply(m_ude, in_nn, p, st_ude)
        return ComponentArray(A_blood=zero(u.A_blood), A_free=zero(u.A_free), A_bound=zero(u.A_bound), DOSE=reshape(nn_o, p_s, p_s, p_s))
    end
    prob = ODEProblem(f, u0, (0.0f0, 300.0f0), θ_u)
    
    println("Benchmarking UDE Inference (300h integration)...")
    t_ude = @belapsed CUDA.@sync solve($prob, VCABM(), saveat=[300.0f0], reltol=1f-3, abstol=1f-3)
    println("  UDE Time: ", round(t_ude, digits=4), " s")

    # --- CNN Benchmark ---
    m_cnn = build_cnn(64, 6); ps_c, st_cnn = Lux.setup(rng, m_cnn); θ_c = dev(ComponentArray(ps_c)); st_cnn = dev(st_cnn)
    input_cnn = dev(randn(Float32, p_s, p_s, p_s, 3, 1))
    
    println("Benchmarking CNN Inference (Forward Pass)...")
    t_cnn = @belapsed CUDA.@sync Lux.apply($m_cnn, $input_cnn, $θ_c, $st_cnn)
    println("  CNN Time: ", round(t_cnn, digits=4), " s")
end

benchmark_all()
