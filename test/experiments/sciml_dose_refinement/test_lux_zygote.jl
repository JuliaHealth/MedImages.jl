using Pkg; Pkg.activate(".")
using Optimization, OptimizationOptimJL, Lux, Zygote, Random, ComponentArrays

function build_neural_transport_model()
    return Chain(
        Conv((3, 3, 3), 3 => 8, relu, pad=1),
        Conv((3, 3, 3), 8 => 16, relu, pad=1),
        Conv((3, 3, 3), 16 => 8, relu, pad=1),
        Conv((3, 3, 3), 8 => 1, pad=1)
    )
end

rng = Random.default_rng()
NN_model = build_neural_transport_model()
NN_params, NN_st = Lux.setup(rng, NN_model)
theta_flat = ComponentArray(NN_params)

x_dummy = rand(Float32, 16, 16, 16, 3, 1)
y_target = rand(Float32, 16, 16, 16)

function loss(theta, p)
    pred, _ = Lux.apply(NN_model, x_dummy, theta, NN_st)
    pred = dropdims(pred, dims=(4,5))
    return sum(abs2, pred .- y_target)
end

opt_f = OptimizationFunction(loss, Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_f, theta_flat)

println("Testing simple Lux Zygote optimization...")
try
    res = solve(opt_prob, BFGS(), maxiters=2)
    println("Success! Loss: ", res.objective)
catch e
    println("Error: ", e)
    Base.showerror(stdout, e, catch_backtrace())
end
