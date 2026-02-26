using Lux, Random, Optimisers, Zygote, Statistics
using KernelAbstractions
using MedImages
using MedImages.MedImage_data_struct
using Printf

# Ensure local imports work relative to this file
if !@isdefined(Preprocessing)
    include("src/preprocessing.jl")
end
if !@isdefined(RegistrationModel)
    include("src/model.jl")
end
if !@isdefined(FusedLoss)
    include("src/fused_loss.jl")
end

using .Preprocessing
using .RegistrationModel
using .FusedLoss

# --- Overfit Experiment ---
function run_overfit_experiment()
    rng = Random.default_rng()

    println("--- Starting Single Organ Overfit Experiment ---")

    # 1. Setup Data
    # 1 Organ, 1 Batch
    # Point at (5,5,5). Target at (10,5,5).
    # This requires Translation X ≈ 5.0.

    batch_size = 1
    num_organs = 1

    # Points Tensor: (3, 512, 1)
    # We set only first point valid.
    points = fill(-1.0f0, (3, 512, 1))
    points[1, 1, 1] = 5.0f0
    points[2, 1, 1] = 5.0f0
    points[3, 1, 1] = 5.0f0

    # Gold Standard Volume: (20, 20, 20, 1)
    # Target at (10, 5, 5).
    gold_vol = zeros(Float32, 20, 20, 20, 1)

    # Create a small gaussian-like blob around (10,5,5) to give gradient
    for x in 1:20, y in 1:20, z in 1:20
        dist = sqrt((x-10)^2 + (y-5)^2 + (z-5)^2)
        if dist < 3.0
            gold_vol[x,y,z,1] = max(0.0f0, 1.0f0 - dist/3.0f0)
        end
    end
    # Ensure center is 1.0
    gold_vol[10, 5, 5, 1] = 1.0f0

    # Metadata
    # Barycenter should be target (10,5,5) for Metric 1 to minimize?
    # Metric 1: distance to barycenter. If we want point to move to (10,5,5),
    # then the barycenter of the GOLD label (which is fixed) IS (10,5,5).
    # Radius: Let's say 2.0.

    meta = [OrganMetadata(1, (10.0f0, 5.0f0, 5.0f0), 2.0f0)]

    # 2. Model
    # Input X doesn't matter much as we overfit parameters for this specific X
    x_input = rand(Float32, 16, 16, 16, 2, 1)

    model = MultiScaleCNN(2, num_organs)
    ps, st = Lux.setup(rng, model)

    # 3. Optimizer
    # Use larger LR for quick convergence test
    opt = Optimisers.Adam(0.05)
    opt_state = Optimisers.setup(opt, ps)

    # 4. Training Loop

    function loss_function(p, state)
        params_pred, new_state = model(x_input, p, state)
        # Apply scaling to translation to make it converge faster?
        # Model output translation is raw linear.
        # Let's trust Adam.

        l = compute_organ_loss(points, params_pred, gold_vol, meta)
        return l, new_state
    end

    losses = Float32[]

    for epoch in 1:100
        (l, st), back = Zygote.pullback(p -> loss_function(p, st), ps)
        grads = back((1.0f0, nothing))[1]
        opt_state, ps = Optimisers.update(opt_state, ps, grads)

        push!(losses, l)
        if epoch % 10 == 0 || epoch == 1
            @printf("Epoch %3d: Loss = %.5f\n", epoch, l)
        end
    end

    initial_loss = losses[1]
    final_loss = losses[end]

    println("Initial Loss: $initial_loss")
    println("Final Loss:   $final_loss")

    if final_loss < initial_loss * 0.5
        println("SUCCESS: Loss dropped significantly.")
    else
        println("FAILURE: Loss did not drop enough.")
    end

    # Analyze Params
    params_final, _ = model(x_input, ps, st)
    tx = params_final[4, 1, 1]
    ty = params_final[5, 1, 1]
    tz = params_final[6, 1, 1]
    println("Final Translation: ($tx, $ty, $tz) | Expected X ~ 5.0")

end

if abspath(PROGRAM_FILE) == @__FILE__
    run_overfit_experiment()
end
