using Pkg

# We need both the MedImages.jl project (for data loading) and the experiment project (for ML deps).
# Strategy: activate the experiment project, then dev MedImages.jl into it so all deps are available.
Pkg.activate(@__DIR__)

# Add the parent MedImages.jl project as a development dependency
medimages_root = joinpath(@__DIR__, "..", "..")
Pkg.develop(path=medimages_root)
Pkg.instantiate()

using MedImages
include("lu_loader.jl")

using Lux
using NNlib
using Optimisers
using Zygote
using MLUtils
using Random
using Statistics
using NeuralOperators
using ComponentArrays
using DifferentialEquations
using SciMLSensitivity
using FFTW

# ── Models (C_in=2: SPECT Recon + CT) ──────────────────────────────────────

function create_pinn_model(; C_in=2, C_out=1)
    return Chain(
        Conv((3, 3, 3), C_in => 8, relu, pad=1),
        Conv((3, 3, 3), 8 => 16, relu, pad=1),
        Conv((3, 3, 3), 16 => 8, relu, pad=1),
        Conv((3, 3, 3), 8 => C_out, pad=1)
    )
end

function pinn_loss(model, ps, st, x, y)
    y_pred, st = model(x, ps, st)
    l_data = mean(abs2, y_pred .- y)

    # Physics constraint: total predicted dose ≈ total SPECT activity (channel 1)
    spect_sum = sum(x[:, :, :, 1:1, :]; dims=(1, 2, 3))
    pred_sum  = sum(y_pred; dims=(1, 2, 3))
    l_phys = mean(abs2, spect_sum .- pred_sum)

    return l_data + 0.1f0 * l_phys, st
end

function create_fno_model(; C_in=2, C_out=1, modes=(4, 4, 4), width=16)
    return FourierNeuralOperator(
        chs=(C_in, width, width, 128, C_out),
        modes=modes,
        relu
    )
end

function fno_loss(model, ps, st, x, y)
    y_pred, st = model(x, ps, st)
    return mean(abs2, y_pred .- y), st
end

function create_ude_neural_term(; width=4)
    # Takes [SPECT, CT, D] (3 channels) as input
    return Chain(
        Conv((3, 3, 3), 3 => width, relu, pad=1),
        Conv((3, 3, 3), width => 1, pad=1)
    )
end

function out_of_place_ude_loss(model, ps, st, x, y)
    # SPECT activity (A) is channel 1, CT is channel 2
    A_fixed  = x[:, :, :, 1:1, :]
    CT_fixed = x[:, :, :, 2:2, :]
    D_init   = A_fixed  # initial dose proxy

    function f(u, p, t)
        nn_input = cat(A_fixed, CT_fixed, u; dims=4)
        mech = A_fixed .* CT_fixed .* 0.01f0
        nt, _ = model(nn_input, p, st)
        return mech .+ nt
    end

    function integrate(u, p, t0, dt, steps)
        for i in 1:steps
            u = u .+ f(u, p, t0 + (i - 1) * dt) .* dt
        end
        return u
    end
    D_refined = integrate(D_init, ps, 0.0f0, 0.5f0, 2)

    return mean(abs2, D_refined .- y), st
end

# ── Training loop ───────────────────────────────────────────────────────────

function train_model!(model_name, model, loss_func, loader; epochs=5, lr=1e-2)
    println("\n=== Starting training for $model_name (REAL DATA) ===")
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    ps, st = Lux.setup(rng, model)
    opt = Adam(lr)
    opt_state = Optimisers.setup(opt, ps)

    for epoch in 1:epochs
        total_loss = 0.0
        batches = 0
        for (x, y) in loader
            loss_val, back = Zygote.pullback(p -> loss_func(model, p, st, x, y), ps)
            grads = back((1.0f0, nothing))[1]
            current_loss, st = loss_val
            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            total_loss += current_loss
            batches += 1
        end
        avg_loss = total_loss / batches
        println("Epoch $epoch/$epochs - Average Loss: $avg_loss")
    end
    println("=== Finished training for $model_name ===\n")
    return ps, st
end

# ── Main ────────────────────────────────────────────────────────────────────

# ── Main ────────────────────────────────────────────────────────────────────

function run_selected_experiment()
    # Default values or from command line
    model_type = length(ARGS) >= 1 ? ARGS[1] : "all"
    base_dir = length(ARGS) >= 2 ? ARGS[2] : (get(ENV, "LU_DATA_DIR", "/home/jm/project_ssd/MedImages.jl/test_data/dataset_Lu"))
    
    # Hyperparameters for full training (can be adjusted via Slurm)
    num_samples = parse(Int, get(ENV, "LU_NUM_SAMPLES", "100"))
    target_size = (64, 64, 64)
    batchsize = parse(Int, get(ENV, "LU_BATCH_SIZE", "4"))
    epochs = parse(Int, get(ENV, "LU_EPOCHS", "20"))

    println("--- Lu-177 SciML Training ---")
    println("Model Type: $model_type")
    println("Data Dir:   $base_dir")
    println("Samples:    $num_samples")
    println("Epochs:     $epochs")
    println("Batch Size: $batchsize")

    loader = create_lu_data_loader(base_dir; num_samples=num_samples, target_size=target_size, batchsize=batchsize)

    if model_type == "pinn" || model_type == "all"
        pinn_model = create_pinn_model()
        train_model!("PINN-style CNN", pinn_model, pinn_loss, loader; epochs=epochs)
    end

    if model_type == "fno" || model_type == "all"
        fno_model = create_fno_model()
        train_model!("FNO-style Spectral Proxy", fno_model, fno_loss, loader; epochs=epochs)
    end

    if model_type == "ude" || model_type == "all"
        ude_model = create_ude_neural_term()
        train_model!("UDE Hybrid Model", ude_model, out_of_place_ude_loss, loader; epochs=epochs)
    end
end

run_selected_experiment()
