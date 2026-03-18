using Pkg
Pkg.activate(@__DIR__)

using MedImages
# using MedImages.Load_and_save: create_nii_from_medimage, update_voxel_data
include("lu_loader.jl")

using Lux, LuxCUDA
using NNlib, Optimisers, Zygote, MLUtils, Random, Statistics
using NeuralOperators, ComponentArrays, DifferentialEquations
using FFTW, CUDA

# ── GPU Setup ──────────────────────────────────────────────────────────────
const device = gpu_device()
const cpu = cpu_device()

# ── Models ─────────────────────────────────────────────────────────────────
function create_pinn_model(; C_in=2, C_out=1)
    return Chain(
        Conv((3, 3, 3), C_in => 16, relu, pad=1),
        Conv((3, 3, 3), 16 => 32, relu, pad=1),
        Conv((3, 3, 3), 32 => 16, relu, pad=1),
        Conv((3, 3, 3), 16 => C_out, pad=1)
    )
end

function pinn_loss(model, ps, st, x, y)
    y_pred, st = model(x, ps, st)
    l_data = mean(abs2, y_pred .- y)
    spect_sum = sum(x[:, :, :, 1:1, :]; dims=(1, 2, 3))
    pred_sum  = sum(y_pred; dims=(1, 2, 3))
    l_phys = mean(abs2, spect_sum .- pred_sum)
    return l_data + 0.1f0 * l_phys, st
end

function create_fno_model(; C_in=2, C_out=1, modes=(16, 16, 16), width=32)
    return FourierNeuralOperator(
        chs=(C_in, width, width, 64, C_out),
        modes=modes,
        relu
    )
end

function fno_loss(model, ps, st, x, y)
    y_pred, st = model(x, ps, st)
    return mean(abs2, y_pred .- y), st
end

function create_ude_neural_term(; width=16)
    return Chain(
        Conv((3, 3, 3), 3 => width, relu, pad=1),
        Conv((3, 3, 3), width => width, relu, pad=1),
        Conv((3, 3, 3), width => 1, pad=1)
    )
end

function out_of_place_ude_loss(model, ps, st, x, y)
    A_fixed  = x[:, :, :, 1:1, :]
    CT_fixed = x[:, :, :, 2:2, :]
    D_init   = A_fixed

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
    D_refined = integrate(D_init, ps, 0.0f0, 0.2f0, 5)
    return mean(abs2, D_refined .- y), st
end

function forward_ude(model, ps, st, x)
    A_fixed  = x[:, :, :, 1:1, :]
    CT_fixed = x[:, :, :, 2:2, :]
    D_init   = A_fixed

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
    return integrate(D_init, ps, 0.0f0, 0.2f0, 5)
end

# ── Training loop ───────────────────────────────────────────────────────────
function train_model!(model_name, model, loss_func, loader; epochs=5, lr=1e-2)
    println("\n=== Starting training for $model_name ===")
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    ps, st = Lux.setup(rng, model)
    ps = ps |> device
    st = st |> device
    model_dev = model |> device
    opt = Adam(lr)
    opt_state = Optimisers.setup(opt, ps) |> device

    for epoch in 1:epochs
        for (x, y) in loader
            x_dev = x |> device
            y_dev = y |> device
            loss_val, back = Zygote.pullback(p -> loss_func(model_dev, p, st, x_dev, y_dev), ps)
            grads = back((1.0f0, nothing))[1]
            current_loss, st = loss_val
            opt_state, ps = Optimisers.update(opt_state, ps, grads)
        end
    end
    println("=== Finished training for $model_name ===")
    return ps |> cpu, st |> cpu
end

# ── Metrics ─────────────────────────────────────────────────────────────────
function calc_l1_loss(y_pred, y_true)
    return mean(abs.(y_pred .- y_true))
end

function calc_ncc(y_pred, y_true)
    yp = y_pred .- mean(y_pred)
    yt = y_true .- mean(y_true)
    num = sum(yp .* yt)
    den = sqrt(sum(yp.^2) * sum(yt.^2)) + 1e-8
    return num / den
end

# ── Main Inference & Eval ───────────────────────────────────────────────────
function run_inference_and_eval()
    data_dir = get(ENV, "LU_DATA_DIR", "/DATA")
    target_size = (64, 64, 64)
    num_samples = 2
    epochs = 15
    lr = 0.001

    println("Loading $num_samples samples for Inference from $data_dir...")
    loader = create_lu_data_loader(data_dir; num_samples=num_samples, target_size=target_size, batchsize=1)
    
    # Train models
    pinn_model = create_pinn_model()
    pinn_ps, pinn_st = train_model!("PINN", pinn_model, pinn_loss, loader; epochs=epochs, lr=lr)

    fno_model = create_fno_model()
    fno_ps, fno_st = train_model!("FNO", fno_model, fno_loss, loader; epochs=epochs, lr=lr)

    ude_model = create_ude_neural_term()
    ude_ps, ude_st = train_model!("UDE", ude_model, out_of_place_ude_loss, loader; epochs=epochs, lr=lr)

    # Output directory
    out_dir = joinpath(@__DIR__, "inference_results")
    mkpath(out_dir)

    # Run Inference
    println("\n=== Running Inference and Computing Metrics ===")
    metrics_log = open(joinpath(out_dir, "inference_metrics.txt"), "w")

    for (i, (x, y)) in enumerate(loader)
        println(metrics_log, "Patient $i Metrics:")
        println("Evaluating Patient $i...")
        
        # Ground Truth
        y_true = y[:, :, :, 1, 1]

        # 1. PINN Inference
        y_pred_pinn, _ = pinn_model(x, pinn_ps, pinn_st)
        y_pred_pinn = y_pred_pinn[:, :, :, 1, 1]
        pinn_l1 = calc_l1_loss(y_pred_pinn, y_true)
        pinn_ncc = calc_ncc(y_pred_pinn, y_true)
        println(metrics_log, "  PINN - L1: $pinn_l1 | Visual Similarity (NCC): $pinn_ncc")

        # 2. FNO Inference
        y_pred_fno, _ = fno_model(x, fno_ps, fno_st)
        y_pred_fno = y_pred_fno[:, :, :, 1, 1]
        fno_l1 = calc_l1_loss(y_pred_fno, y_true)
        fno_ncc = calc_ncc(y_pred_fno, y_true)
        println(metrics_log, "  FNO  - L1: $fno_l1 | Visual Similarity (NCC): $fno_ncc")

        # 3. UDE Inference
        y_pred_ude = forward_ude(ude_model, ude_ps, ude_st, x)
        y_pred_ude = y_pred_ude[:, :, :, 1, 1]
        ude_l1 = calc_l1_loss(y_pred_ude, y_true)
        ude_ncc = calc_ncc(y_pred_ude, y_true)
        println(metrics_log, "  UDE  - L1: $ude_l1 | Visual Similarity (NCC): $ude_ncc")

        # Save NIfTI files
        # We'll create a dummy MedImage to save
        origin = (0.0, 0.0, 0.0)
        spacing = (1.0, 1.0, 1.0)
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        
        function save_nii(data, filename)
            # Create a basic MedImage wrapper manually
            # we can use the constructor from Load_and_save / MedImage_data_struct
            voxel_f32 = Array{Float32, 3}(data)
            meta = MedImages.ITKIOWrapper.DataStructs.SpatialMetaData(
                origin, spacing, (64,64,64), direction
            )
            vd = MedImages.ITKIOWrapper.DataStructs.VoxelData(voxel_f32)
            MedImages.ITKIOWrapper.save_image(vd, meta, joinpath(out_dir, filename), false)
        end
        
        save_nii(y_true, "Patient_$(i)_Original_Dose.nii.gz")
        
        save_nii(y_pred_pinn, "Patient_$(i)_PINN_Inferred.nii.gz")
        save_nii(abs.(y_pred_pinn .- y_true), "Patient_$(i)_PINN_Diff.nii.gz")
        
        save_nii(y_pred_fno, "Patient_$(i)_FNO_Inferred.nii.gz")
        save_nii(abs.(y_pred_fno .- y_true), "Patient_$(i)_FNO_Diff.nii.gz")

        save_nii(y_pred_ude, "Patient_$(i)_UDE_Inferred.nii.gz")
        save_nii(abs.(y_pred_ude .- y_true), "Patient_$(i)_UDE_Diff.nii.gz")
    end
    
    close(metrics_log)
    println("=== Evaluation Complete. Results written to $(out_dir) ===")
end

run_inference_and_eval()
