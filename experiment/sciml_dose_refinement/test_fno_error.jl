using Lux, LuxCUDA, NeuralOperators, Random, CUDA

function test_fno()
    C_in = 2
    C_out = 1
    modes = (16, 16, 16)
    width = 32
    target_size = (64, 64, 64) # smaller for test
    
    model = FourierNeuralOperator(
        chs=(C_in, width, width, 128, C_out),
        modes=modes,
        relu
    )
    
    device = gpu_device()
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    ps = ps |> device
    st = st |> device
    model_dev = model |> device
    
    x = randn(Float32, target_size..., C_in, 1) |> device
    
    println("Starting FNO forward pass...")
    y, st = model_dev(x, ps, st)
    println("Success! Output size: ", size(y))
end

test_fno()
