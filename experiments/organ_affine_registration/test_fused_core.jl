using MedImages
using MedImages.Load_and_save
using MedImages.Resample_to_target
using MedImages.MedImage_data_struct
using MedImages.Basic_transformations
using MedImages.Utils
using CUDA, LuxCUDA
using KernelAbstractions
using LinearAlgebra

# Include the logic from preprocess_fused.jl
include("/media/jm/hddData/projects_new/MedImages.jl/experiments/organ_affine_registration/preprocess_fused.jl")

function test_core()
    println("Testing Core Components...")
    
    # 1. Grid Generation
    gx, gy, gz = get_coords_grid_gpu()
    println("Grid generated. Sizes: ", size(gx))
    @assert size(gx) == (128, 128, 128)
    
    # 2. Matrix
    M = get_fused_preprocess_matrix((200, 200, 200), (1.0, 1.0, 1.0), (128, 128, 128), (1.5, 1.5, 1.5))
    println("Matrix M:\n", M)
    @assert size(M) == (4, 4)
    
    # 3. Fused Kernel Forward
    src = CUDA.rand(Float32, 64, 64, 64)
    M_gpu = CuArray(Float32.(Matrix(I, 4, 4)))
    res = interpolate_fused_affine(src, M_gpu, (32, 32, 32), Linear_en)
    println("Forward pass success. Result size: ", size(res))
    @assert size(res) == (32, 32, 32)
    
    # 4. Pullback (Zygote/Enzyme)
    using Zygote
    println("Testing Gradient...")
    loss_val, back = Zygote.pullback(x -> sum(interpolate_fused_affine(x, M_gpu, (16, 16, 16), Linear_en)), src)
    grads = back(1.0f0)
    println("Gradient computed. Success.")
    @assert grads[1] !== nothing
end

test_core()
