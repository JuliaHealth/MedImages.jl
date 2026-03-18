using MedImages
using MedImages.Utils
using MedImages.MedImage_data_struct
using CUDA
using ChainRulesCore
using Test
using Zygote

function test_differentiability_single()
    println("Testing differentiability (Single Subject)...")
    
    # 1. Setup small data for testing gradients
    sz = (16, 16, 16)
    src = rand(Float32, sz...)
    
    M = Float32[
        1.1  0.0  0.0  0.0;
        0.0  1.1  0.0  0.0;
        0.0  0.0  1.1  0.0;
        0.1  0.1  0.1  1.0
    ]
    
    target_size = (12, 12, 12)
    interpolator = MedImage_data_struct.Linear_en
    
    function loss(src, M)
        out = interpolate_fused_affine(src, M, target_size, interpolator)
        return sum(out .^ 2)
    end
    
    println("  Evaluating CPU gradients...")
    g_src_zy, g_M_zy = Zygote.gradient(loss, src, M)
    
    @test g_src_zy !== nothing
    @test g_M_zy !== nothing
    
    if CUDA.functional()
        println("  Evaluating GPU gradients...")
        src_gpu = CuArray(src)
        M_gpu = CuArray(M)
        
        g_src_gpu, g_M_gpu = Zygote.gradient((s, m) -> sum(interpolate_fused_affine(s, m, target_size, interpolator) .^ 2), src_gpu, M_gpu)
        
        diff_src = sum(abs.(Array(g_src_gpu) .- g_src_zy)) / length(g_src_zy)
        diff_M = sum(abs.(Array(g_M_gpu) .- g_M_zy)) / length(g_M_zy)
        
        println("  GPU/CPU src grad diff: $diff_src")
        println("  GPU/CPU M grad diff: $diff_M")
        
        @test diff_src < 1e-4
        @test diff_M < 1e-4
    end
end

function test_differentiability_batched()
    println("Testing differentiability (Batched)...")
    
    batch_size = 2
    target_size = (12, 12, 12)
    interpolator = MedImage_data_struct.Linear_en
    
    M = zeros(Float32, 4, 4, batch_size)
    M[:, :, 1] .= Float32[
        1.1  0.0  0.0  0.0;
        0.0  1.1  0.0  0.0;
        0.0  0.0  1.1  0.0;
        0.1  0.1  0.1  1.0
    ]
    M[:, :, 2] .= Float32[
        0.9  0.1  0.0  0.0;
        -0.1 0.9  0.0  0.0;
        0.0  0.0  1.0  0.0;
        0.2  0.3  0.4  1.0
    ]
    
    src = rand(Float32, 16, 16, 16, batch_size)
    
    function loss_batched(src, M)
        out = interpolate_fused_affine(src, M, target_size, interpolator)
        return sum(out .^ 2)
    end
    
    println("  Evaluating Batched CPU gradients...")
    g_src_zy, g_M_zy = Zygote.gradient(loss_batched, src, M)
    
    if CUDA.functional()
        println("  Evaluating Batched GPU gradients...")
        src_gpu = CuArray(src)
        M_gpu = CuArray(M)
        
        g_src_gpu, g_M_gpu = Zygote.gradient((s, m) -> sum(interpolate_fused_affine(s, m, target_size, interpolator) .^ 2), src_gpu, M_gpu)
        
        @test g_src_gpu !== nothing
        @test g_M_gpu !== nothing
        
        if g_M_zy !== nothing
            diff_M = sum(abs.(Array(g_M_gpu) .- g_M_zy)) / length(g_M_zy)
            println("  Batched GPU/CPU M grad diff: $diff_M")
            if diff_M >= 1e-4
                for b in 1:batch_size
                    println("  Batch $b:")
                    println("    g_M_gpu[1, :, $b]: $(Array(g_M_gpu)[1, :, b])")
                    println("    g_M_zy[1, :, $b]:  $(g_M_zy[1, :, b])")
                end
            end
            @test diff_M < 1e-4
        else
            println("  Batched GPU gradients computed successfully (magnitude $(sum(abs.(g_M_gpu))))")
            @test sum(abs.(g_M_gpu)) > 0
        end
        
        println("  Success: Batched GPU gradients computed.")
    end
end

@testset "Differentiability Tests" begin
    test_differentiability_single()
    test_differentiability_batched()
end
