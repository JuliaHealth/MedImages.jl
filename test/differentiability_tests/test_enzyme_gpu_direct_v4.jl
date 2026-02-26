using Enzyme
using KernelAbstractions
using CUDA
using Test

if !CUDA.functional()
    exit(0)
end

# Simple function to differentiate
@inline function simple_add(x::Float32, y::Float32)
    return x * y
end

@kernel function simple_grad_kernel(d_x, d_y, @Const(x), @Const(y))
    I = @index(Global)
    
    # Simple deferred autodiff
    # We want to differentiate simple_add(x[I], y[I]) w.r.t x and y
    # Seed is 1.0f0
    Enzyme.autodiff_deferred(
        Reverse,
        simple_add,
        Active,
        Active(x[I]),
        Active(y[I])
    )
    # Note: Enzyme.autodiff_deferred returns the gradients as a tuple if arguments are Active
    # But how to store them?
    # Usually: grads = Enzyme.autodiff_deferred(...)
    # d_x[I] = grads[1]
end

function test_v4()
    x = CuArray([2.0f0, 3.0f0])
    y = CuArray([4.0f0, 5.0f0])
    d_x = CuArray([0.0f0, 0.0f0])
    d_y = CuArray([0.0f0, 0.0f0])
    
    # Wait, simple_grad_kernel needs to be defined correctly to store results
end

# Correct pattern for device-side AD with Enzyme:
@inline function simple_add_mut(res::CuDeviceArray{Float32,1}, x::Float32, y::Float32, I::Int)
    res[I] = x * y
    return nothing
end

@kernel function simple_grad_kernel_v5(d_res, d_x, d_y, @Const(res), @Const(x), @Const(y))
    I = @index(Global)
    
    Enzyme.autodiff_deferred(
        Reverse,
        simple_add_mut,
        Const,
        Duplicated(res, d_res),
        Active(x[I]),
        Active(y[I]),
        Const(I)
    )
end

function test_v5()
    x = CuArray([2.0f0, 3.0f0])
    y = CuArray([4.0f0, 5.0f0])
    res = CuArray([8.0f0, 15.0f0])
    d_res = CuArray([1.0f0, 1.0f0])
    d_x = CuArray([0.0f0, 0.0f0])
    d_y = CuArray([0.0f0, 0.0f0])
    
    println("Launching V5...")
    backend = CUDABackend()
    k = simple_grad_kernel_v5(backend)
    # The Active results are returned from autodiff_deferred
    # But wait, if they are Active, they are returned.
    # If they are Duplicated, they are mutated.
end

# Actually, the most reliable way to differentiate a big kernel on GPU with Enzyme 
# is to NOT use KernelAbstractions for the gradient kernel if it's too complex, 
# OR to use the one-kernel-per-thread-logic pattern.

println("Trying V5 logic...")
# (I'll just try to compile a simple one)
