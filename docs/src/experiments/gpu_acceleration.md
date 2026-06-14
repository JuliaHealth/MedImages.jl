# Challenge 2: Speed (GPU Acceleration)

**Objective:** Assess raw execution speed of fundamental spatial transformations across different backends.

## The "Two-Language" Barrier

Researchers often prototype in Python but rely on compiled C++ backends (like ITK) for speed. Julia circumvents this by compiling directly to efficient machine code via LLVM JIT, allowing operations to execute natively on CPUs and GPUs.

## Detailed Code Walkthrough

We benchmarked $256 \times 256 \times 128$ volumetric arrays comparing MedImages.jl on CPU and GPU (via `KernelAbstractions.jl`). Let's look at how the benchmarking suite is structured to isolate the performance of a fused affine transformation versus sequential operations.

### Fused Affine Benchmark Pipeline

```julia
# File: experiments/benchmark/gpu_benchmarks.jl

function benchmark_fused_affine(image::MedImage, backend::String, device_id::Int=0)
    # ... (Setup and variable initialization) ...
    dims = size(image.voxel_data)
    
    # 1. Define transformation parameters
    translation = (10.0, 10.0, 10.0)
    rotation_angle = 45.0 # deg around Z
    scale = (1.2, 1.2, 1.2)
    rotation = (0.0, 0.0, rotation_angle) 
    
    # 2. Create the unified transformation matrix
    affine_matrix = MedImages.create_affine_matrix(
        translation=translation, rotation=rotation, scale=scale, shear=(0.0, 0.0, 0.0)
    )
    
    # 3. Create a Batched image structure (required for affine_transform_mi)
    images_list = [image]
    batch_input = MedImages.create_batched_medimage(images_list)
    
    # 4. Transfer to GPU if testing CUDA backend
    if backend == "CUDA"
        CUDA.device!(device_id)
        batch_input.voxel_data = CuArray(batch_input.voxel_data)
    end

    affine_matrices = [affine_matrix]

    # 5. Warmup to compile the Julia GPU kernel
    MedImages.affine_transform_mi(batch_input, affine_matrices, MedImages.Linear_en)
    if backend == "CUDA"
        CUDA.@sync nothing # Wait for GPU to finish
    end

    # 6. Benchmark execution using BenchmarkTools.jl
    b = @benchmark begin
        result = MedImages.affine_transform_mi($batch_input, $affine_matrices, MedImages.Linear_en)
        if $backend == "CUDA"
            CUDA.@sync nothing
        end
    end samples=BENCHMARK_SAMPLES seconds=BENCHMARK_SECONDS

    # ... (Calculate throughput and return results) ...
end
```

### Line-by-Line Breakdown:

1. **Lines 8-11:** Define the exact spatial transformation: a 10-voxel translation, a 45-degree rotation around the Z-axis, and a 1.2x scale factor.
2. **Lines 14-17 (`create_affine_matrix`):** Instead of applying these sequentially, `MedImages.jl` multiplies these distinct operations into a single $4 \times 4$ homogeneous affine matrix.
3. **Lines 20-21 (`create_batched_medimage`):** The `affine_transform_mi` function expects a 4D batched tensor. We wrap our single image in a list and create a batch of size 1. This encapsulates the spatial metadata into a vectorized format.
4. **Lines 24-27 (GPU Transfer):** If the backend is "CUDA", the `voxel_data` array is explicitly converted to a `CuArray`. In Julia, dispatch is handled by types; because it's a `CuArray`, `KernelAbstractions.jl` will automatically compile and dispatch the downstream interpolation code to the GPU.
5. **Lines 32-35 (Warmup):** Julia is a Just-In-Time (JIT) compiled language. The first time `affine_transform_mi` is called with a `CuArray`, LLVM compiles the kernel to PTX code. We run a warmup pass and explicitly synchronize the GPU (`CUDA.@sync nothing`) to prevent this compilation time from polluting the benchmark.
6. **Lines 38-43 (`@benchmark`):** `BenchmarkTools.jl` repeatedly executes the block to get statistically significant results. The `$` interpolates the variables into the benchmarking expression to avoid global variable overhead.

## Results

MedImages.jl on GPU achieved massive speedups due to `KernelAbstractions.jl`:

*   **Resampling (Nearest Neighbor):** 115x speedup vs MedImages CPU.
*   **Orientation Changes:** 71x speedup vs MedImages CPU.
*   **Fused Affine Transformation:** 135x speedup vs MedImages CPU.

For the 100-subject pipeline resampled to a $512^3$ grid, per-subject GPU kernel times were strictly under 200ms.

![GPU Acceleration Benchmarks](viz/gpu_acceleration.png)
*Figure 1: Performance speedup factor achieved by natively compiling Julia code to PTX kernels vs standard CPU execution.*
