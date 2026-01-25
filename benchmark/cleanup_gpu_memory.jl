#!/usr/bin/env julia
"""
GPU Memory Cleanup Script
Clears GPU memory and initializes CUDA environment before benchmark
"""

println("="^80)
println("GPU Memory Cleanup and Initialization")
println("="^80)

# Clear any existing loaded packages
empty!(Base.loaded_modules)
GC.gc(true)  # Aggressive garbage collection

# Import CUDA and clear device
try
    using CUDA
    println("\nCUDA imported successfully")

    # Get current device
    device = CUDA.device()
    println("Current device: $(CUDA.name(device))")

    # Show initial memory
    mem = CUDA.MemoryInfo()
    println("\nInitial GPU Memory:")
    println("  Available: $(mem.free / 1024^3) GB / $(mem.total / 1024^3) GB")

    # Clear all GPU arrays and memory pool
    CUDA.Memory.release_pool()
    println("\nMemory pool released")

    GC.gc(true)
    sleep(1)

    # Show final memory
    mem = CUDA.MemoryInfo()
    println("\nFinal GPU Memory:")
    println("  Available: $(mem.free / 1024^3) GB / $(mem.total / 1024^3) GB")
    println("  Used: $(( mem.total - mem.free) / 1024^3) GB")

    usage_percent = ((mem.total - mem.free) / mem.total) * 100
    println("  Usage: $(round(usage_percent, digits=2))%")

    if usage_percent > 10
        println("\n⚠ WARNING: GPU still has high memory usage")
        println("Attempting additional cleanup...")

        # Try to empty CUDA caches more aggressively
        CUDA.reclaim(reset_all=true)
        GC.gc(true)
        sleep(2)

        mem = CUDA.MemoryInfo()
        println("\nAfter aggressive cleanup:")
        println("  Available: $(mem.free / 1024^3) GB / $(mem.total / 1024^3) GB")
        usage_percent = ((mem.total - mem.free) / mem.total) * 100
        println("  Usage: $(round(usage_percent, digits=2))%")
    end

    if usage_percent < 10
        println("\n✓ GPU memory cleared successfully!")
    else
        println("\n⚠ GPU memory usage still high, proceeding with benchmark anyway...")
    end

catch e
    println("Error during cleanup: $e")
    exit(1)
end

println("\n"^1)
println("Cleanup complete. Ready for benchmark.")
println("="^80)
