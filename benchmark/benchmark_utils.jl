"""
Benchmark Utilities
Helper functions for benchmarking, reporting, and visualization.
"""

using Printf
using DataFrames
using CSV
using Statistics
using Dates

# Optional imports - may not be available
CUDA_LOADED = false
try
    using CUDA
    global CUDA_LOADED = CUDA.functional()
catch
end

UNICODEPLOTS_LOADED = false
try
    using UnicodePlots
    global UNICODEPLOTS_LOADED = true
catch
end

# benchmark_config.jl is included by run_gpu_benchmarks.jl

"""
    save_results_csv(results::Vector{BenchmarkResult}, output_file::String)

Save benchmark results to CSV file.
"""
function save_results_csv(results::Vector, output_file::String)
    mkpath(dirname(output_file))

    # Convert to DataFrame
    df = DataFrame(
        name=[r.name for r in results],
        operation=[r.operation for r in results],
        backend=[r.backend for r in results],
        image_size=[r.image_size for r in results],
        time_mean_ms=[r.time_mean * 1000 for r in results],
        time_median_ms=[r.time_median * 1000 for r in results],
        time_std_ms=[r.time_std * 1000 for r in results],
        memory_mb=[r.memory_bytes / 1024^2 for r in results],
        throughput=[r.throughput for r in results],
        timestamp=[r.timestamp for r in results]
    )

    # Collect all parameter keys first
    all_keys = Set{String}()
    for r in results
        for (key, value) in r.parameters
            push!(all_keys, string(key))
        end
    end

    # Add parameter columns initialized with missing
    for key in all_keys
        col_name = Symbol("param_$key")
        df[!, col_name] = Vector{Union{Missing,Any}}(missing, nrow(df))
    end

    # Fill parameter values
    for (i, r) in enumerate(results)
        for (key, value) in r.parameters
            col_name = Symbol("param_$key")
            df[i, col_name] = value
        end
    end

    CSV.write(output_file, df)
    println("Results saved to: $output_file")
end

"""
    generate_markdown_report(results::Vector{BenchmarkResult}, output_file::String; memory_stats::Dict=Dict())

Generate a comprehensive markdown report from benchmark results.
"""
function generate_markdown_report(results::Vector, output_file::String; memory_stats::Dict=Dict())
    mkpath(dirname(output_file))

    open(output_file, "w") do io
        # Header
        println(io, "# MedImages.jl GPU Benchmark Report")
        println(io)
        println(io, "**Generated:** $(now())")
        println(io)

        # System info
        println(io, "## System Information")
        println(io)
        println(io, "- Julia version: $(VERSION)")
        println(io, "- Platform: $(Sys.MACHINE)")
        println(io, "- CPU: $(Sys.CPU_NAME) ($(Sys.CPU_THREADS) threads)")

        # GPU info
        if CUDA_LOADED
            try
                dev = CUDA.device()
                println(io, "- GPU: $(CUDA.name(dev))")
                println(io, "- CUDA version: $(CUDA.version())")
                println(io, "- GPU memory: $(CUDA.totalmem(dev) ÷ 1024^3) GB")
            catch
                println(io, "- GPU: Not available")
            end
        else
            println(io, "- GPU: Not available")
        end

        println(io)

        # Configuration
        println(io, "## Benchmark Configuration")
        println(io)
        println(io, "- Samples per benchmark: $BENCHMARK_SAMPLES")
        println(io, "- Minimum benchmark time: $(BENCHMARK_SECONDS)s")
        println(io, "- Warmup iterations: $WARMUP_ITERATIONS")
        println(io)

        # Group results by operation
        operations = unique([r.operation for r in results])

        for op in operations
            op_results = filter(r -> r.operation == op, results)

            println(io, "## $(titlecase(replace(op, "_" => " ")))")
            println(io)

            # Create comparison table
            println(io, "| Image Size | Backend | Time (ms) | Speedup | Throughput | Memory (MB) |")
            println(io, "|------------|---------|-----------|---------|------------|-------------|")

            # Group by image size and parameters
            sizes = unique([r.image_size for r in op_results])

            for size in sizes
                size_results = filter(r -> r.image_size == size, op_results)

                # Find CPU baseline
                cpu_result = findfirst(r -> r.backend == "CPU", size_results)
                cpu_time = cpu_result !== nothing ? size_results[cpu_result].time_median * 1000 : NaN

                for r in size_results
                    time_ms = r.time_median * 1000
                    speedup = cpu_time / time_ms
                    speedup_str = isnan(speedup) || speedup == 1.0 ? "-" : @sprintf("%.2fx", speedup)

                    # Format throughput based on operation type
                    if r.operation == "interpolation"
                        throughput_str = @sprintf("%.2e pts/s", r.throughput)
                    else
                        throughput_str = @sprintf("%.2e vx/s", r.throughput)
                    end

                    println(io, "| $(r.image_size) | $(r.backend) | $(@sprintf("%.2f", time_ms)) | $speedup_str | $throughput_str | $(@sprintf("%.1f", r.memory_bytes / 1024^2)) |")
                end

                println(io, "|------------|---------|-----------|---------|------------|-------------|")
            end

            println(io)
        end

        # Memory transfer stats
        if !isempty(memory_stats)
            println(io, "## Memory Transfer Performance")
            println(io)
            println(io, "| Direction | Time (ms) | Transfer Rate (MB/s) | Data Size (MB) |")
            println(io, "|-----------|-----------|----------------------|----------------|")

            if haskey(memory_stats, "cpu_to_gpu_time")
                @printf(io, "| CPU → GPU | %.2f | %.2f | %.1f |\n",
                    memory_stats["cpu_to_gpu_time"] * 1000,
                    memory_stats["cpu_to_gpu_rate_mbs"],
                    memory_stats["data_size_mb"])
            end

            if haskey(memory_stats, "gpu_to_cpu_time")
                @printf(io, "| GPU → CPU | %.2f | %.2f | %.1f |\n",
                    memory_stats["gpu_to_cpu_time"] * 1000,
                    memory_stats["gpu_to_cpu_rate_mbs"],
                    memory_stats["data_size_mb"])
            end

            println(io)
        end

        # Summary statistics
        println(io, "## Summary")
        println(io)

        # Calculate overall speedups
        for op in operations
            op_results = filter(r -> r.operation == op, results)

            cpu_results = filter(r -> r.backend == "CPU", op_results)
            gpu_results = filter(r -> r.backend == "CUDA", op_results)

            if !isempty(cpu_results) && !isempty(gpu_results)
                cpu_times = [r.time_median for r in cpu_results]
                gpu_times = [r.time_median for r in gpu_results]

                avg_speedup = mean(cpu_times) / mean(gpu_times)

                println(io, "- **$(titlecase(replace(op, "_" => " "))):** Average CUDA speedup: $(@sprintf("%.2fx", avg_speedup))")
            end
        end

        println(io)

        # Recommendations
        println(io, "## Recommendations")
        println(io)

        # Find best/worst performing operations
        if !isempty(results)
            cuda_results = filter(r -> r.backend == "CUDA", results)

            if !isempty(cuda_results)
                # Find operations with highest throughput
                best_throughput = maximum([r.throughput for r in cuda_results])
                best_result = findfirst(r -> r.throughput == best_throughput, cuda_results)

                println(io, "- Best GPU performance: $(cuda_results[best_result].name) ($(@sprintf("%.2e", best_throughput)) throughput)")

                # Check for memory bottlenecks
                if !isempty(memory_stats) && haskey(memory_stats, "cpu_to_gpu_rate_mbs")
                    transfer_rate = memory_stats["cpu_to_gpu_rate_mbs"]

                    if transfer_rate < 10000  # Less than 10 GB/s
                        println(io, "- Memory transfer rate is $(@sprintf("%.1f", transfer_rate)) MB/s. Consider:")
                        println(io, "  - Keeping data on GPU between operations")
                        println(io, "  - Batch processing multiple operations")
                        println(io, "  - Using PCIe Gen4 if available")
                    end
                end
            end
        end

        println(io)
        println(io, "---")
        println(io, "Report generated by MedImages.jl benchmark suite")
    end

    println("Report saved to: $output_file")
end

"""
    plot_benchmark_results(results::Vector{BenchmarkResult})

Create ASCII plots of benchmark results using UnicodePlots.
"""
function plot_benchmark_results(results::Vector)
    if !UNICODEPLOTS_LOADED
        @warn "UnicodePlots not available, skipping visualization"
        return
    end

    try
        println("\n" * "="^80)
        println("Benchmark Visualization")
        println("="^80)

        # Group by operation
        operations = unique([r.operation for r in results])

        for op in operations
            op_results = filter(r -> r.operation == op, results)

            # Get CPU and CUDA times
            cpu_results = filter(r -> r.backend == "CPU", op_results)
            cuda_results = filter(r -> r.backend == "CUDA", op_results)

            if !isempty(cpu_results) && !isempty(cuda_results)
                # Speedup plot
                sizes = unique([r.image_size for r in op_results])
                speedups = Float64[]

                for size in sizes
                    cpu_r = findfirst(r -> r.backend == "CPU" && r.image_size == size, op_results)
                    cuda_r = findfirst(r -> r.backend == "CUDA" && r.image_size == size, op_results)

                    if cpu_r !== nothing && cuda_r !== nothing
                        speedup = op_results[cpu_r].time_median / op_results[cuda_r].time_median
                        push!(speedups, speedup)
                    end
                end

                if !isempty(speedups)
                    println("\n$(titlecase(replace(op, "_" => " "))) Speedup:")
                    plt = barplot(sizes, speedups, title="CPU vs CUDA", xlabel="Image Size", ylabel="Speedup")
                    println(plt)
                end
            end
        end

    catch e
        @warn "Failed to generate plots (UnicodePlots may not be available)" exception = e
    end
end

"""
    compare_with_simpleitk(results::Vector{BenchmarkResult}, images::Dict=Dict())

Compare MedImages.jl results with SimpleITK baseline (if available).
"""
function compare_with_simpleitk(results::Vector, images::Dict=Dict())
    println("\n" * "="^80)
    println("SimpleITK Comparison")
    println("="^80)

    if isempty(images)
        println("Note: No images provided for SimpleITK comparison")
        println("Skipping detailed comparison...")
        println("="^80)
        return nothing
    end

    # Include SimpleITK benchmarks
    try
        include(joinpath(@__DIR__, "simpleitk_benchmarks.jl"))

        # Run SimpleITK benchmarks
        # compare_with_simpleitk_full is defined by the include above; use
        # invokelatest to avoid world-age issues when the method is first loaded
        sitk_results = Base.invokelatest(compare_with_simpleitk_full, results, images)

        # Generate comparison table
        comparison_data = generate_comparison_table(results, sitk_results)

        return sitk_results, comparison_data
    catch e
        @warn "SimpleITK comparison failed" exception = e
        println("SimpleITK comparison skipped due to error")
        return nothing
    end
end

"""
    print_summary(results::Vector{BenchmarkResult})

Print a summary of benchmark results to console.
"""
function print_summary(results::Vector)
    println("\n" * "="^80)
    println("Benchmark Summary")
    println("="^80)

    println("\nTotal benchmarks run: $(length(results))")

    # Count by operation
    operations = unique([r.operation for r in results])
    for op in operations
        count = length(filter(r -> r.operation == op, results))
        println("  $(titlecase(replace(op, "_" => " "))): $count")
    end

    # Calculate average speedups
    println("\nAverage CUDA Speedups:")
    for op in operations
        op_results = filter(r -> r.operation == op, results)

        cpu_results = filter(r -> r.backend == "CPU", op_results)
        cuda_results = filter(r -> r.backend == "CUDA", op_results)

        if !isempty(cpu_results) && !isempty(cuda_results)
            cpu_times = [r.time_median for r in cpu_results]
            gpu_times = [r.time_median for r in cuda_results]

            if length(cpu_times) == length(gpu_times)
                speedups = cpu_times ./ gpu_times
                avg_speedup = mean(speedups)
                min_speedup = minimum(speedups)
                max_speedup = maximum(speedups)

                @printf("  %s: %.2fx (min: %.2fx, max: %.2fx)\n",
                    titlecase(replace(op, "_" => " ")),
                    avg_speedup, min_speedup, max_speedup)
            end
        end
    end

    println("="^80)
end
