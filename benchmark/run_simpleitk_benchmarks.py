#!/usr/bin/env python3
"""
SimpleITK Benchmark Script
Runs benchmarks for comparison with MedImages.jl
"""

import SimpleITK as sitk
import numpy as np
import time
import csv
import sys
from datetime import datetime

def create_synthetic_image(dims, spacing=(1.0, 1.0, 1.0)):
    """Create a synthetic test image."""
    # Create sphere structure
    data = np.zeros(dims, dtype=np.float32)
    center = [d // 2 for d in dims]
    radius = min(dims) // 4

    for k in range(dims[2]):
        for j in range(dims[1]):
            for i in range(dims[0]):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                if dist < radius:
                    data[i, j, k] = 1000.0 * (1.0 - dist/radius) + np.random.randn() * 10.0
                else:
                    data[i, j, k] = np.random.randn() * 10.0

    # Create SimpleITK image (expects [z, y, x] order)
    img = sitk.GetImageFromArray(data.T)
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    return img

def benchmark_resample(img, new_spacing, interpolator, num_runs=10):
    """Benchmark resampling operation."""
    resampler = sitk.ResampleImageFilter()

    orig_size = img.GetSize()
    orig_spacing = img.GetSpacing()

    new_size = [int(np.ceil(orig_size[i] * orig_spacing[i] / new_spacing[i])) for i in range(3)]

    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())

    if interpolator == "nearest":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    # Warmup
    resampler.Execute(img)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = resampler.Execute(img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times

def benchmark_rotation(img, angle_degrees, axis=2, num_runs=10):
    """Benchmark rotation operation."""
    size = img.GetSize()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()

    center = [origin[i] + (size[i] * spacing[i]) / 2.0 for i in range(3)]

    transform = sitk.Euler3DTransform()
    transform.SetCenter(center)

    angle_rad = np.deg2rad(angle_degrees)
    if axis == 0:
        transform.SetRotation(angle_rad, 0.0, 0.0)
    elif axis == 1:
        transform.SetRotation(0.0, angle_rad, 0.0)
    else:
        transform.SetRotation(0.0, 0.0, angle_rad)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(transform)

    # Warmup
    resampler.Execute(img)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = resampler.Execute(img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times

def benchmark_crop(img, crop_start, crop_size, num_runs=10):
    """Benchmark crop operation."""
    # Warmup
    result = sitk.Extract(img, crop_size, crop_start)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = sitk.Extract(img, crop_size, crop_start)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times

def benchmark_pad(img, pad_lower, pad_upper, pad_value=0.0, num_runs=10):
    """Benchmark pad operation."""
    # Warmup
    result = sitk.ConstantPad(img, pad_lower, pad_upper, pad_value)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = sitk.ConstantPad(img, pad_lower, pad_upper, pad_value)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times

def run_benchmarks(dims, size_name, num_runs=10):
    """Run all benchmarks for a given image size."""
    results = []
    spacing = (1.0, 1.0, 1.0)

    print(f"\n### SimpleITK Benchmarks for {size_name} ({dims[0]}x{dims[1]}x{dims[2]}) ###")

    # Create image
    print("  Creating synthetic image...")
    img = create_synthetic_image(dims, spacing)

    # 1. Resampling benchmarks
    print("\n  Resampling benchmarks:")

    # Downsample 2x
    new_spacing_down = (spacing[0] * 2.0, spacing[1] * 2.0, spacing[2] * 2.0)
    for interp_name, interp_type in [("Nearest", "nearest"), ("Linear", "linear")]:
        times = benchmark_resample(img, new_spacing_down, interp_type, num_runs)
        median_ms = np.median(times) * 1000
        print(f"    Resample (downsample 2x, {interp_name}): {median_ms:.3f} ms")
        results.append({
            "operation": f"resample_downsample_2x_{interp_type}",
            "image_size": size_name,
            "time_mean_ms": np.mean(times) * 1000,
            "time_median_ms": median_ms,
            "time_std_ms": np.std(times) * 1000
        })

    # Upsample 1.5x
    new_spacing_up = (spacing[0] / 1.5, spacing[1] / 1.5, spacing[2] / 1.5)
    for interp_name, interp_type in [("Nearest", "nearest"), ("Linear", "linear")]:
        times = benchmark_resample(img, new_spacing_up, interp_type, num_runs)
        median_ms = np.median(times) * 1000
        print(f"    Resample (upsample 1.5x, {interp_name}): {median_ms:.3f} ms")
        results.append({
            "operation": f"resample_upsample_1.5x_{interp_type}",
            "image_size": size_name,
            "time_mean_ms": np.mean(times) * 1000,
            "time_median_ms": median_ms,
            "time_std_ms": np.std(times) * 1000
        })

    # Downsample 4x
    new_spacing_4x = (spacing[0] * 4.0, spacing[1] * 4.0, spacing[2] * 4.0)
    for interp_name, interp_type in [("Nearest", "nearest"), ("Linear", "linear")]:
        times = benchmark_resample(img, new_spacing_4x, interp_type, num_runs)
        median_ms = np.median(times) * 1000
        print(f"    Resample (downsample 4x, {interp_name}): {median_ms:.3f} ms")
        results.append({
            "operation": f"resample_downsample_4x_{interp_type}",
            "image_size": size_name,
            "time_mean_ms": np.mean(times) * 1000,
            "time_median_ms": median_ms,
            "time_std_ms": np.std(times) * 1000
        })

    # 2. Rotation benchmarks
    print("\n  Rotation benchmarks:")
    for angle in [90, 180, 270]:
        times = benchmark_rotation(img, float(angle), axis=2, num_runs=num_runs)
        median_ms = np.median(times) * 1000
        print(f"    Rotate {angle} degrees: {median_ms:.3f} ms")
        results.append({
            "operation": f"rotate_{angle}deg",
            "image_size": size_name,
            "time_mean_ms": np.mean(times) * 1000,
            "time_median_ms": median_ms,
            "time_std_ms": np.std(times) * 1000
        })

    # 3. Crop benchmarks
    print("\n  Crop benchmarks:")
    crop_offset = [d // 4 for d in dims]
    crop_size = [d // 2 for d in dims]

    times = benchmark_crop(img, crop_offset, crop_size, num_runs)
    median_ms = np.median(times) * 1000
    print(f"    Crop 50%: {median_ms:.3f} ms")
    results.append({
        "operation": "crop_50pct",
        "image_size": size_name,
        "time_mean_ms": np.mean(times) * 1000,
        "time_median_ms": median_ms,
        "time_std_ms": np.std(times) * 1000
    })

    # 4. Pad benchmarks
    print("\n  Pad benchmarks:")
    pad_amount = [d // 4 for d in dims]

    times = benchmark_pad(img, pad_amount, pad_amount, 0.0, num_runs)
    median_ms = np.median(times) * 1000
    print(f"    Pad 50%: {median_ms:.3f} ms")
    results.append({
        "operation": "pad_50pct",
        "image_size": size_name,
        "time_mean_ms": np.mean(times) * 1000,
        "time_median_ms": median_ms,
        "time_std_ms": np.std(times) * 1000
    })

    return results

def main():
    print("="*80)
    print("SimpleITK Benchmark Suite")
    print("="*80)
    print(f"SimpleITK version: {sitk.Version_VersionString()}")
    print(f"NumPy version: {np.__version__}")
    print(f"Start time: {datetime.now().isoformat()}")

    all_results = []

    # Run benchmarks for different sizes
    sizes = [
        ((256, 256, 128), "medium"),
        ((512, 512, 256), "large"),
    ]

    for dims, name in sizes:
        results = run_benchmarks(dims, name, num_runs=10)
        all_results.extend(results)

    # Save results to CSV
    output_file = "benchmark_results/simpleitk_results.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["operation", "image_size", "time_mean_ms", "time_median_ms", "time_std_ms"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "="*80)
    print("SimpleITK Benchmark Results Summary")
    print("="*80)
    print(f"\n{'Operation':<40} | {'Size':<10} | {'Time (ms)':<10}")
    print("-"*65)
    for r in all_results:
        print(f"{r['operation']:<40} | {r['image_size']:<10} | {r['time_median_ms']:>8.3f}")

    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
