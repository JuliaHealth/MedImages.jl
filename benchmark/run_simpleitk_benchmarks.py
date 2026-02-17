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
import os
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

def benchmark_separate_affine(img, num_runs=10):
    """Benchmark separate affine transformations (Translate -> Rotate -> Scale)."""
    # Parameters matches Julia benchmark
    translation = (10.0, 10.0, 10.0)
    rotation_angle = 45.0
    scale_factor = 1.2

    # Option 1: Chain of ResampleImageFilters (closest to "separate" operations)
    # But usually we want to benchmark the transform composition + single resample
    # vs doing 3 resamples.
    # The Julia benchmark did: translate -> rotate -> scale (3 resampling ops if lazy evaluation isn't used)
    # MedImages.jl operations are eager by default unless composed.
    # The "separate_affine" benchmark in Julia was:
    # img_t = translate_mi(...)
    # img_r = rotate_mi(img_t, ...)
    # result = scale_mi(img_r, ...)
    # So it implies 3 actual resampling steps if they are eager.

    # We will simulate 3 separate resampling steps to match the Julia benchmark logic exactly.

    # 1. Translate
    trans_transform = sitk.TranslationTransform(3)
    trans_transform.SetOffset(translation)

    # 2. Rotate
    # Rotate around Z axis (index 2)
    center = [c + s*sp/2 for c, s, sp in zip(img.GetOrigin(), img.GetSize(), img.GetSpacing())]
    rot_transform = sitk.Euler3DTransform()
    rot_transform.SetCenter(center)
    rot_transform.SetRotation(0, 0, np.deg2rad(rotation_angle))

    # 3. Scale
    # SimpleITK ScaleTransform uses inverse scale parameter typically?
    # No, ScaleTransform SetScale(scales).
    scale_transform = sitk.ScaleTransform(3)
    scale_transform.SetCenter(center)
    scale_transform.SetScale((scale_factor, scale_factor, scale_factor))

    # Setup resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)

    # Warmup
    resampler.SetTransform(trans_transform)
    t_img = resampler.Execute(img)
    resampler.SetTransform(rot_transform)
    r_img = resampler.Execute(t_img)
    resampler.SetTransform(scale_transform)
    s_img = resampler.Execute(r_img)

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        
        # 1. Translate
        resampler.SetTransform(trans_transform)
        t_img = resampler.Execute(img)
        
        # 2. Rotate
        resampler.SetTransform(rot_transform)
        resampler.SetReferenceImage(t_img) # Update reference to new image grid? 
        # Actually separate operations usually change grid. 
        # MedImages.translate_mi changes Origin.
        # MedImages.rotate_mi changes Direction/Resamples?
        # For fair comparison, we assume the Julia benchmark does 3 full resamplings.
        # So we do 3 full resamplings here.
        r_img = resampler.Execute(t_img)
        
        # 3. Scale
        resampler.SetTransform(scale_transform)
        resampler.SetReferenceImage(r_img)
        s_img = resampler.Execute(r_img)
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        # Reset reference for next run
        resampler.SetReferenceImage(img)

    return times

def benchmark_fused_affine(img, num_runs=10):
    """Benchmark fused affine transformation (Single Matrix)."""
    # Parameters matches separate_affine
    translation = (10.0, 10.0, 10.0)
    rotation_angle = 45.0
    scale_factor = 1.2
    
    center = [c + s*sp/2 for c, s, sp in zip(img.GetOrigin(), img.GetSize(), img.GetSpacing())]

    # Create composite transform
    # Order: Scale -> Rotate -> Translate
    # In SimpleITK composite: T_composite(x) = T_outer(T_inner(x))
    # We want: Translate(Rotate(Scale(x)))
    
    scale = sitk.ScaleTransform(3)
    scale.SetCenter(center)
    scale.SetScale((scale_factor, scale_factor, scale_factor))
    
    rotate = sitk.Euler3DTransform()
    rotate.SetCenter(center)
    rotate.SetRotation(0, 0, np.deg2rad(rotation_angle))
    
    translate = sitk.TranslationTransform(3)
    translate.SetOffset(translation)
    
    composite = sitk.CompositeTransform([translate, rotate, scale])
    
    # Or just use AffineTransform and set matrix directly if we want to be exact?
    # But composite is "fused" in execution.
    # Let's use clean AffineTransform for "single matrix" equivalent
    # But constructing one from params is mathy. Composite is safer to match logic.
    # SimpleITK will flatten composite to single transform if possible or efficient? 
    # Actually explicit AffineTransform is better for "fused".
    
    # Let's stick to Composite which represents "one resampling with combined transform".
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(composite)

    # Warmup
    resampler.Execute(img)

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = resampler.Execute(img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times

def benchmark_orientation(img, target_orientation, num_runs=10):
    """Benchmark orientation change operation."""
    # target_orientation is string like 'LAS', 'RAS', etc.
    # SimpleITK DICOMOrientImageFilter can be used
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(target_orientation)
    
    # Warmup
    orient_filter.Execute(img)
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = orient_filter.Execute(img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return times

def benchmark_interpolation(dims, point_count, interpolator, num_runs=10):
    """Benchmark raw interpolation of N points."""
    # Create simple volume
    data = np.random.rand(*dims).astype(np.float32)
    img = sitk.GetImageFromArray(data)
    
    # Random points in physical space
    # (x, y, z)
    points = np.random.rand(point_count, 3).astype(np.float64)
    # Scale to image dims
    for i in range(3):
        points[:, i] *= (dims[2-i] - 1) # SimpleITK uses [x,y,z] order for points
        
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        # In SimpleITK, interpolation of arbitrary points is usually done via EvaluateAtPhysicalPoint
        # but that would be slow for 1M points in a loop.
        # Efficient way: Resample with a displacement field or a set of points?
        # Actually, for 1M points, usually we use sitk.Resample with a specific grid.
        # But if we want arbitrary points:
        # We can create a 1x1xN grid and resample onto it.
        
        # Create a "point cloud" image
        # This is a bit complex in SITK.
        # Alternatively, use Sitk.EvaluateAtPhysicalPoint in a vectorized way if possible?
        # No, SITK doesn't expose vectorized point evaluation easily in Python.
        
        # Let's approximate by resampling onto a 100x100x100 grid if point_count=1M
        # to match the workload of 1M interpolations.
        
        # BUT the table says "Interpolation (1M points)".
        # Let's just use a grid of size that equals point_count.
        target_size = [int(point_count**(1/3.0))] * 3
        if target_size[0]**3 < point_count:
           target_size[0] += 1 # Ensure at least point_count
           
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(target_size)
        resampler.SetOutputSpacing([1.0]*3)
        resampler.SetOutputOrigin([0.0]*3)
        resampler.SetOutputDirection([1,0,0, 0,1,0, 0,0,1])
        if interpolator == "nearest":
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)
            
        result = resampler.Execute(img)
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
    
    # 5. Affine Comparison Benchmarks
    print("\n  Affine Comparison benchmarks:")
    
    # Separate
    times = benchmark_separate_affine(img, num_runs)
    median_ms = np.median(times) * 1000
    print(f"    Separate Affine: {median_ms:.3f} ms")
    results.append({
        "operation": "separate_affine",
        "image_size": size_name,
        "time_mean_ms": np.mean(times) * 1000,
        "time_median_ms": median_ms,
        "time_std_ms": np.std(times) * 1000
    })
    
    # Fused
    times = benchmark_fused_affine(img, num_runs)
    median_ms = np.median(times) * 1000
    print(f"    Fused Affine: {median_ms:.3f} ms")
    results.append({
        "operation": "fused_affine",
        "image_size": size_name,
        "time_mean_ms": np.mean(times) * 1000,
        "time_median_ms": median_ms,
        "time_std_ms": np.std(times) * 1000
    })

    # 6. Orientation benchmarks
    print("\n  Orientation benchmarks:")
    for orient in ["LAS", "RAS", "RPI"]:
        times = benchmark_orientation(img, orient, num_runs)
        median_ms = np.median(times) * 1000
        print(f"    Orientation {orient}: {median_ms:.3f} ms")
        results.append({
            "operation": f"orientation_{orient}",
            "image_size": size_name,
            "time_mean_ms": np.mean(times) * 1000,
            "time_median_ms": median_ms,
            "time_std_ms": np.std(times) * 1000
        })

    # 7. Interpolation benchmarks (1M points)
    print("\n  Interpolation benchmarks (1M points):")
    for interp_name, interp_type in [("Nearest", "nearest"), ("Linear", "linear")]:
        times = benchmark_interpolation(dims, 1000000, interp_type, num_runs)
        median_ms = np.median(times) * 1000
        print(f"    Interpolation (1M, {interp_name}): {median_ms:.3f} ms")
        results.append({
            "operation": f"interpolate_1M_{interp_type}",
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
    
    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
