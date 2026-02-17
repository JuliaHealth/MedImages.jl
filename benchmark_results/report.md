# MedImages.jl GPU Benchmark Report

**Generated:** 2026-02-17T10:31:31.480

## System Information

- Julia version: 1.10.10
- Platform: x86_64-linux-gnu
- CPU: goldmont (24 threads)
- GPU: NVIDIA GeForce RTX 3090
- GPU: Not available

## Benchmark Configuration

- Samples per benchmark: 10
- Minimum benchmark time: 2.0s
- Warmup iterations: 3

## Affine

| Image Size | Backend | Time (ms) | Speedup | Throughput | Memory (MB) |
|------------|---------|-----------|---------|------------|-------------|
| 256x256x128 | CPU | 656.05 | - | 5.11e+07 vx/s | 516.5 |
| 256x256x128 | CPU | 2432.22 | 0.27x | 5.52e+07 vx/s | 2066.1 |
| 256x256x128 | CUDA | 2.85 | 230.12x | 1.18e+10 vx/s | 0.0 |
| 256x256x128 | CUDA | 10.29 | 63.78x | 1.30e+10 vx/s | 0.1 |
|------------|---------|-----------|---------|------------|-------------|

## Affine Comparison

| Image Size | Backend | Time (ms) | Speedup | Throughput | Memory (MB) |
|------------|---------|-----------|---------|------------|-------------|
| 256x256x128 | CPU | 1462.88 | - | 5.73e+06 vx/s | 1722.9 |
| 256x256x128 | CPU | 155.50 | 9.41x | 5.39e+07 vx/s | 129.1 |
| 256x256x128 | CUDA | 1326.27 | 1.10x | 6.32e+06 vx/s | 1810.3 |
| 256x256x128 | CUDA | 0.95 | 1543.66x | 8.85e+09 vx/s | 0.0 |
|------------|---------|-----------|---------|------------|-------------|

## Interpolation

| Image Size | Backend | Time (ms) | Speedup | Throughput | Memory (MB) |
|------------|---------|-----------|---------|------------|-------------|
| 256x256x128 | CPU | 0.45 | - | 2.21e+08 pts/s | 0.4 |
| 256x256x128 | CPU | 1.88 | 0.24x | 5.33e+07 pts/s | 0.4 |
| 256x256x128 | CPU | 12.22 | 0.04x | 8.19e+07 pts/s | 3.8 |
| 256x256x128 | CPU | 23.10 | 0.02x | 4.33e+07 pts/s | 3.8 |
| 256x256x128 | CUDA | 0.02 | 19.34x | 4.26e+09 pts/s | 0.0 |
| 256x256x128 | CUDA | 0.21 | 2.16x | 4.76e+08 pts/s | 0.0 |
| 256x256x128 | CUDA | 0.28 | 1.59x | 3.51e+09 pts/s | 0.0 |
| 256x256x128 | CUDA | 0.69 | 0.65x | 1.44e+09 pts/s | 0.0 |
|------------|---------|-----------|---------|------------|-------------|

## Resampling

| Image Size | Backend | Time (ms) | Speedup | Throughput | Memory (MB) |
|------------|---------|-----------|---------|------------|-------------|
| 256x256x128 | CPU | 6.95 | - | 1.21e+09 vx/s | 4.0 |
| 256x256x128 | CPU | 9.68 | 0.72x | 8.66e+08 vx/s | 4.0 |
| 256x256x128 | CPU | 165.16 | 0.04x | 5.08e+07 vx/s | 108.0 |
| 256x256x128 | CPU | 263.04 | 0.03x | 3.19e+07 vx/s | 108.0 |
| 256x256x128 | CPU | 0.69 | 10.14x | 1.22e+10 vx/s | 0.5 |
| 256x256x128 | CPU | 1.13 | 6.17x | 7.45e+09 vx/s | 0.5 |
| 256x256x128 | CUDA | 0.06 | 123.96x | 1.50e+11 vx/s | 0.0 |
| 256x256x128 | CUDA | 0.32 | 21.48x | 2.59e+10 vx/s | 0.0 |
| 256x256x128 | CUDA | 1.37 | 5.06x | 6.11e+09 vx/s | 0.0 |
| 256x256x128 | CUDA | 1.97 | 3.52x | 4.25e+09 vx/s | 0.0 |
| 256x256x128 | CUDA | 0.02 | 421.20x | 5.08e+11 vx/s | 0.0 |
| 256x256x128 | CUDA | 0.02 | 310.09x | 3.74e+11 vx/s | 0.0 |
|------------|---------|-----------|---------|------------|-------------|

## Rotation

| Image Size | Backend | Time (ms) | Speedup | Throughput | Memory (MB) |
|------------|---------|-----------|---------|------------|-------------|
| 256x256x128 | CPU | 1273.57 | - | 6.59e+06 vx/s | 1280.0 |
| 256x256x128 | CPU | 1291.35 | 0.99x | 6.50e+06 vx/s | 1280.0 |
| 256x256x128 | CPU | 1265.69 | 1.01x | 6.63e+06 vx/s | 1280.0 |
| 256x256x128 | CUDA | 1275.62 | 1.00x | 6.58e+06 vx/s | 1312.0 |
| 256x256x128 | CUDA | 1274.13 | 1.00x | 6.58e+06 vx/s | 1312.0 |
| 256x256x128 | CUDA | 1330.65 | 0.96x | 6.30e+06 vx/s | 1312.0 |
|------------|---------|-----------|---------|------------|-------------|

## Crop

| Image Size | Backend | Time (ms) | Speedup | Throughput | Memory (MB) |
|------------|---------|-----------|---------|------------|-------------|
| 256x256x128 | CPU | 0.00 | - | 2.07e+13 vx/s | 0.0 |
| 256x256x128 | CPU | 0.00 | 0.99x | 2.06e+13 vx/s | 0.0 |
| 256x256x128 | CPU | 0.00 | 0.65x | 1.34e+13 vx/s | 0.0 |
| 256x256x128 | CUDA | 0.00 | 0.51x | 1.05e+13 vx/s | 0.0 |
| 256x256x128 | CUDA | 0.00 | 0.51x | 1.07e+13 vx/s | 0.0 |
| 256x256x128 | CUDA | 0.00 | 0.50x | 1.03e+13 vx/s | 0.0 |
|------------|---------|-----------|---------|------------|-------------|

## Pad

| Image Size | Backend | Time (ms) | Speedup | Throughput | Memory (MB) |
|------------|---------|-----------|---------|------------|-------------|
| 256x256x128 | CPU | 42.23 | - | 1.99e+08 vx/s | 127.0 |
| 256x256x128 | CPU | 78.85 | 0.54x | 1.06e+08 vx/s | 304.0 |
| 256x256x128 | CPU | 62.60 | 0.67x | 1.34e+08 vx/s | 202.9 |
|------------|---------|-----------|---------|------------|-------------|

## Orientation

| Image Size | Backend | Time (ms) | Speedup | Throughput | Memory (MB) |
|------------|---------|-----------|---------|------------|-------------|
| 256x256x128 | CPU | 13.85 | - | 6.06e+08 vx/s | 32.0 |
| 256x256x128 | CPU | 13.42 | 1.03x | 6.25e+08 vx/s | 32.0 |
| 256x256x128 | CPU | 15.87 | 0.87x | 5.29e+08 vx/s | 32.0 |
| 256x256x128 | CUDA | 0.19 | 71.48x | 4.33e+10 vx/s | 0.0 |
| 256x256x128 | CUDA | 0.32 | 43.10x | 2.61e+10 vx/s | 0.0 |
| 256x256x128 | CUDA | 0.11 | 121.10x | 7.34e+10 vx/s | 0.0 |
|------------|---------|-----------|---------|------------|-------------|

## Summary

- **Affine:** Average CUDA speedup: 235.10x
- **Affine Comparison:** Average CUDA speedup: 1.22x
- **Interpolation:** Average CUDA speedup: 31.04x
- **Resampling:** Average CUDA speedup: 118.58x
- **Rotation:** Average CUDA speedup: 0.99x
- **Crop:** Average CUDA speedup: 0.60x
- **Orientation:** Average CUDA speedup: 68.54x

## Recommendations

- Best GPU performance: crop_mi (1.07e+13 throughput)

---
Report generated by MedImages.jl benchmark suite
