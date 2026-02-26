# Organ-Specific Affine Registration Network

This experiment implements a deep learning pipeline for performing **multi-organ affine registration**. The model takes a CT scan and an Atlas mask as input, predicts affine transformation parameters (Rotation, Translation, Scale, Shear, Center) for each organ independently, and applies them to match a Gold Standard segmentation.

## Key Features

*   **Multi-Scale CNN**: A 3D CNN with parallel branches of varying kernel sizes (3x3, 5x5, 7x7) to capture features at different scales.
*   **Constrained Affine Output**: The MLP head outputs specific parameters (e.g., rotation in radians, positive scaling) rather than a raw affine matrix, allowing for interpretable and constrained updates.
*   **Fused Differentiable Kernel**: A custom `KernelAbstractions.jl` kernel computes the loss efficiently on GPU. It combines:
    1.  **Distance Metric**: Penalizes points moving too far from the organ's barycenter.
    2.  **Interpolation Metric**: Checks consistency with the Gold Standard segmentation (one-hot encoded).
    *   Optimized with binary tree reduction in shared memory (no atomics) and inlined helper functions.
*   **Distributed Training**: Supports multi-GPU/multi-node training via `Lux.DistributedUtils`.

## Directory Structure

*   `src/`: Source code.
    *   `preprocessing.jl`: Data preparation (Atlas point extraction, downsampling, one-hot encoding).
    *   `model.jl`: Lux.jl model definition.
    *   `fused_loss.jl`: Custom differentiable loss kernel.
*   `test/`: Unit tests.
*   `train.jl`: Main training script (supports distributed execution).

## Dependencies

Required Julia packages (see `Project.toml`):
*   `Lux.jl`
*   `KernelAbstractions.jl`
*   `CUDA.jl` (for GPU support)
*   `Enzyme.jl` & `Zygote.jl` (for AD)
*   `MedImages.jl` (local dependency)
*   `Optimisers.jl`
*   `MPI.jl` (for distributed training)

## How to Run

### 1. Single Process (Testing/Debug)

```bash
julia --project=. experiments/organ_affine_registration/train.jl
```

### 2. Distributed Training (MPI)

To run on multiple GPUs or nodes using MPI:

```bash
mpiexec -n <NUM_PROCESSES> julia --project=. experiments/organ_affine_registration/train.jl
```

### 3. Running Tests

```bash
julia --project=. experiments/organ_affine_registration/test/test_organ_registration.jl
```

## Algorithm Details

1.  **Preprocessing**:
    *   Extracts voxel coordinates for each organ from the Atlas.
    *   Downsamples points deterministically to a fixed size (e.g., 512).
    *   Computes barycenters and max radii for penalty constraints.
2.  **Forward Pass**:
    *   CNN processes the CT + Atlas input.
    *   MLP predicts affine parameters per organ.
3.  **Loss Calculation (Fused Kernel)**:
    *   Applies the predicted affine transform to the Atlas points.
    *   Computes `Softplus(distance - radius)` penalty.
    *   Samples the Gold Standard volume at the transformed coordinates using trilinear interpolation.
    *   Reduces losses using efficient GPU-friendly patterns.
4.  **Optimization**:
    *   Gradients flow back from the loss kernel (via Enzyme/Zygote) to update the CNN weights.
