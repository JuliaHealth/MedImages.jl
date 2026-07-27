# MedImages.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliahealth.org/MedImages.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliahealth.org/MedImages.jl/dev)
[![Build Status](https://github.com/JuliaHealth/MedImages.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaHealth/MedImages.jl/actions/workflows/CI.yml)

GPU-accelerated, differentiable medical image processing in Julia. Unified I/O for NIfTI, DICOM, HDF5, and MHA with automatic spatial metadata management.

---

## Architecture

![Architecture](docs/assets/architecture.png)

---

## Quick Start

```julia
using MedImages
ct = load_image("scan.nii.gz", "CT")
resampled = resample_to_spacing(ct, (1.0, 1.0, 1.0), Linear_en)
create_nii_from_medimage(resampled, "output.nii.gz")
```

---

## MedImage Data Structure

All fields travel with the voxel data through every operation.

![MedImage Data Structure](docs/assets/table-datastructure.png)

Additional fields: `date_of_saving`, `acquistion_time`, `study_uid`, `patient_uid`, `series_uid`, `study_description`, `legacy_file_name`, `display_data`, `clinical_data`, `is_contrast_administered`, `metadata`.

---

## I/O

```julia
ct  = load_image("scan.nii.gz", "CT")          # NIfTI
mri = load_image("dicom_dir/", "MRI")           # DICOM
create_nii_from_medimage(ct, "out.nii.gz")       # Export NIfTI
save_med_image(ct, "scan.h5")                    # Save HDF5
```

---

## Transformations

All transforms preserve spatial metadata and are differentiable.

```julia
rotated    = rotate_mi(im, 3, 45.0, Linear_en)          # axis 3 (Z), 45 degrees
cropped    = crop_mi(im, (10,10,5), (100,100,50), Linear_en)
padded     = pad_mi(im, (5,5,5), (5,5,5), 0.0, Linear_en)
translated = translate_mi(im, 10, 2, Linear_en)
```

> **Note:** `rotate_mi` takes the axis as an `Int` (`1`=X, `2`=Y, `3`=Z) and the angle in **degrees** (not radians). `rotate_mi(ct, 3, 45.0, Linear_en)` matches SimpleITK's −45° rotation about the image centre pixel-for-pixel (Pearson `1.0000`); the two use opposite handedness.

Synthetic batched transforms (mid-slice of each output) and the same operations on a real CT volume:

![Synthetic transforms grid](test/visual_output/screenshots/Synth_transforms_grid.png)

```bash
# Reproduce the MedImages transform outputs (NIfTI)
julia --startup-file=no --project=. test/generate_visual_samples.jl
```

![Real-CT transforms grid](test/visual_output/screenshots/CT_comparison_grid_labeled.png)

```bash
# Apply the same operations to a real CT (test_data/volume-0.nii.gz)
julia --startup-file=no --project=. experiments/make_real_ct_samples.jl
```

---

## Fused Affine Kernel

Rotation, scaling, shearing, and translation compose into a single 4×4 homogeneous matrix that is applied in **one interpolation pass**. Fusing the composition avoids the compounding blur and edge artifacts of chaining separate resamples, and maps directly onto a single GPU kernel launch.

```julia
# Build a combined transform: rotate 30° about Z, scale 0.8×, translate (+5, -2, 0)
mat = create_affine_matrix(
    rotation    = (0.0, 0.0, 30.0),   # degrees, applied as Rz * Ry * Rx
    scale       = (0.8, 0.8, 0.8),
    translation = (5.0, -2.0, 0.0),
)

# Single fused interpolation pass (about the image centre by default)
fused = affine_transform_mi(im, mat, Linear_en)

# Compose several matrices into one, or apply a unique matrix per batch element
combined = compose_affine_matrices(mat_a, mat_b)          # applies mat_b, then mat_a
batched  = affine_transform_mi(batch, [mat_a, mat_b], Linear_en)
```

`create_affine_matrix` combines the components in the order `T * R * Sh * S` (points transformed as `M * p`). The fused result (30° Z rotation + 0.8× scale + translation, in a single interpolation pass) is shown as the **Fused affine** panel in both the synthetic and real-CT grids above. Full verification of every operation is in [`docs/VISUAL_VERIFICATION.md`](docs/VISUAL_VERIFICATION.md).

```bash
# Reproduce the fused-affine outputs (synthetic + real CT)
julia --startup-file=no --project=. test/generate_visual_samples.jl   # fused_transform_*.nii.gz
julia --startup-file=no --project=. experiments/make_real_ct_samples.jl # ct_fused.nii.gz
```

---

## Resampling and Orientation

```julia
isotropic = resample_to_spacing(ct, (1.0, 1.0, 1.0), Linear_en)
ras       = change_orientation(ct, ORIENTATION_RAS)
aligned   = resample_to_image(ct, pet, Linear_en)
```

`resample_to_image` aligns a moving image to a fixed image's geometry (grid dimensions, origin, spacing, direction). Essential for multi-modal fusion and deep learning data preparation.

---

## Interpolation

![Interpolation Methods](docs/assets/table-interpolation.png)

Always use `Nearest_neighbour_en` for segmentation masks to preserve label integrity.

---

## Spatial Coordinates

Voxel-to-world mapping: `world = origin + direction * diag(spacing) * (index - 1)`

Orientation codes follow the three-letter anatomical convention. `ORIENTATION_RAS` (NIfTI standard) and `ORIENTATION_LPS` (DICOM standard) are the most common. All eight combinations of R/L, A/P, S/I are supported.

---

## GPU

Backend selection is automatic via KernelAbstractions.jl. The same functions work on CPU and GPU.

```julia
using CUDA
gpu_ct = update_voxel_data(ct, CuArray(Float32.(ct.voxel_data)))
rotated = rotate_mi(gpu_ct, 3, 45.0, Linear_en)
```


---

## Differentiability

All resampling and interpolation operations define ChainRulesCore rrules, enabling end-to-end gradient computation through geometric transforms.

- **Zygote.jl** -- reverse-mode AD, integrates with Flux.jl
- **Enzyme.jl** -- high-performance AD for GPU kernels

```julia
using Zygote
grads = Zygote.gradient(data) do x
    sum(resample_to_spacing(make_medimage(x), (2.0,2.0,2.0), Linear_en).voxel_data)
end
```

---

## Visual Verification & Reproduced Results

Every claim below is reproduced from the scripts in this repository; full instructions live in [`docs/VISUAL_VERIFICATION.md`](docs/VISUAL_VERIFICATION.md). Generated artifacts are in [`test/visual_output/screenshots/`](test/visual_output/screenshots/).

**Rotation vs SimpleITK — pixel-perfect (Pearson 1.0000).** Panels: original, SimpleITK reference, MedImages, and the absolute difference.

![SimpleITK vs MedImages rotation](paper_figures/fig3_rotation_vs_simpleitk.png)

**All spatial operations vs SimpleITK** on a real CT (axial mid-slice, soft-tissue window) — rotate 45°, scale 0.5×, resample 2 mm, crop, pad, and the fused affine (rotate 30° · scale 0.8× · translate). Left: MedImages.jl; middle: SimpleITK on the same grid; right: the absolute difference (Pearson `1.0000` for rotate/scale/crop/pad; `0.9939` resample; `0.9836` fused affine — the small residuals are sub-voxel interpolation differences at edges).

![MedImages vs SimpleITK across all spatial operations](paper_figures/fig14_medimages_vs_simpleitk_allops.png)


```bash
# Julia timings (the plot itself is a saved artifact)
julia --project=experiments/sciml_dose_refinement/ experiments/sciml_dose_refinement/benchmark_speed.jl
```

**MedImages dose vs F-18 dose-point-kernel** on real TCIA FDG PET/CT FDG PET CT was used just to show that code compiles as we can not publish Lu PSMA dataset (body-masked Pearson 0.97). Panels: FDG activity (SUV), MedImages local-deposition dose, DPK reference, signed difference:

![MedImages dose vs DPK reference](test/visual_output/screenshots/Dosimetry_MedImages_vs_DPK.png)

```bash
# MedImages local-deposition dose (Julia)
julia --startup-file=no --project=. experiments/sciml_dose_refinement/medimages_dose.jl
```

---

## API Reference

![API Reference](docs/assets/table-api.png)

---

## Docker

```bash
make build       # Build image
make shell       # Julia REPL with GPU
make shell-cpu   # Julia REPL, CPU only
make test        # Run test suite
make benchmark   # GPU benchmarks (synthetic data)
make help        # All commands
```

Test data goes in `test_data/`. Run `make download-data` for real benchmark data.

---

## Contributing

Contributions are welcome, particularly from those with medical imaging or ultrasonography expertise.

## References

[1] Gorgolewski et al. The brain imaging data structure. Sci Data 3, 160044 (2016). https://www.nature.com/articles/sdata201644
</content>
