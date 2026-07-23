#!/usr/bin/env python3
"""Single grid: MedImages.jl vs SimpleITK across all spatial operations on a real CT.

For each operation the MedImages output defines the reference output geometry;
SimpleITK then resamples the ORIGINAL onto that exact geometry (identity transform
for crop/pad/scale/resample; the rotation transform for rotation). This compares the
two implementations' voxel values on an identical grid — no parameter guessing.
"""
import os, math
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300,
                     "font.family": "DejaVu Sans", "savefig.bbox": "tight"})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "paper_figures")
SRC = os.path.join(ROOT, "test", "visual_output", "real_ct")  # from make_real_ct_samples.jl
os.makedirs(OUT, exist_ok=True)
ORIG = os.path.join(SRC, "ct_original.nii.gz")
AIR = -1000.0


def ctwin(a, lo=-160, hi=240):
    return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _affine_matrix(translation=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1),
                   shear=(0, 0, 0)):
    """Replica of MedImages.create_affine_matrix: M = T * R * Sh * S (deg, Rz*Ry*Rx)."""
    tx, ty, tz = translation
    T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]], float)
    rx, ry, rz = np.deg2rad(rotation)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = np.eye(4); R[:3, :3] = Rz @ Ry @ Rx
    S = np.diag([*scale, 1.0])
    sxy, sxz, syz = shear
    Sh = np.array([[1, sxy, sxz, 0], [0, 1, syz, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    return T @ R @ Sh @ S


def fused_affine_transform(ref, M):
    """SITK transform (output_phys -> input_phys) reproducing MedImages'
    affine_transform_mi, which applies M in INDEX space about c = size/2 (1-based)
    and ignores spacing/origin/direction. create_nii permutes [x,y,z]->[z,y,x],
    so MedImages array dims map 1:1 to SITK (x,y,z)."""
    size = np.array(ref.GetSize(), float)
    O = np.array(ref.GetOrigin(), float)
    L = np.array(ref.GetDirection(), float).reshape(3, 3) @ np.diag(ref.GetSpacing())
    Minv = np.linalg.inv(M)
    A, tinv = Minv[:3, :3], Minv[:3, 3]
    cc = size / 2.0 - 1.0                       # 0-based center matching size/2 (1-based)
    t_idx = cc - A @ cc + tinv
    aff = sitk.AffineTransform(3)
    aff.SetMatrix((L @ A @ np.linalg.inv(L)).flatten().tolist())
    aff.SetCenter(O.tolist())
    aff.SetTranslation((L @ t_idx).tolist())
    return aff


def sitk_ref(original, med_img, transform):
    """Resample original onto med_img's geometry with the given transform."""
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(med_img)
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(AIR)
    rs.SetTransform(transform)
    return rs.Execute(original)


def main():
    original = sitk.ReadImage(ORIG)
    c = original.GetSize()
    rot = sitk.Euler3DTransform()
    rot.SetCenter(original.TransformIndexToPhysicalPoint([c[0] // 2, c[1] // 2, c[2] // 2]))
    rot.SetRotation(0, 0, math.radians(-45.0))  # SimpleITK transform convention -> +45 CCW image

    # scale_mi(0.5) samples original at index 2*(i-1)+1 -> scale about the origin by 1/0.5=2
    scl = sitk.ScaleTransform(3, [2.0, 2.0, 1.0])
    scl.SetCenter(original.GetOrigin())

    # Fused affine: rotate 30° about Z, scale 0.8×, translate (+5, -2, 0) — the
    # same single-pass transform shown in the synthetic/real-CT grids.
    fused = fused_affine_transform(original, _affine_matrix(
        rotation=(0, 0, 30), scale=(0.8, 0.8, 0.8), translation=(5, -2, 0)))

    ops = [
        ("Rotate 45°",      os.path.join(SRC, "ct_rotate45.nii.gz"),    rot),
        ("Scale 0.5×",      os.path.join(SRC, "ct_scale05.nii.gz"),     scl),
        ("Resample 2 mm",   os.path.join(SRC, "ct_resample2mm.nii.gz"), sitk.Transform()),
        ("Crop 256×256×40", os.path.join(SRC, "ct_crop.nii.gz"),        sitk.Transform()),
        ("Pad +40/+5",      os.path.join(SRC, "ct_pad.nii.gz"),         sitk.Transform()),
        ("Fused affine\n(rot 30°·0.8×·+5/−2)", os.path.join(SRC, "ct_fused.nii.gz"), fused),
    ]

    n = len(ops)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.4 * n), constrained_layout=True)
    for r, (name, path, T) in enumerate(ops):
        med = sitk.ReadImage(path)
        ref = sitk_ref(original, med, T)
        a_med = sitk.GetArrayFromImage(med)
        a_ref = sitk.GetArrayFromImage(ref)
        z = a_med.shape[0] // 2
        m, rf = ctwin(a_med[z]), ctwin(a_ref[z])
        diff = np.abs(m - rf)
        fg = (a_med > -900) | (a_ref > -900)
        pear = np.corrcoef(ctwin(a_med[fg]).ravel(), ctwin(a_ref[fg]).ravel())[0, 1]

        axes[r, 0].imshow(np.rot90(m), cmap="gray", aspect="equal", interpolation="bilinear")
        axes[r, 0].set_ylabel(name, fontsize=13, fontweight="bold")
        axes[r, 1].imshow(np.rot90(rf), cmap="gray", aspect="equal", interpolation="bilinear")
        im = axes[r, 2].imshow(np.rot90(diff), cmap="inferno", vmin=0, vmax=1, aspect="equal", interpolation="bilinear")
        axes[r, 2].set_title(f"|diff|   Pearson = {pear:.4f}", fontsize=11)
        for c2 in range(3):
            axes[r, c2].set_xticks([]); axes[r, c2].set_yticks([])
        fig.colorbar(im, ax=axes[r, 2], fraction=0.046, pad=0.03)
        print(f"{name:18s} pearson={pear:.4f}  med{a_med.shape} ref{a_ref.shape}")

    axes[0, 0].set_title("MedImages.jl", fontsize=14, fontweight="bold")
    axes[0, 1].set_title("SimpleITK (same grid)", fontsize=14, fontweight="bold")
    fig.suptitle("MedImages.jl vs SimpleITK across spatial operations (real CT, axial mid-slice, soft-tissue window)",
                 fontsize=15, fontweight="bold")
    out = os.path.join(OUT, "fig14_medimages_vs_simpleitk_allops.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
