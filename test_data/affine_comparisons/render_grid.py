import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = "/workspaces/MedImages.jl/test_data/affine_comparisons"

experiments = [
    "01_pure_rotation",
    "02_pure_translation",
    "03_pure_shear",
    "04_composite"
]

fig, axes = plt.subplots(4, 3, figsize=(15, 20))
fig.suptitle('MedImages vs SimpleITK - Affine Resampling Transformations', fontsize=16)

def get_middle_slice(img):
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 3:
        # Get middle axial slice
        return arr[arr.shape[0] // 2, :, :]
    return arr

for i, exp in enumerate(experiments):
    mi_path = os.path.join(OUT_DIR, f"medimages_{exp}.nii.gz")
    sitk_path = os.path.join(OUT_DIR, f"sitk_{exp}.nii.gz")
    
    mi_img = sitk.ReadImage(mi_path)
    sitk_img = sitk.ReadImage(sitk_path)
    
    mi_slice = get_middle_slice(mi_img)
    sitk_slice = get_middle_slice(sitk_img)
    
    diff = np.abs(mi_slice - sitk_slice)
    
    axes[i, 0].imshow(mi_slice, cmap='gray')
    axes[i, 0].set_title(f"MedImages: {exp}")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(sitk_slice, cmap='gray')
    axes[i, 1].set_title(f"SimpleITK: {exp}")
    axes[i, 1].axis('off')
    
    # Calculate difference magnitude 
    diff_plot = axes[i, 2].imshow(diff, cmap='hot')
    axes[i, 2].set_title(f"Difference (Max: {diff.max():.1f})")
    axes[i, 2].axis('off')
    plt.colorbar(diff_plot, ax=axes[i, 2])

plt.tight_layout()
plt.savefig("/workspaces/MedImages.jl/test_data/affine_comparisons_grid.png", dpi=150, bbox_inches='tight')
print("Saved grid to /workspaces/MedImages.jl/test_data/affine_comparisons_grid.png")
