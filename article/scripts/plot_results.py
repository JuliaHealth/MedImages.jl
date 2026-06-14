import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def plot_results():
    try:
        # Load images
        gold_img = nib.load("gold_standard.nii.gz").get_fdata()
        uncorrected_img = nib.load("uncorrected.nii.gz").get_fdata()
        reconstructed_img = nib.load("reconstructed.nii.gz").get_fdata()

        # Plot middle slice
        z = gold_img.shape[2] // 2

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(gold_img[:, :, z], cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("Gold Standard")
        axes[0].axis('off')

        axes[1].imshow(uncorrected_img[:, :, z], cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Uncorrected (Rotated)")
        axes[1].axis('off')

        axes[2].imshow(reconstructed_img[:, :, z], cmap='gray', vmin=0, vmax=1)
        axes[2].set_title("Reconstructed (Learned Inverse)")
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig("/home/user/MedImages.jl/docs/src/experiments/viz/differentiability_results.png")
        print("Successfully saved differentiability_results.png")
    except Exception as e:
        print(f"Error plotting: {e}")

if __name__ == "__main__":
    plot_results()
