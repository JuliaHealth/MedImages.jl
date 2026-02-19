import SimpleITK as sitk
import numpy as np

def compare_metadata(file1, file2):
    img1 = sitk.ReadImage(file1)
    img2 = sitk.ReadImage(file2)
    
    print(f"Comparing {file1} and {file2}:")
    print(f"Origin - {file1}: {img1.GetOrigin()}, {file2}: {img2.GetOrigin()}")
    print(f"Spacing - {file1}: {img1.GetSpacing()}, {file2}: {img2.GetSpacing()}")
    print(f"Direction - {file1}: {img1.GetDirection()}, {file2}: {img2.GetDirection()}")
    print(f"Size - {file1}: {img1.GetSize()}, {file2}: {img2.GetSize()}")
    
    # Check if they are nearly identical
    origin_ok = np.allclose(img1.GetOrigin(), img2.GetOrigin(), atol=1e-5)
    spacing_ok = np.allclose(img1.GetSpacing(), img2.GetSpacing(), atol=1e-5)
    direction_ok = np.allclose(img1.GetDirection(), img2.GetDirection(), atol=1e-5)
    size_ok = img1.GetSize() == img2.GetSize()
    
    print(f"Origin Match: {origin_ok}")
    print(f"Spacing Match: {spacing_ok}")
    print(f"Direction Match: {direction_ok}")
    print(f"Size Match: {size_ok}")

if __name__ == "__main__":
    compare_metadata("juliasuv.nii.gz", "pythonsuv.nii.gz")
