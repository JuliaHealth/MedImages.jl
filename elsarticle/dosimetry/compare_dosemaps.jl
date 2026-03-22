using Pkg
Pkg.activate("/home/user/MedImages.jl")

using NIfTI
using Statistics

pat_dir = "/DATA/FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_2__Pat54"
mc_path = joinpath(pat_dir, "DOSEMAP", "dosemap.nii.gz")
approx_path = joinpath(pat_dir, "SPECT_DATA", "nifti_files", "Dosemap.nii.gz")

mc_nii = niread(mc_path)
approx_nii = niread(approx_path)

ct_path = joinpath(pat_dir, "SPECT_DATA", "nifti_files", "CT.nii.gz")
spect_path = joinpath(pat_dir, "SPECT_DATA", "nifti_files", "NM_Vendor.nii.gz")
ct_nii = niread(ct_path)
spect_nii = niread(spect_path)

println("\nCT (SPECT_DATA/nifti_files/CT.nii.gz):")
println("  Shape: ", size(ct_nii))
println("\nSPECT (SPECT_DATA/nifti_files/NM_Vendor.nii.gz):")
println("  Shape: ", size(spect_nii))

println("Monte Carlo (DOSEMAP/dosemap.nii.gz):")
println("  Shape: ", size(mc_nii))
println("  Pixdim: ", mc_nii.header.pixdim[2:4])
mc_data = mc_nii.raw
println("  Sum: ", sum(mc_data))
println("  Max: ", maximum(mc_data))
println("  Mean: ", mean(mc_data))

println("\nApproximate (SPECT_DATA/nifti_files/Dosemap.nii.gz):")
println("  Shape: ", size(approx_nii))
println("  Pixdim: ", approx_nii.header.pixdim[2:4])
approx_data = approx_nii.raw
println("  Sum: ", sum(approx_data))
println("  Max: ", maximum(approx_data))
println("  Mean: ", mean(approx_data))

if size(mc_data) == size(approx_data)
    mc_flat = vec(mc_data)
    approx_flat = vec(approx_data)
    corr = cor(mc_flat, approx_flat)
    println("\nCorrelation: ", corr)
else
    println("\nShapes do not match, cannot compute direct correlation without resampling.")
end
