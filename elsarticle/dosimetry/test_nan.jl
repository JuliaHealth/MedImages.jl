using Pkg
Pkg.activate("/home/user/MedImages.jl")
using NIfTI
patient_dir = "/home/user/MedImages.jl/elsarticle/dosimetry/data/FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_0__Pat101"
ct_full = niread(joinpath(patient_dir, "ct.nii.gz"))
spect_full = niread(joinpath(patient_dir, "spect.nii.gz"))
dose_full = niread(joinpath(patient_dir, "dosemap_approx.nii.gz"))
println("CT pixdim: ", ct_full.header.pixdim)
println("SPECT pixdim: ", spect_full.header.pixdim)
println("DOSE pixdim: ", dose_full.header.pixdim)
vol_voxel = prod(ct_full.header.pixdim[2:4])
println("vol_voxel: ", vol_voxel)
