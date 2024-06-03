using  LinearAlgebra, DICOM,HDF5,Test
include("../src/Load_and_save.jl")
# include("../src/Basic_transformations.jl")
# include("./test_visualize.jl")
include("./dicom_nifti.jl")
include("../src/HDF5_manag.jl")

path_nifti = "/home/jm/projects_new/MedImage.jl/test_data/volume-0.nii.gz"
med_im = load_image(path_nifti)

h5_path="/home/jm/projects_new/MedImage.jl/test_data/debug.h5"
f = h5open(h5_path, "w")
typeof(f)
uid=save_med_image(f,"aa",med_im)
med_im_2=load_med_image(f,"aa",uid)

@test med_im.voxel_data==med_im_2.voxel_data
