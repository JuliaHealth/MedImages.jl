using Pkg;
Pkg.add(["NIfTI","ImageView"])
using NIfTI
using Plots
using ImageView

nifti_file = "./volume-0.nii.gz"
image = NIfTI.niread(nifti_file)
nifti_header = image.header

pixel_data = image.raw
size_of_pixel_data = size(pixel_data)
plotly()
heatmap(pixel_data[:, :, size_of_pixel_data[3]])


