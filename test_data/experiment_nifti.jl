using NIfTI

nifti_file_path  = "/home/hurtbadly/Desktop/julia_stuff/MedImage.jl/MedImage3D/test_data/volume-0.nii.gz"
load_nifti(path::String) = begin
  nifti_image = NIfTI.niread(path)
  nifti_image_header = nifti_image.header 
  return [nifti_image, nifti_image_header]
end

create_volume() = begin
  new_nifti_volume = NIfTI.NIVolume()
  return new_nifti_volume
end

nifti_volume_attributes(data::Array{Any}) = begin
  return [size(data[2],)]
end


#variables in julia repl
nifti_image 
nifti_image_header 
nifti_image_pixel_data_array
nifti_image_pixel_data_array_size 
nifti_image_affine
nifti_image_affine_size 
new_nifti_volume 




important  data for header of the volume
sizeof_hdr 
dim_info
dim
bitpix
pixdim
vox_offset
descrip
#spatial meta data 
qform_code
sform_code
quatern_b
quatern_c
quatenr_d
qoffset_x
qoffset_y
qoffset_z
srow_x
sow_y
srow_z

important data for extension of the volume
affine matrix

important data for the raw of the volume 
pixel data arrays


