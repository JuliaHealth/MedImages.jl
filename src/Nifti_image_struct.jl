# #defining a nifti image struct for loading nifti data with important fields
# # n stuff and a few of d stuff might be for Analyze files
# using NIfTI



# struct Nifti_image_io
#   #number_of_dimensions
#   rescale_slope
#   rescale_intercept
#   must_rescale
# end

# struct Nifti_image
#   ndim
#   #dimension of grid array begin
#   nx
#   ny
#   nz
#   nt
#   nu
#   nv
#   nw
#   #dimension of grid array end

#   dim
#   nvox
#   #nbyper #bytes per voxel dont reall need that 
#   datatype #type of data in voxels

#   #grid spacings begin
#   dx
#   dy
#   dz
#   dt
#   du
#   dv
#   dw
#   #grid spacing end

#   pixdim

#   scl_slope #scaling parameter : slope
#   scl_inter #scaling parameter : intercept

#   cal_min #calibration parameter minimum
#   cal_max#calibration parameter maximum

#   qform_code #codes for (x,y,z) space meaning
#   sform_code #codes for (x,y,z) space meaning 

#   freq_dim #indexes (1,2,3 or 0) for MRI code : dim_info & 0x03
#   phase_dim #directions in dim/pixdim code : (dim_info >> 2) &0x03
#   slice_dim #directions in dim/pixdim code : dim_info >>4 

#   slice_code #code for slice timing pattern
#   slice_start #index for start of slices
#   slice_end #index for end of slices
#   slice_duration #time between individual slices

#   #quaternion transform parameters 

#   quatern_b
#   quatern_c
#   quatern_d
#   qoffset_x
#   qoffset_y
#   qoffset_z
#   qfac
#   #when writing to a nifti file dataset , these are used for qform not qto_xyz

#   qto_xyz #qform transform (i,j,k) to (x,y,z)
#   qto_ijk #qform transform (x,y,z) to (i,j,k)

#   #only if sform_code > 0
#   sto_xyz #sform transform (i,j,k) to (x,y,z)
#   sto_ijk #sform transform (x,y,z) to (i,j,k)

#   toffset #time coordinate offset
#   xyz_units #dx, dy, dz units 
#   time_units #dt unit
#   nifti_type #only 1-File nifti 

#   intent_code
#   intent_p1
#   intent_p2
#   intent_p3
#   intent_name::String #optional description of intent data 

#   #fname for saving the file name of the nii file

#   descrip::String #optional text to describe dataset
#   aux_file::String #auxiliary filename

#   data #iamge voxel data 

#   num_ext #number of extensions in ext list 
#   ext_list #

#   nifti_image_io_information

# end

# function Nifti_image(nifti_image_field_values::Array{Any})::Nifti_image
#   return Nifti_image(nifti_image_field_values...)
# end
