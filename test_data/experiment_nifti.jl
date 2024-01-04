using NIfTI

nifti_file_path = "/home/hurtbadly/Desktop/julia_stuff/MedImage.jl/test_data/volume-0.nii.gz"

img = niread(nifti_file_path)
header = img.header

function data_for_medimage(img, header)
  return [img.raw, "", "", header.regular, header.extents, header.dim_info, header.dim, header.data_type,
    header.bitpix, header.descrip,
    header.sizeof_hdr, header.pixdim, header.vox_offset,
    header.intent_p1, header.intent_p2, header.intent_p3, header.intent_code, header.slice_start, header.slice_end, header.slice_duration, header.slice_code,
    header.toffset, header.xyzt_units, header.cal_max, header.cal_min, header.glmax, header.glmin,
    header.qform_code, header.sform_code, header.quatern_b, header.quatern_c, header.quatern_d,
    header.qoffset_x, header.qoffset_y, header.qoffset_z, header.srow_x, header.srow_y, header.srow_z,
    header.scl_slope, header.scl_inter, header.intent_name, header.magic, img.extensions, header.db_name, header.session_error
  ]
end

value_array = data_for_medimage(img, header)
struct MedImage
  #some important attributes related to medical imaging 

  pixel_data
  date_of_saving::String
  patient_id::String

  #other data for nifti header
  regular
  extents
  dim_info::Int8
  dim::NTuple{8,Int16}
  data_type::NTuple{10,UInt8}
  bitpix::Int16
  descrip::NTuple{80,UInt8}
  sizeof_hdr::Int32
  pixdim::NTuple{8,Float32}
  vox_offset::Float32
  intent_p1
  intent_p2
  intent_p3
  intent_code::Int16
  slice_start::Int16
  slice_end::Int16
  slice_duration::Float32
  slice_code::Int8
  toffset::Float32
  xyzt_units::Int8
  cal_max
  cal_min
  glmax
  glmin
  #spatial metadata for nifti header
  qform_code::Int16
  sform_code::Int16
  quatern_b::Float32
  quatern_c::Float32
  quatern_d::Float32
  qoffset_x::Float32
  qoffset_y::Float32
  qoffset_z::Float32
  srow_x::NTuple{4,Float32}
  srow_y::NTuple{4,Float32}
  srow_z::NTuple{4,Float32}
  scl_slope
  scl_inter
  intent_name

  magic
  ext
  db_name
  session_error
end

object_one = MedImage(value_array...)


function save_file(object)
  new_volume = NIfTI.NIVolume(object.pixel_data)

  new_volume.header.data_type = object.data_type
  new_volume.header.bitpix = object.bitpix
  new_volume.header.descrip = object.descrip
  new_volume.header.sizeof_hdr = object.sizeof_hdr
  new_volume.header.pixdim = object.pixdim
  new_volume.header.vox_offset = object.vox_offset
  new_volume.header.intent_p1 = object.intent_p1
  new_volume.header.intent_p2 = object.intent_p2
  new_volume.header.intent_p3 = object.intent_p3
  new_volume.header.intent_code = object.intent_code
  new_volume.header.slice_start = object.slice_start
  new_volume.header.slice_end = object.slice_end
  new_volume.header.slice_duration = object.slice_duration
  new_volume.header.slice_code = object.slice_code
  new_volume.header.toffset = object.toffset
  new_volume.header.xyzt_units = object.xyzt_units
  new_volume.header.cal_max = object.cal_max
  new_volume.header.cal_min = object.cal_min
  new_volume.header.glmax = object.glmax
  new_volume.header.glmin = object.glmin
  new_volume.header.extents = object.extents
  new_volume.header.regular = object.regular
  new_volume.header.dim_info = object.dim_info
  new_volume.header.dim = object.dim

  new_volume.header.qform_code = object.qform_code
  new_volume.header.sform_code = object.sform_code
  new_volume.header.quatern_b = object.quatern_b
  new_volume.header.quatern_c = object.quatern_c
  new_volume.header.quatern_d = object.quatern_d
  new_volume.header.qoffset_x = object.qoffset_x
  new_volume.header.qoffset_y = object.qoffset_y
  new_volume.header.qoffset_z = object.qoffset_z
  new_volume.header.srow_x = object.srow_x
  new_volume.header.srow_y = object.srow_y
  new_volume.header.srow_z = object.srow_z
  new_volume.header.scl_slope = object.scl_slope
  new_volume.header.scl_inter = object.scl_inter
  new_volume.header.intent_name = object.intent_name
  new_volume.header.magic = object.magic
  new_volume.extensions = object.ext
  new_volume.header.session_error = object.session_error
  new_volume.header.db_name = object.db_name
  niwrite("./outputted_three.nii.gz", new_volume)
end

save_file(object_one)
