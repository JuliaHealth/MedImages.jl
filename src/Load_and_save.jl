module Load_and_save
using Dictionaries, Dates, ZipFile, EzXML
using Accessors, UUIDs, LinearAlgebra, ITKIOWrapper, NIfTI
using ..MedImage_data_struct: MedImage, BatchedMedImage
import ..MedImage_data_struct
using ..Utils
export load_image
export update_voxel_and_spatial_data
export update_voxel_data
export create_nii_from_medimage

"""
helper function for dicom #2
returns an array of pixel data for unique ids within dicom files
"""
function get_pixel_data(dicom_data_array)
  if length(dicom_data_array) == 1
    #in case e have 2D image
    return only(dicom_data_array).PixelData
  else
    #in case we have 3D image
    return cat([dcm.PixelData for dcm in dicom_data_array]...; dims=3)
  end
end

"""
    create_nii_from_medimage(med_image, file_path, extension=".nii.gz")

Save a MedImage object to a NIfTI file natively using ITKIOWrapper.
"""
function create_nii_from_medimage(med_image::MedImage, file_path::String, extension::String=".nii.gz")
  # Ensure path has extension if not already present
  full_path = endswith(file_path, extension) ? file_path : file_path * extension

  # Prepare VoxelData (requires Float32 for ITKIOWrapper)
  voxel_f32 = Array{Float32, 3}(med_image.voxel_data)
  vd = ITKIOWrapper.DataStructs.VoxelData(voxel_f32)

  # Prepare SpatialMetaData
  origin = NTuple{3, Float64}(med_image.origin)
  spacing = NTuple{3, Float64}(med_image.spacing)
  sz = NTuple{3, Int64}(size(med_image.voxel_data))
  direction = NTuple{9, Float64}(med_image.direction)
  
  meta = ITKIOWrapper.DataStructs.SpatialMetaData(origin, spacing, sz, direction)

  # Save using native ITK wrapper
  ITKIOWrapper.save_image(vd, meta, full_path, false)
end

"""
    update_voxel_data(old_image::MedImage, new_voxel_data::AbstractArray)
"""
function update_voxel_data(old_image::MedImage, new_voxel_data::AbstractArray)
  return @set old_image.voxel_data = new_voxel_data
end

function update_voxel_data(old_image::BatchedMedImage, new_voxel_data::AbstractArray)
  return @set old_image.voxel_data = new_voxel_data
end

"""
    update_voxel_and_spatial_data(old_image, new_voxel_data::AbstractArray, new_origin, new_spacing, new_direction=nothing)
"""
function update_voxel_and_spatial_data(old_image, new_voxel_data::AbstractArray, new_origin, new_spacing, new_direction=nothing)
    res = @set old_image.voxel_data = new_voxel_data
    res = @set res.origin = Utils.ensure_tuple(new_origin)
    res = @set res.spacing = Utils.ensure_tuple(new_spacing)
    if !isnothing(new_direction)
        res = @set res.direction = Utils.ensure_tuple(new_direction)
    end
    return res
end


"""
    load_image(path, type)

Load a medical image from file. Uses NIfTI.jl for NIfTI files (.nii, .nii.gz)
and ITKIOWrapper for DICOM directories. NIfTI.jl reads raw voxel data without
any internal caching or spatial resampling, which is critical for binary mask
files that are repeatedly overwritten at the same path.
"""
function load_image(path::String, type::String="CT")::MedImage
  is_nifti = endswith(lowercase(path), ".nii.gz") || endswith(lowercase(path), ".nii")
  
  if is_nifti
    # Use NIfTI.jl for NIfTI files — avoids ITK's internal file caching
    # which can return stale data when a file at the same path is overwritten
    nii = NIfTI.niread(path)
    voxel_arr = Array{Float32}(nii.raw)
    
    # Extract spatial metadata from NIfTI header
    # NIfTI.jl stores the 4x4 affine (sform or qform) in nii.header
    hdr = nii.header
    
    # Get voxel dimensions from pixdim
    pixdim = hdr.pixdim
    spacing = (Float64(abs(pixdim[2])), Float64(abs(pixdim[3])), Float64(abs(pixdim[4])))
    
    # Compute affine matrix from sform rows (srow_x, srow_y, srow_z)
    # NIfTI stores the affine as srow_x/y/z which are 4-element vectors
    sform_code = hdr.sform_code
    qform_code = hdr.qform_code
    
    if sform_code > 0
      # Use sform
      sx = hdr.srow_x
      sy = hdr.srow_y
      sz = hdr.srow_z
      
      origin = (Float64(sx[4]), Float64(sy[4]), Float64(sz[4]))
      
      # Direction is the 3x3 rotation/scale matrix normalized by spacing
      dir = zeros(Float64, 9)
      for i in 1:3
        row = i == 1 ? sx : (i == 2 ? sy : sz)
        for j in 1:3
          dir[(i-1)*3 + j] = Float64(row[j]) / spacing[j]
        end
      end
      direction = NTuple{9, Float64}(dir)
    elseif qform_code > 0
      # Use qform — fall back to ITKIOWrapper for proper quaternion handling
      imaging_study = ITKIOWrapper.load_image(path)
      spatial_meta = ITKIOWrapper.load_spatial_metadata(imaging_study)
      # But use NIfTI.jl voxel data (no caching issues)
      origin = spatial_meta.origin
      direction = spatial_meta.direction
      spacing = spatial_meta.spacing
    else
      # No transform info — use identity
      origin = (0.0, 0.0, 0.0)
      direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    end
    
    # Convert NIfTI RAS to ITK LPS convention
    # NIfTI sform maps voxels to RAS world. ITK uses LPS world.
    # LPS flips X and Y world axes, so negate the first two ROWS entirely.
    lps_origin = (-origin[1], -origin[2], origin[3])
    lps_dir = (
      -direction[1], -direction[2], -direction[3],
      -direction[4], -direction[5], -direction[6],
      direction[7], direction[8], direction[9]
    )
    
  else
    # DICOM or other format — use ITKIOWrapper (no caching issues for DICOM dirs)
    imaging_study = ITKIOWrapper.load_image(path)
    spatial_meta = ITKIOWrapper.load_spatial_metadata(imaging_study)
    voxel_arr_struct = ITKIOWrapper.load_voxel_data(imaging_study, spatial_meta)
    voxel_arr = collect(voxel_arr_struct.dat)
    lps_origin = spatial_meta.origin
    spacing = spatial_meta.spacing
    lps_dir = spatial_meta.direction
  end
  
  study_type = type == "CT" ? MedImage_data_struct.CT_type : MedImage_data_struct.PET_type
  subtype = type == "CT" ? MedImage_data_struct.CT_subtype : MedImage_data_struct.FDG_subtype
  legacy_file_name_field = string(split(path, "/")[length(split(path, "/"))])

  metadata_dict = Dict{Any, Any}()

  return MedImage(voxel_data=voxel_arr, origin=lps_origin, spacing=spacing, direction=lps_dir, patient_id="test_id", image_type=study_type, image_subtype=subtype, legacy_file_name=legacy_file_name_field, metadata=metadata_dict)
end

# DICOM metadata extraction is currently deferred to ITKIOWrapper which is expected
# to be updated in the future to extract full DICOM dictionaries along with spatial metadata.

export load_mrb

"""
    load_mrb(path::String)

Load an MRB file or an extracted MRB directory, parse the MRML file,
and load all Volume and Segmentation nodes. Transforms are parsed and
applied to the MedImage metadata (origin, spacing, direction).

Returns a Dict{String, MedImage} where keys are the node names.
"""
function load_mrb(path::String)
    is_mrb = endswith(lowercase(path), ".mrb") || endswith(lowercase(path), ".zip")
    
    temp_dir = ""
    target_dir = path
    
    if is_mrb
        temp_dir = mktempdir()
        r = ZipFile.Reader(path)
        for f in r.files
            out_path = joinpath(temp_dir, f.name)
            if endswith(f.name, "/")
                mkpath(out_path)
            else
                mkpath(dirname(out_path))
                open(out_path, "w") do io
                    write(io, read(f))
                end
            end
        end
        close(r)
        target_dir = temp_dir
    end
    
    mrml_file = ""
    for (root, dirs, files) in walkdir(target_dir)
        for f in files
            if endswith(lowercase(f), ".mrml")
                mrml_file = joinpath(root, f)
                break
            end
        end
        if mrml_file != ""
            break
        end
    end
    
    if mrml_file == ""
        if temp_dir != ""
            rm(temp_dir, force=true, recursive=true)
        end
        error("No .mrml file found in the provided path/archive.")
    end
    
    doc = readxml(mrml_file)
    root_node = root(doc)
    
    transforms = Dict{String, Dict}()
    for tnode in findall("//LinearTransform", root_node)
        id = tnode["id"]
        matrix_str = haskey(tnode, "matrixTransformToParent") ? tnode["matrixTransformToParent"] : ""
        if matrix_str != ""
            vals = parse.(Float64, split(matrix_str))
            matrix = transpose(reshape(vals, 4, 4))
            
            parent_id = ""
            if haskey(tnode, "references")
                refs = split(tnode["references"], ";")
                for ref in refs
                    if startswith(ref, "transform:")
                        parent_id = split(ref, ":")[2]
                    end
                end
            end
            
            transforms[id] = Dict("matrix" => matrix, "parent" => parent_id)
        end
    end
    
    storages = Dict{String, String}()
    for snode in findall("//VolumeArchetypeStorage", root_node)
        id = snode["id"]
        if haskey(snode, "fileName")
            storages[id] = snode["fileName"]
        end
    end
    for snode in findall("//SegmentationStorage", root_node)
        id = snode["id"]
        if haskey(snode, "fileName")
            storages[id] = snode["fileName"]
        end
    end
    
    results = Dict{String, Any}()
    nodes_to_process = [findall("//Volume", root_node); findall("//Segmentation", root_node)]
    mrml_dir = dirname(mrml_file)
    
    for vnode in nodes_to_process
        name = haskey(vnode, "name") ? vnode["name"] : ""
        if !haskey(vnode, "references")
            continue
        end
        
        refs = split(vnode["references"], ";")
        storage_id = ""
        transform_id = ""
        
        for ref in refs
            if startswith(ref, "storage:")
                storage_id = split(ref, ":")[2]
            elseif startswith(ref, "transform:")
                transform_id = split(ref, ":")[2]
            end
        end
        
        if storage_id != "" && haskey(storages, storage_id)
            rel_file = storages[storage_id]
            full_file = joinpath(mrml_dir, rel_file)
            
            if isfile(full_file)
                itype = occursin(r"(?i)pet", name) ? "PET" : "CT"
                img = load_image(full_file, itype)
                
                current_tid = transform_id
                T_RAS = Matrix{Float64}(LinearAlgebra.I, 4, 4)
                
                while current_tid != "" && haskey(transforms, current_tid)
                    t_info = transforms[current_tid]
                    T_RAS = t_info["matrix"] * T_RAS
                    current_tid = t_info["parent"]
                end
                
                if T_RAS != LinearAlgebra.I
                    L = LinearAlgebra.Diagonal([-1.0, -1.0, 1.0, 1.0])
                    T_LPS = L * T_RAS * L
                    
                    old_spacing = img.spacing
                    old_dir = transpose(reshape(collect(img.direction), 3, 3))
                    old_orig = img.origin
                    
                    M_old = zeros(Float64, 4, 4)
                    for i in 1:3, j in 1:3
                        M_old[i, j] = old_dir[i, j] * old_spacing[j]
                    end
                    for i in 1:3
                        M_old[i, 4] = old_orig[i]
                    end
                    M_old[4, 4] = 1.0
                    
                    M_new = T_LPS * M_old
                    
                    new_orig = (M_new[1, 4], M_new[2, 4], M_new[3, 4])
                    new_spacing = zeros(Float64, 3)
                    new_dir = zeros(Float64, 3, 3)
                    
                    for j in 1:3
                        col = M_new[1:3, j]
                        s = LinearAlgebra.norm(col)
                        new_spacing[j] = s
                        new_dir[:, j] = col / s
                    end
                    
                    new_dir_tuple = Tuple(transpose(new_dir))
                    img = update_voxel_and_spatial_data(img, img.voxel_data, new_orig, Tuple(new_spacing), new_dir_tuple)
                end
                
                results[name] = img
            end
        end
    end
    
    if temp_dir != ""
        rm(temp_dir, force=true, recursive=true)
    end
    
    return results
end


end
