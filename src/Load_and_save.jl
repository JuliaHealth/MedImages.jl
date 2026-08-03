module Load_and_save
using Dictionaries, Dates, PyCall
using Accessors, UUIDs, LinearAlgebra
using ..MedImage_data_struct
using ..MedImage_data_struct: MedImage, BatchedMedImage
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

Save a MedImage object to a NIfTI file using SimpleITK via PyCall.
"""
function create_nii_from_medimage(med_image::MedImage, file_path::String, extension::String=".nii.gz")
  sitk = pyimport("SimpleITK")
  np = pyimport("numpy")
  
  full_path = endswith(file_path, extension) ? file_path : file_path * extension

  # Permute from [x, y, z] to [z, y, x] for SimpleITK
  voxel_arr = permutedims(Float32.(med_image.voxel_data), (3, 2, 1))
  img = sitk.GetImageFromArray(np.array(voxel_arr))
  
  img.SetOrigin(collect(Float64.(med_image.origin)))
  img.SetSpacing(collect(Float64.(med_image.spacing)))
  img.SetDirection(collect(Float64.(med_image.direction)))

  sitk.WriteImage(img, full_path)
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

Load a medical image from file using SimpleITK via PyCall.
"""
function load_image(path::String, type::String)::MedImage
  sitk = pyimport("SimpleITK")
  
  img = nothing
  if isdir(path)
      # Assume DICOM series
      reader = sitk.ImageSeriesReader()
      dicom_names = reader.GetGDCMSeriesFileNames(path)
      reader.SetFileNames(dicom_names)
      img = reader.Execute()
  else
      img = sitk.ReadImage(path)
  end
  
  # SimpleITK image to array returns [z, y, x]
  voxel_arr_np = sitk.GetArrayFromImage(img)
  # Permute to [x, y, z] for MedImages.jl consistency
  voxel_arr = permutedims(Float32.(voxel_arr_np), (3, 2, 1))
  
  origin = Tuple(Float64.(img.GetOrigin()))
  spacing = Tuple(Float64.(img.GetSpacing()))
  direction = Tuple(Float64.(img.GetDirection()))
  
  study_type = type == "CT" ? MedImage_data_struct.CT_type : MedImage_data_struct.PET_type
  subtype = type == "CT" ? MedImage_data_struct.CT_subtype : MedImage_data_struct.FDG_subtype
  legacy_file_name_field = string(split(path, "/")[length(split(path, "/"))])

  metadata_dict = Dict{Any, Any}()
  try
      metadata_dict = _get_metadata(path)
  catch e
      @warn "Could not extract DICOM metadata using pydicom: $e"
  end

  return MedImage(voxel_data=voxel_arr, origin=origin, spacing=spacing, direction=direction, patient_id="test_id", image_type=study_type, image_subtype=subtype, legacy_file_name=legacy_file_name_field, metadata=metadata_dict)
end

function _get_metadata(path::String)
    pydicom = pyimport("pydicom")
    os = pyimport("os")

    target_path = path
    if isdir(path)
        # Find first DICOM file in directory
        files = readdir(path)
        # Try to find .dcm extension
        dcm_files = filter(f -> endswith(lowercase(f), ".dcm"), files)
        if !isempty(dcm_files)
            target_path = joinpath(path, dcm_files[1])
        else
            # If no .dcm, try first file that is not a directory
            for f in files
                full_p = joinpath(path, f)
                if isfile(full_p)
                    target_path = full_p
                    break
                end
            end
        end
    end

    if !isfile(target_path)
        return Dict{Any, Any}()
    end

    try
        ds = pydicom.dcmread(target_path, stop_before_pixels=true)
        return _pydicom_ds_to_dict(ds)
    catch e
        # If not a DICOM file or other error
        return Dict{Any, Any}()
    end
end

function _pydicom_ds_to_dict(ds)
    d = Dict{Any, Any}()
    for elem in ds
        key = elem.keyword
        if isempty(key)
            # Use tag if keyword is empty (e.g. private tags)
            key = string(elem.tag)
        end

        val = elem.value

        # Handle Sequences
        if elem.VR == "SQ"
            seq_list = []
            for item in val
                push!(seq_list, _pydicom_ds_to_dict(item))
            end
            d[key] = seq_list
        # Handle MultiValue
        elseif typeof(val) <: PyObject && pybuiltin("isinstance")(val, pyimport("pydicom.multival").MultiValue)
             d[key] = collect(val)
        else
             d[key] = val
        end
    end
    return d
end

export load_mrb

"""
    load_mrb(path::String)

Load an MRB file or an extracted MRB directory, parse the MRML file,
and load all Volume and Segmentation nodes. Transforms are parsed and
applied to the MedImage metadata (origin, spacing, direction).

Returns a Dict{String, MedImage} where keys are the node names.
"""
function load_mrb(path::String)
    pyzip = pyimport("zipfile")
    pytemp = pyimport("tempfile")
    pyos = pyimport("os")
    pyxml = pyimport("xml.etree.ElementTree")
    
    is_mrb = endswith(lowercase(path), ".mrb") || endswith(lowercase(path), ".zip")
    
    temp_dir = ""
    target_dir = path
    
    if is_mrb
        temp_dir = pytemp.mkdtemp()
        with(pyzip.ZipFile(path, "r")) do zip_ref
            zip_ref.extractall(temp_dir)
        end
        target_dir = temp_dir
    end
    
    # Find .mrml file
    mrml_file = ""
    for (root, dirs, files) in pyos.walk(target_dir)
        for f in files
            if endswith(lowercase(f), ".mrml")
                mrml_file = pyos.path.join(root, f)
                break
            end
        end
        if mrml_file != ""
            break
        end
    end
    
    if mrml_file == ""
        if temp_dir != ""
            pyimport("shutil").rmtree(temp_dir)
        end
        error("No .mrml file found in the provided path/archive.")
    end
    
    tree = pyxml.parse(mrml_file)
    root_node = tree.getroot()
    
    # 1. Parse transforms
    transforms = Dict{String, Dict}()
    for tnode in root_node.findall("LinearTransform")
        id = tnode.attrib["id"]
        matrix_str = get(tnode.attrib, "matrixTransformToParent", "")
        if matrix_str != ""
            vals = parse.(Float64, split(matrix_str))
            # Slicer matrix is row-major 16-element array. Reshape to 4x4.
            # Julia reshape is column-major, so we transpose.
            matrix = transpose(reshape(vals, 4, 4))
            
            # Check for parent transform
            parent_id = ""
            if haskey(tnode.attrib, "references")
                refs = split(tnode.attrib["references"], ";")
                for ref in refs
                    if startswith(ref, "transform:")
                        parent_id = split(ref, ":")[2]
                    end
                end
            end
            
            transforms[id] = Dict("matrix" => matrix, "parent" => parent_id)
        end
    end
    
    # 2. Parse storages (for file names)
    storages = Dict{String, String}()
    for snode in root_node.findall("VolumeArchetypeStorage")
        id = snode.attrib["id"]
        if haskey(snode.attrib, "fileName")
            storages[id] = snode.attrib["fileName"]
        end
    end
    for snode in root_node.findall("SegmentationStorage")
        id = snode.attrib["id"]
        if haskey(snode.attrib, "fileName")
            storages[id] = snode.attrib["fileName"]
        end
    end
    
    # 3. Parse Volumes and Segmentations
    results = Dict{String, Any}()
    
    nodes_to_process = [root_node.findall("Volume"); root_node.findall("Segmentation")]
    
    mrml_dir = pyos.path.dirname(mrml_file)
    
    for vnode in nodes_to_process
        name = get(vnode.attrib, "name", "")
        if !haskey(vnode.attrib, "references")
            continue
        end
        
        refs = split(vnode.attrib["references"], ";")
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
            full_file = pyos.path.join(mrml_dir, rel_file)
            
            if pyos.path.exists(full_file)
                # Load the raw MedImage
                itype = occursin(r"(?i)pet", name) ? "PET" : "CT"
                img = load_image(full_file, itype)
                
                # Resolve transform chain
                current_tid = transform_id
                T_RAS = Matrix{Float64}(LinearAlgebra.I, 4, 4)
                
                while current_tid != "" && haskey(transforms, current_tid)
                    t_info = transforms[current_tid]
                    # Post-multiply since parent transform is applied after child
                    T_RAS = t_info["matrix"] * T_RAS
                    current_tid = t_info["parent"]
                end
                
                if T_RAS != LinearAlgebra.I
                    # Apply transform. MedImage is in LPS. T_RAS is in RAS.
                    L = LinearAlgebra.Diagonal([-1.0, -1.0, 1.0, 1.0])
                    T_LPS = L * T_RAS * L
                    
                    # Construct current LPS affine
                    old_spacing = img.spacing
                    old_dir = reshape(collect(img.direction), 3, 3)
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
                    
                    img = update_voxel_and_spatial_data(img, img.voxel_data, new_orig, Tuple(new_spacing), Tuple(new_dir))
                end
                
                results[name] = img
            end
        end
    end
    
    if temp_dir != ""
        pyimport("shutil").rmtree(temp_dir)
    end
    
    return results
end

end
