include("./MedImage_data_struct.jl")
include("./Utils.jl")
include("./Load_and_save.jl")
using HDF5

"""
saving a MedImage object to a HDF5 file into a group with the given name
"""
function save_med_image(f::HDF5File, group_name::String, image::MedImage)
    if !exists(f, group_name)
        g = create_group(f, group_name)
    else
        g = f[group_name]
    end

    dset = create_dataset(g, image.study_uid, image.voxel_data)
    attributes(dset)["origin"] = image.origin
    attributes(dset)["spacing"] = image.spacing
    attributes(dset)["direction"] = image.direction
    attributes(dset)["image_type"] = string(image.image_type)
    attributes(dset)["image_subtype"] = string(image.image_subtype)
    attributes(dset)["voxel_datatype"] = typeof(image.voxel_data)
    attributes(dset)["date_of_saving"] = image.date_of_saving
    attributes(dset)["acquistion_time"] = image.acquistion_time
    attributes(dset)["patient_id"] = image.patient_id
    attributes(dset)["current_device"] = image.current_device
    attributes(dset)["patient_uid"] = image.patient_uid
    attributes(dset)["series_uid"] = image.series_uid
    attributes(dset)["study_description"] = image.study_description
    attributes(dset)["legacy_file_name"] = image.legacy_file_name
    attributes(dset)["display_data"] = image.display_data
    attributes(dset)["clinical_data"] = image.clinical_data
    attributes(dset)["is_contrast_administered"] = image.is_contrast_administered
    attributes(dset)["metadata"] = image.metadata
end

"""
loading a MedImage object from a HDF5 file from a group with the given name
"""
function load_med_image(f::HDF5File, group_name::String, dataset_name::String)
    g = f[group_name]
    dset = g[dataset_name]

    voxel_data = read(dset)
    origin = attributes(dset)["origin"]
    spacing = attributes(dset)["spacing"]
    direction = attributes(dset)["direction"]
    image_type = Symbol(attributes(dset)["image_type"])
    image_subtype = Symbol(attributes(dset)["image_subtype"])
    date_of_saving = attributes(dset)["date_of_saving"]
    acquistion_time = attributes(dset)["acquistion_time"]
    patient_id = attributes(dset)["patient_id"]
    current_device = attributes(dset)["current_device"]
    study_uid = dataset_name
    patient_uid = attributes(dset)["patient_uid"]
    series_uid = attributes(dset)["series_uid"]
    study_description = attributes(dset)["study_description"]
    legacy_file_name = attributes(dset)["legacy_file_name"]
    display_data = attributes(dset)["display_data"]
    clinical_data = attributes(dset)["clinical_data"]
    is_contrast_administered = attributes(dset)["is_contrast_administered"]
    metadata = attributes(dset)["metadata"]

    return MedImage(voxel_data, origin, spacing, direction, image_type, image_subtype, typeof(voxel_data), date_of_saving, acquistion_time, patient_id, current_device, study_uid, patient_uid, series_uid, study_description, legacy_file_name, display_data, clinical_data, is_contrast_administered, metadata)
end