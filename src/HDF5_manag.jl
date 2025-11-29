using HDF5
using JSON
using Dates
using .MedImage_data_struct: MedImage

"""
saving a MedImage object to a HDF5 file into a group with the given name
we use Json to encode the display_data, clinical_data and metadata fields dictionaries to strings so it can be saved as attribute
"""
function save_med_image(f::HDF5.File, group_name::String, image::MedImage)
    if group_name in keys(f)
        g = f[group_name]
    else
        g = create_group(f, group_name)
    end

    # Ensure voxel_data is a standard Array for HDF5 compatibility
    voxel_data_arr = collect(image.voxel_data)
    dset = create_dataset(g, image.study_uid, voxel_data_arr)
    dset = g[image.study_uid]
    write(dset, voxel_data_arr)
    attributes(dset)["origin"] = collect(image.origin)
    attributes(dset)["spacing"] = collect(image.spacing)
    attributes(dset)["direction"] = collect(image.direction)
    attributes(dset)["image_type"] = Int(image.image_type)
    attributes(dset)["image_subtype"] = Int(image.image_subtype)
    attributes(dset)["date_of_saving"] = string(image.date_of_saving)
    attributes(dset)["acquistion_time"] = string(image.acquistion_time)
    attributes(dset)["patient_id"] = image.patient_id
    attributes(dset)["current_device"] = Int(image.current_device)
    attributes(dset)["patient_uid"] = image.patient_uid
    attributes(dset)["series_uid"] = image.series_uid
    attributes(dset)["study_description"] = image.study_description
    attributes(dset)["legacy_file_name"] = image.legacy_file_name
    attributes(dset)["is_contrast_administered"] = image.is_contrast_administered
    attributes(dset)["display_data"] = string(JSON.json(image.display_data))
    attributes(dset)["clinical_data"] = string(JSON.json(image.clinical_data))
    attributes(dset)["metadata"] = string(JSON.json(image.metadata))

    return image.study_uid
end

"""
loading a MedImage object from a HDF5 file from a group with the given name
we use Json to encode the display_data, clinical_data and metadata fields dictionaries to strings so it can be saved as attribute
"""
function load_med_image(f::HDF5.File, group_name::String, dataset_name::String)
    g = f[group_name]
    dset = g[dataset_name]

    voxel_data = read(dset)
    origin = read_attribute(dset, "origin")
    spacing = read_attribute(dset, "spacing")
    direction = read_attribute(dset, "direction")
    image_type = instances(Image_type)[read_attribute(dset, "image_type")+1]
    image_subtype = instances(Image_subtype)[read_attribute(dset, "image_subtype")+1]
    date_of_saving = DateTime(read_attribute(dset, "date_of_saving"))
    acquistion_time = DateTime(read_attribute(dset, "acquistion_time"))
    patient_id = read_attribute(dset, "patient_id")
    current_device = instances(current_device_enum)[read_attribute(dset, "current_device")+1]
    study_uid = dataset_name
    patient_uid = read_attribute(dset, "patient_uid")
    series_uid = read_attribute(dset, "series_uid")
    study_description = read_attribute(dset, "study_description")
    legacy_file_name = read_attribute(dset, "legacy_file_name")
    display_data = JSON.parse(read_attribute(dset, "display_data"))
    clinical_data = JSON.parse(read_attribute(dset, "clinical_data"))
    is_contrast_administered = read_attribute(dset, "is_contrast_administered")
    metadata = JSON.parse(read_attribute(dset, "metadata"))

    return MedImage(voxel_data=voxel_data, origin=ensure_tuple(origin), spacing=ensure_tuple(spacing), direction=ensure_tuple(direction), image_type=image_type, image_subtype=image_subtype, date_of_saving=date_of_saving, acquistion_time=acquistion_time, patient_id=patient_id, current_device=current_device, study_uid=study_uid, patient_uid=patient_uid, series_uid=series_uid, study_description=study_description, legacy_file_name=legacy_file_name, display_data=display_data, clinical_data=clinical_data, is_contrast_administered=is_contrast_administered, metadata=metadata)
end
