using PyCall, Pkg

sitk = pyimport_conda("SimpleITK", "simpleitk")
np = pyimport("numpy")

"""
get_spatial_metadata(image_path::String)::MedImage

Funkcja wczytuje obraz z podanej ścieżki i ekstrahuje jego podstawowe metadane przestrzenne.
Zwraca obiekt MedImage zawierający te metadane oraz dane obrazu.
"""
function get_spatial_metadata(image_path::String)::MedImage
    image = sitk.ReadImage(image_path)
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    voxel_data = sitk.GetArrayFromImage(image)
   
    med_image = MedImage(
        voxel_data,  # Przykład pustej wielowymiarowej tablicy
        origin,  # Pusty Tuple dla origin
        spacing,  # Pusty Tuple dla spacing
        direction,  # Pusty Tuple dla direction
        Dictionary(),  # Pusty słownik
        MRI,  # Założenie, że Image_type to Enum z wartością MRI jako domyślną
        subtypes,  # Założenie, że Image_subtype to Enum z wartością subtypes jako domyślną; upewnij się, że to ma sens w kontekście twojego kodu
        typeof(1.0),  # Przykład typu danych; dostosuj do swoich potrzeb
        "", "", "", "CPU", "", "", "", "", "", "", Dictionary(), false, Dictionary()
      )
    return med_image
end


"""
create_nii_from_medimage(med_image::MedImage, file_path::String)

Create a .nii.gz file from a MedImage object and save it to the given file path.
"""
function create_nii_from_medimage(med_image::MedImage, file_path::String)
    # Convert voxel_data to a numpy array (Assuming voxel_data is stored in Julia array format)
    voxel_data_np = np.array(med_image.voxel_data)
    
    # Create a SimpleITK image from numpy array
    image_sitk = sitk.GetImageFromArray(voxel_data_np)
    
    # Set spatial metadata
    image_sitk.SetOrigin(med_image.origin)
    image_sitk.SetSpacing(med_image.spacing)
    image_sitk.SetDirection(med_image.direction)
    
    # Save the image as .nii.gz
    sitk.WriteImage(image_sitk, file_path* ".nii.gz")
end


function update_voxel_data(old_image::MedImage, new_voxel_data::AbstractArray)
  
    return MedImage(
        new_voxel_data, 
        old_image.origin, 
        old_image.spacing, 
        old_image.direction, 
        old_image.spatial_metadata, 
        old_image.image_type, 
        old_image.image_subtype, 
        old_image.voxel_datatype, 
        old_image.date_of_saving, 
        old_image.acquistion_time, 
        old_image.patient_id, 
        old_image.current_device, 
        old_image.study_uid, 
        old_image.patient_uid, 
        old_image.series_uid, 
        old_image.study_description, 
        old_image.legacy_file_name, 
        old_image.display_data, 
        old_image.clinical_data, 
        old_image.is_contrast_administered, 
        old_image.metadata)

image_3D=get_spatial_metadata("C:\\MedImage\\MedImage.jl\\test_data\\volume-0.nii.gz")
image_path_3D = "C:\\MedImage\\MedImage.jl\\test_data\\volume-0.nii.gz"
image_test_3D = sitk.ReadImage(image_path_3D)
image_4D=get_spatial_metadata("C:\\MedImage\\MedImage.jl\\test_data\\filtered_func_data.nii.gz")
image_path_4D = "C:\\MedImage\\MedImage.jl\\test_data\\filtered_func_data.nii.gz"
image_test_4D = sitk.ReadImage(image_path_4D)
print(image.GetOrigin())
print(image.GetSpacing())
print(image.GetDirection())


