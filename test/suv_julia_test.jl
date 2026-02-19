using MedImages
using MedImages.SUV_calc
using MedImages.Load_and_save

function verify_suv_julia(dicom_dir)
    println("Loading PET image from: $dicom_dir")
    # PET type is explicitly passed
    mi = load_image(dicom_dir, "PET")
    
    println("Metadata keys: ", keys(mi.metadata))
    if haskey(mi.metadata, "RadiopharmaceuticalInformationSequence")
        println("Radio seq found. length: ", length(mi.metadata["RadiopharmaceuticalInformationSequence"]))
        println("First item keys: ", keys(mi.metadata["RadiopharmaceuticalInformationSequence"][1]))
    else
        println("Radio seq MISSING from mi.metadata")
    end

    println("Calculating SUV factor...")
    factor = calculate_suv_factor(mi)
    
    if factor === nothing
        println("Error: Could not calculate SUV factor. Check metadata.")
        return
    end
    
    println("Julia SUV Factor: $factor")
    
    # Apply factor to voxel data
    suv_voxel_data = mi.voxel_data .* Float32(factor)
    
    # Create a new MedImage with SUV data
    mi_suv = update_voxel_data(mi, suv_voxel_data)
    
    # Save as NIfTI
    save_path = "juliasuv.nii.gz"
    println("Saving Julia SUV image to: $save_path")
    create_nii_from_medimage(mi_suv, save_path)
end

dicom_dir = "/home/jm/project_ssd/MedImages.jl/test/visual_output/local/dicoms/5-PET_WB_120min_Uncorrected/resources/DICOM/files"
verify_suv_julia(dicom_dir)
