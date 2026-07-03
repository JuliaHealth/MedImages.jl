using MedImages
using MedImages.MedImage_data_struct
using MedImages.Load_and_save
using MedImages.SUV_calc
using MedImages.Resample_to_target

const PET_DIR = joinpath(@__DIR__, "..", "..", "test_data", "tcia_pet", "PT_dcm")
const CT_DIR  = joinpath(@__DIR__, "..", "..", "test_data", "tcia_pet", "CT_dcm")
const OUT     = joinpath(@__DIR__, "..", "..", "test_data", "tcia_pet")

# F-18 mean energy emitted per decay (keV): 0.969*beta_mean + 2*511*branch
const E_TOTAL_keV = 0.969 * 249.8 + 2.0 * 511.0 * 0.969   # ~1232.4

# CT HU -> mass density (g/mL), piecewise (same mapping as analytical baseline)
hu_to_den(hu) = hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) :
                          1.0f0 + 0.0007f0 * Float32(hu)

println("Loading PET series: $PET_DIR")
pet = Load_and_save.load_image(PET_DIR, "PET")
println("  PET size ", size(pet.voxel_data), " spacing ", pet.spacing)

println("Loading CT series: $CT_DIR")
ct = Load_and_save.load_image(CT_DIR, "CT")
println("  CT size ", size(ct.voxel_data), " spacing ", ct.spacing)

# Resample CT onto the PET grid using MedImages' own resample_to_image
println("Resampling CT -> PET grid via MedImages.resample_to_image ...")
ct_on_pet = Resample_to_target.resample_to_image(pet, ct, MedImage_data_struct.Linear_en)
println("  CT-on-PET size ", size(ct_on_pet.voxel_data))

# SUV factor from MedImages (uses DICOM radiopharmaceutical metadata)
suv = calculate_suv_factor(pet)
println("SUV factor (MedImages): ", suv)
suv_factor = suv === nothing ? 1.0 : suv

# activity / SUV map (spatial distribution of tracer uptake)
activity = Float32.(pet.voxel_data) .* Float32(suv_factor)
activity[activity .< 0] .= 0.0f0

# density from CT (clamped to plausible range)
density = clamp.(hu_to_den.(ct_on_pet.voxel_data), 0.01f0, 3.0f0)

# local-energy-deposition absorbed dose (relative): all decay energy deposited in-voxel
dose_local = activity .* Float32(E_TOTAL_keV) ./ density

# save activity, density, and local dose on the PET grid
act_mi  = update_voxel_data(pet, activity)
den_mi  = update_voxel_data(pet, density)
dose_mi = update_voxel_data(pet, dose_local)
Load_and_save.create_nii_from_medimage(act_mi,  joinpath(OUT, "activity"))
Load_and_save.create_nii_from_medimage(den_mi,  joinpath(OUT, "density"))
Load_and_save.create_nii_from_medimage(dose_mi, joinpath(OUT, "dose_local"))

println("activity max ", maximum(activity), " | density range ",
        minimum(density), "-", maximum(density),
        " | dose_local max ", maximum(dose_local))
println("MEDIMAGES DOSE DONE")
