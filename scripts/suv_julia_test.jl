using MedImages
using Dates

if length(ARGS) < 1
    println("Usage: julia suv_julia_test.jl <dicom_folder>")
    exit(1)
end

folder = ARGS[1]

println("Loading image from: ", folder)
try
    # Load image (fetching metadata via pydicom internally)
    # We pass "PET" as type to ensure correct subtypes are set, though SUV calc is generic on metadata
    mi = MedImages.load_image(folder, "PET")

    println("Calculating SUV factor...")
    factor = MedImages.calculate_suv_factor(mi)

    if factor !== nothing
        println("SUV Factor: ", factor)
    else
        println("SUV Factor could not be calculated (missing metadata).")
        exit(1)
    end
catch e
    println("Error: ", e)
    exit(1)
end
