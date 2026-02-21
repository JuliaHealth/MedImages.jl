using PyCall
const sitk = pyimport("SimpleITK")

println("SimpleITK version: ", sitk.Version_Version())

writer = sitk.ImageFileWriter()
println("\nRegistered ImageIOs:")
# Note: In SimpleITK, ImageFileWriter doesn't have a direct list of registered IOs exposed in the same way as ITK, 
# but we can try to see what it supports by trial and error or checking the version/features.
# However, we can check if it can *find* a writer for NIfTI.

test_filename = "test.nii.gz"
try
    io_name = writer.GetImageIO()
    println("Default ImageIO: ", io_name)
catch
    println("Could not get default ImageIO")
end

println("\nAttempting to find writer for .nii.gz...")
try
    # We can't easily query the writer list, so let's just create a dummy image and try to WriteImage
    # (but without actually writing if possible, or just write to temp)
    img = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    # sitk.WriteImage(img, "temp.nii.gz") # This would throw if no writer
    println(".nii.gz writer exists (if this script doesn't crash)")
catch e
    println("Error for .nii.gz: ", e)
end

# Let's try .mha
println("\nAttempting to find writer for .mha...")
try
    img = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    # sitk.WriteImage(img, "temp.mha")
    println(".mha writer exists (if this script doesn't crash)")
catch e
    println("Error for .mha: ", e)
end
