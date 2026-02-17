using PyCall
const sitk = pyimport("SimpleITK")
const np = pyimport("numpy")
using MedImages

# Define path
out_dir = abspath("test/visual_output/batched/SimpleITK")
if !isdir(out_dir)
    println("Directory does not exist: ", out_dir)
    mkpath(out_dir)
end

out_file = joinpath(out_dir, "debug.nii.gz")

println("Writing to: ", out_file)
try
    img = sitk.GetImageFromArray(np.zeros((10,10,10)))
    sitk.WriteImage(img, out_file)
    println("Success!")
catch e
    println("Error writing file:")
    println(e)
end
