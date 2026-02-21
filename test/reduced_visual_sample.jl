using MedImages
using MedImages.MedImage_data_struct
using MedImages.Utils
using MedImages.Basic_transformations
using PyCall

println("Loading SimpleITK...")
flush(stdout)
const sitk = pyimport("SimpleITK")
println("SimpleITK loaded!")
flush(stdout)

function create_synthetic_medimage_local(size_t::Tuple, type::Symbol)
    data = zeros(Float32, size_t)
    cx, cy, cz = size_t .÷ 2
    data[cx-5:cx+5, cy-5:cy+5, cz-5:cz+5] .= 1.0
    return MedImage(
        voxel_data=data,
        origin=(0.0,0.0,0.0),
        spacing=(1.0,1.0,1.0),
        direction=(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0),
        image_type=MedImages.MedImage_data_struct.MRI_type,
        image_subtype=MedImages.MedImage_data_struct.T1_subtype,
        patient_id="test"
    )
end

function main()
    img = create_synthetic_medimage_local((32, 32, 32), :block)
    batch = create_batched_medimage([img])
    
    println("Rotating...")
    flush(stdout)
    batch_rot = rotate_mi(batch, 3, 45.0, Linear_en)
    println("Rotated!")
    flush(stdout)
    
    println("Done!")
    flush(stdout)
end

main()
