using LinearAlgebra, Test
using Dictionaries, Dates, PyCall


# Conda.add("SimpleITK")
include("../src/MedImage_data_struct.jl")
include("../src/Orientation_dicts.jl")
include("../src/Brute_force_orientation.jl")
include("../src/Utils.jl")

include("../src/Load_and_save.jl")

import ..MedImage_data_struct: MedImage
# include("./test_visualize.jl")

# import ..Load_and_save


"""
given a path to a dicom file and a path to a nifti file,
will load them and check weather :
1)the pixel arrays are the same pixel arrays constitute the main part of the image data that hold a number ussually a floating point number 
describing image properties such as intensity or density for each point

2) check is origin data is preserved    

3) check is direction data is preserved    

4) check is spacing data is preserved
"""

"""
compare two nifti files
"""
# function test_image_equality(path_nifti_a,path_nifti_b)
#     nifti_img = niread(path_nifti_a)
#     nifti_dicom_img = niread(path_nifti_b)

#     #check if the pixel arrays are the same
#     @test nifti_img.raw ≈ nifti_dicom_img.raw atol=0.1

#     #check if the origin data is preserved
#     @test nifti_img.header.qoffset_x ≈ nifti_dicom_img.header.qoffset_x atol=0.001
#     @test nifti_img.header.qoffset_y ≈ nifti_dicom_img.header.qoffset_y atol=0.001
#     @test nifti_img.header.qoffset_z ≈ nifti_dicom_img.header.qoffset_z atol=0.001

#     #check if the direction data is preserved
#     @test collect(nifti_img.header.srow_x) ≈ collect(nifti_dicom_img.header.srow_x) atol=0.001
#     @test collect(nifti_img.header.srow_y) ≈ collect(nifti_dicom_img.header.srow_y) atol=0.001
#     @test collect(nifti_img.header.srow_z) ≈ collect(nifti_dicom_img.header.srow_z) atol=0.001

#     #check if the spacing data is preserved
#     @test collect(nifti_img.header.pixdim) ≈ collect(nifti_dicom_img.header.pixdim) atol=0.001

# end#test_image_equality


"""
compare two object MedImage and simpleITK image in some cases like rotations
we can expect that there will be some diffrences on the edges of pixel
arrays still the center should be the same
"""
function test_object_equality(medIm, sitk_image)
    sitk = pyimport("SimpleITK")
    #e want just the center of the image as we have artifacts on edges

    # arr = pyconvert(Array,sitk.GetArrayFromImage(sitk_image))# we need numpy in order for pycall to automatically change it into julia array
    arr = sitk.GetArrayFromImage(sitk_image)# we need numpy in order for pycall to automatically change it into julia array
    spacing = sitk_image.GetSpacing()

    # sx,sy,sz=size(arr)
    # sx=Int(round(sx/4))
    # sy=Int(round(sy/4))
    # sz=Int(round(sz/4))

    # print("ssss shape  $(size(arr)) $([sx,sy,sz]) medim $(size(medIm.voxel_data[sx:end-sx,sy:end-sy,sz:end-sz]))  sitk $(size(medIm.voxel_data[sx:end-sx,sy:end-sy,sz:end-sz] )) \n")
    vox = medIm.voxel_data
    # vox = permutedims(vox, (3, 2, 1))
    vv = vox - arr
    # print("\n mmmmmmmmm $(maximum(vv)) $(maximum(arr))  \n")
    # print("vvvox arrr $(isapprox(arr ,vox; rtol =0.4)) \n ")

    # @test isapprox(arr[sx:end-sx,sy:end-sy,sz:end-sz]
    # ,medIm.voxel_data[sx:end-sx,sy:end-sy,sz:end-sz]; atol =0.1)
    # print("\n spacc sitk $(collect(sitk_image.GetSpacing())) medim $(collect(medIm.spacing)) \n")
    print("\n origin sitk $(collect(sitk_image.GetOrigin())) medim $(collect(medIm.origin)) \n")


    @test isapprox(collect(spacing), collect(Tuple{Float64,Float64,Float64}(medIm.spacing)); atol=0.1)
    # @test isapprox(spacing
    # ,medIm.spacing; atol =0.1)
    # print("\n dirrr sitk $(collect(sitk_image.GetDirection())) medim $(collect(medIm.direction)) \n")



    @test isapprox(collect(sitk_image.GetDirection()), collect(medIm.direction); atol=0.2)

    @test isapprox(collect(sitk_image.GetSpacing()), collect(medIm.spacing); atol=0.1)

    @test isapprox(collect(sitk_image.GetOrigin()), collect(medIm.origin); atol=0.1)

    @test isapprox(arr, vox; rtol=0.3)


end#test_image_equality



function test_image_equality_full(path_dicom, path_nifti)
    sitk = pyimport("SimpleITK")
    path_nifti_from_dicom = "./test_data/test_nifti.nii.gz"
    dicom_im = load_image(path_dicom)
    save_image(dicom_im, path_nifti_from_dicom)

    test_image_equality(path_nifti, path_nifti_from_dicom)


end

"""
create_nii_from_medimage(med_image::MedImage, file_path::String)

Create a .nii.gz file from a MedImage object and save it to the given file path.
"""
function create_nii_from_medimage(med_image, file_path::String)
    sitk = pyimport("SimpleITK")
    np = pyimport("numpy")
    # Convert voxel_data to a numpy array (Assuming voxel_data is stored in Julia array format)
    voxel_data_np = np.array(med_image.voxel_data)
    # Create a SimpleITK image from numpy array
    image_sitk = sitk.GetImageFromArray(voxel_data_np)

    # Set spatial metadata
    image_sitk.SetOrigin(med_image.origin)
    image_sitk.SetSpacing(med_image.spacing)
    image_sitk.SetDirection(med_image.direction)

    # Save the image as .nii.gz
    sitk.WriteImage(image_sitk, file_path * ".nii.gz")
end



# p="/home/jakubmitura/projects/MedImage.jl/test_data/volume-0.nii.gz"
# # test_image_equality(p,p)

# medimage_instance_array = load_images("/home/jakubmitura/projects/MedImage.jl/test_data/volume-0.nii.gz")
# medimage_instance = medimage_instance_array[1]

# test_object_equality(medimage_instance,sitk.ReadImage(p))

# dcm_data_array = dcmdir_parse("/workspaces/MedImage.jl/MedImage/test_data/ScalarVolume_0")