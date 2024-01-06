
using NIfTI,LinearAlgebra,DICOM,Test
include("./test_visualize.jl")

# import ..Load_and_save


# TODO: continue with the rest of the code
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
function test_image_equality(path_nifti_a,path_nifti_b)
    nifti_img = niread(path_nifti_a)
    nifti_dicom_img = niread(path_nifti_b)

    #check if the pixel arrays are the same
    @test nifti_img.raw ≈ nifti_dicom_img.raw atol=0.1

    #check if the origin data is preserved
    @test nifti_img.header.qoffset_x ≈ nifti_dicom_img.header.qoffset_x atol=0.001
    @test nifti_img.header.qoffset_y ≈ nifti_dicom_img.header.qoffset_y atol=0.001
    @test nifti_img.header.qoffset_z ≈ nifti_dicom_img.header.qoffset_z atol=0.001

    #check if the direction data is preserved
    @test collect(nifti_img.header.srow_x) ≈ collect(nifti_dicom_img.header.srow_x) atol=0.001
    @test collect(nifti_img.header.srow_y) ≈ collect(nifti_dicom_img.header.srow_y) atol=0.001
    @test collect(nifti_img.header.srow_z) ≈ collect(nifti_dicom_img.header.srow_z) atol=0.001

    #check if the spacing data is preserved
    @test collect(nifti_img.header.pixdim) ≈ collect(nifti_dicom_img.header.pixdim) atol=0.001

end#test_image_equality


"""
compare two object MedImage and simpleITK image in some cases like rotations
we can expect that there will be some diffrences on the edges of pixel
arrays still the center should be the same
"""
function test_object_equality(medIm::MedImage,sitk_image)
    
    #e want just the center of the image as we have artifacts on edges

    arr = pyconvert(Array,sitk.GetArrayFromImage(sitk_image))# we need numpy in order for pycall to automatically change it into julia array
    spacing = sitk_image.GetSpacing()
    
    sx,sy,sz=size(arr)
    sx=Int(round(sx/4))
    sy=Int(round(sy/4))
    sz=Int(round(sz/4))

    @test isapprox(arr[sx:-sx,sy:-sy,sz:-sz]
    ,medIm.pixel_array[sx:-sx,sy:-sy,sz:-sz]; atol =0.1)

    @test isapprox(spacing
    ,medIm.spacing; atol =0.1)

    
    @test isapprox(rotated.GetOrigin()
    ,medIm.origin; atol =0.1)
        
    @test isapprox(rotated.GetDirection()
    ,medIm.direction; atol =0.1)
end#test_image_equality



function test_image_equality_full(path_dicom,path_nifti)

    path_nifti_from_dicom="./test_data/test_nifti.nii.gz"
    dicom_im=load_image(path_dicom)
    save_image(dicom_im,path_nifti_from_dicom)

    test_image_equality(path_nifti,path_nifti_from_dicom)


end    

# p="/workspaces/MedImage.jl/MedImage3D/test_data/volume-0.nii.gz"
# test_image_equality(p,p)

# dcm_data_array = dcmdir_parse("/workspaces/MedImage.jl/MedImage3D/test_data/ScalarVolume_0")