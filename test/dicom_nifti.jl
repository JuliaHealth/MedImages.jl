
using NIfTI,LinearAlgebra,DICOM,Test
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
function test_image_equality(path_dicom,path_nifti)

    dicom_im=load_image(path_dicom)
    save_image(dicom_im,path_nifti_from_dicom)

    # Load the image from path
    nifti_img = niread(path_nifti)
    nifti_dicom_img = niread(path_nifti_from_dicom)

    #check if the pixel arrays are the same
    @test nifti_img.raw ≈ nifti_dicom_img.raw atol=0.01

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


    ######## check additionally weather nifti loading is working also

    orig_nifti=load_image(path_nifti)
    save_image(dicom_im,path_nifti_from_dicom)


    #check if the pixel arrays are the same
    @test nifti_img.raw ≈ nifti_dicom_img.raw atol=0.01

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


end    

# p="/workspaces/MedImage.jl/MedImage3D/test_data/volume-0.nii.gz"
# test_image_equality(p,p)

# dcm_data_array = dcmdir_parse("/workspaces/MedImage.jl/MedImage3D/test_data/ScalarVolume_0")