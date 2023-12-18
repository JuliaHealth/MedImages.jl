using NIfTI,LinearAlgebra,DICOM


"""
given a path to a nifti file that we got from dicom file and a path to a nifti file,
will load them and check weather :
1)the pixel arrays are the same pixel arrays constitute the main part of the image data that hold a number ussually a floating point number 
describing image properties such as intensity or density for each point

2) check is origin data is preserved    

3) check is direction data is preserved    

4) check is spacing data is preserved
"""
function test_pixel_array(path_nifti_from_dicom,path_nifti)

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
    @test nifti_img.header.srow_x ≈ nifti_dicom_img.header.srow_x atol=0.001
    @test nifti_img.header.srow_y ≈ nifti_dicom_img.header.srow_y atol=0.001
    @test nifti_img.header.srow_z ≈ nifti_dicom_img.header.srow_z atol=0.001

    #check if the spacing data is preserved
    @test nifti_img.header.pixdim ≈ nifti_dicom_img.header.pixdim atol=0.001

end    