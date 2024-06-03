using  LinearAlgebra,
include("../src/Load_and_save.jl")
# include("../src/Basic_transformations.jl")
# include("./test_visualize.jl")
include("./dicom_nifti.jl")
include("../src/Spatial_metadata_change.jl")

# sitk = pyimport_conda("SimpleITK", "simpleitk")


function sitk_resample(path_nifti, targetSpac)

    image = sitk.ReadImage(path_nifti)
    origSize = image.GetSize()
    orig_spacing = image.GetSpacing()
    new_size = Tuple{Int64,Int64,Int64}([Int64(ceil(origSize[1] * (orig_spacing[1] / targetSpac[1]))),
        Int64(ceil(origSize[2] * (orig_spacing[2] / targetSpac[2]))),
        Int64(ceil(origSize[3] * (orig_spacing[3] / targetSpac[3])))])

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(targetSpac)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)
    resample.SetSize(new_size)
    res = resample.Execute(image)
    return res

end


function change_image_orientation(path_nifti, orientation)
    # Read the image
    image = sitk.ReadImage(path_nifti)

    # Create a DICOMOrientImageFilter
    orient_filter = sitk.DICOMOrientImageFilter()

    # Set the desired orientation
    orient_filter.SetDesiredCoordinateOrientation(orientation)

    # Apply the filter to the image
    oriented_image = orient_filter.Execute(image)
    return oriented_image
    # Write the oriented image back to the file
    # sitk.WriteImage(oriented_image, path_nifti)

end
"""
test if the resample_to_spacing of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function
we nned to check the nearest neuhnbor interpolation and b spline interpolation
OnCell() - give interpolation in the center of the voxel
"""
function test_resample_to_spacing(path_nifti)


    for spac in [(1.0, 2.0, 1.1), (2.0, 3.0, 4.0), (5.0, 0.9, 0.7)]        # Load SimpleITK
        # Load the image from path
        med_im = load_image(path_nifti)
        # sitk.ReadImage(path_nifti)
        sitk_image = sitk_resample(path_nifti, spac)
        med_im = resample_to_spacing(med_im, spac, B_spline_en)
        test_object_equality(med_im, sitk_image)

    end

end


"""
test if the resample_to_spacing of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function
We need to check can it change between RAS and LPS orientationas those are most common
"""
function test_change_orientation(path_nifti)

    # for orientation in ["LIP","LSP","RIA","LIA","RSA","LSA","IRP","ILP","SRP","SLP","IRA","ILA","SRA","SLA","RPI","LPI","RAI","LAI","RPS","LPS","RAS","LAS","PRI","PLI","ARI","ALI","PRS","PLS","ARS","ALS","IPR","SPR","IAR","SAR","IPL","SPL","IAL","SAL","PIR","PSR","AIR","ASR","PIL","PSL","AIL","ASL"]#,"RSP","RIP"
    for orientation in ["RAS","LAS" ,"RPI","LPI","RAI","LAI","RPS","LPS"]#,"RSP","RIP"
        med_im = load_image(path_nifti)
        sitk_image = change_image_orientation(path_nifti, orientation)
        med_im = change_orientation(med_im, string_to_orientation_enum[orientation])
        test_object_equality(med_im, sitk_image)
    end
end


path_nifti = "/home/jm/projects_new/MedImage.jl/test_data/volume-0.nii.gz"
# path_nifti = "/home/jm/projects_new/MedImage.jl/test_data/for_resample_target/pat_2_sudy_0_2022-09-16_Standardized_Uptake_Value_body_weight.nii.gz"
# path_nifti = "/home/jm/projects_new/MedImage.jl/test_data/synthethic_small.nii.gz"

# test_resample_to_spacing(path_nifti)
test_change_orientation(path_nifti)

# sitk = pyimport_conda("SimpleITK", "simpleitk")


# spac=(6.3,4.1,0.5)
# # Load the image from path
# med_im=load_image(path_nifti)
# # sitk.ReadImage(path_nifti)
# sitk_image=sitk_resample(path_nifti,spac)
# med_im=resample_to_spacing(med_im,spac,B_spline_en)
# test_object_equality(med_im,sitk_image)

# sitk.WriteImage(sitk_image,"/home/jakubmitura/projects/MedImage.jl/test_data/volume-0_resampled_sitk.nii.gz")
# create_nii_from_medimage(med_im,"/home/jakubmitura/projects/MedImage.jl/test_data/volume-0_resampled_medimage.nii.gz")
