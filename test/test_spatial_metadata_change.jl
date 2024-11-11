using LinearAlgebra
using PyCall
using BenchmarkTools,CUDA

include("../src/MedImage_data_struct.jl")
include("../src/Orientation_dicts.jl")
include("../src/Brute_force_orientation.jl")
include("../src/Utils.jl")
include("../src/Load_and_save.jl")
include("../src/Spatial_metadata_change.jl")
include("./dicom_nifti.jl")

import ..MedImage_data_struct: MedImage, Interpolator_enum, Mode_mi, Orientation_code, Nearest_neighbour_en, Linear_en, B_spline_en
import ..Load_and_save: load_image,create_nii_from_medimage,update_voxel_data
import ..Spatial_metadata_change: change_orientation, resample_to_spacing
# sitk = pyimport_conda("SimpleITK", "simpleitk")

function sitk_resample(path_nifti, targetSpac)
    sitk = pyimport_conda("SimpleITK", "simpleitk")
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
    sitk = pyimport_conda("SimpleITK", "simpleitk")
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

    sitk = pyimport_conda("SimpleITK", "simpleitk")
    index=0
    for spac in [ (5.0, 0.9, 0.7),(1.0, 2.0, 1.1), (2.0, 3.0, 4.0)]        # Load SimpleITK
        index+=1
        # Load the image from path
        med_im = load_image(path_nifti)
        # sitk.ReadImage(path_nifti)

        #precompile
        sitk_resample(path_nifti, spac)     
        resample_to_spacing(med_im, spac, B_spline_en)
        resample_to_spacing(update_voxel_data(med_im,CuArray(med_im.voxel_data)), spac, B_spline_en,true)


        current_time = Dates.now()
        sitk_image = sitk_resample(path_nifti, spac)
        println("Time for SimpleITK resample: ", Dates.now() - current_time)
        current_time = Dates.now()
        med_im = resample_to_spacing(med_im, spac, B_spline_en)
        println("Time for MedImage resample: ", Dates.now() - current_time)
        current_time = Dates.now()
        resample_to_spacing(update_voxel_data(med_im,CuArray(med_im.voxel_data)), spac, B_spline_en,true)
        println("Time for MedImage GPU resample: ", Dates.now() - current_time)


        sitk.WriteImage(sitk_image, "/workspaces/MedImage.jl/test_data/debug/sitk_$(index).nii.gz")
        create_nii_from_medimage(med_im,"/workspaces/MedImage.jl/test_data/debug/medim_$(index)")

        test_object_equality(med_im, sitk_image)

    end

end


"""
test if the resample_to_spacing of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function
We need to check can it change between RAS and LPS orientationas those are most common
"""
function test_change_orientation(path_nifti)

    # for orientation in ["LIP","LSP","RIA","LIA","RSA","LSA","IRP","ILP","SRP","SLP","IRA","ILA","SRA","SLA","RPI","LPI","RAI","LAI","RPS","LPS"]#,"RSP","RIP"
    for orientation in ["RAS","LAS" ,"RPI","LPI","RAI","LAI","RPS","LPS"]#,"RSP","RIP"
        med_im = load_image(path_nifti)
        sitk_image = change_image_orientation(path_nifti, orientation)
        med_im = change_orientation(med_im, string_to_orientation_enum[orientation])
        test_object_equality(med_im, sitk_image)
    end
end


path_nifti = "/workspaces/MedImage.jl/test_data/for_resample_target/ct_soft_pat_3_sudy_0.nii.gz"
test_resample_to_spacing(path_nifti)
# test_change_orientation(path_nifti)

