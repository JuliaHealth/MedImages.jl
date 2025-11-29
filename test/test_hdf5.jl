using Test
using LinearAlgebra
using HDF5
using MedImages

function test_hdf5_suite()
    @testset "HDF5 Test Suite" begin
        # Test data path
        test_data_dir = joinpath(dirname(@__FILE__), "..", "test_data")
        path_nifti = joinpath(test_data_dir, "volume-0.nii.gz")
        
        @testset "HDF5 Save and Load" begin
            @test begin
                h5_path = tempname() * ".h5"
                # Load image using MedImages
                med_im = MedImages.load_image(path_nifti, "CT")
                
                # Save to HDF5
                f = h5open(h5_path, "w")
                uid = MedImages.save_med_image(f, "test_image", med_im)
                close(f)
                
                f = h5open(h5_path, "r")
                # Load from HDF5
                med_im_2 = MedImages.load_med_image(f, "test_image", uid)
                close(f)
                rm(h5_path)
                
                # Test that voxel data is preserved
                med_im.voxel_data == med_im_2.voxel_data
            end
        end
        
        @testset "HDF5 Metadata Preservation" begin
            @test begin
                h5_path = tempname() * ".h5"
                # Load image using MedImages
                med_im = MedImages.load_image(path_nifti, "CT")
                
                # Save and load via HDF5
                f = h5open(h5_path, "w")
                uid = MedImages.save_med_image(f, "test_image", med_im)
                close(f)

                f = h5open(h5_path, "r")
                med_im_2 = MedImages.load_med_image(f, "test_image", uid)
                close(f)
                rm(h5_path)
                
                # Test metadata preservation
                (med_im.spacing == med_im_2.spacing) &&
                (med_im.origin == med_im_2.origin) &&
                (med_im.direction == med_im_2.direction)
            end
        end
    end
end