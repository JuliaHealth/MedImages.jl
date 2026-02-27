# test/basic_transformations_tests/test_affine_transform_mi.jl
# Tests for affine_transform_mi function for single MedImage

module TestAffineTransform
    using Test
    using MedImages
    using LinearAlgebra

    # Import test infrastructure
    # Since we are including this from runtests.jl, we can assume Main has the helpers
    # or we can include them locally if they aren't defined.
    # For independent runs, we include them.
    if !@isdefined(TestHelpers)
        include(joinpath(@__DIR__, "..", "test_helpers.jl"))
        include(joinpath(@__DIR__, "..", "test_config.jl"))
    end
    import .TestHelpers: load_test_image
    import .TestConfig: TEST_NIFTI_FILE

    @testset "affine_transform_mi Single MedImage Tests" begin
        
        # Check if test data exists
        if !isfile(TEST_NIFTI_FILE)
            @test_skip "Test file not found: $TEST_NIFTI_FILE"
        else
            # Load a single MedImage
            med_im = load_test_image()
            @test med_im isa MedImage
            
            # Define an affine transformation
            M = create_affine_matrix(rotation=(0, 0, 30), translation=(10, 5, 0))
            
            @testset "Basic Dispatch and Execution" begin
                # Test linear interpolation
                med_im_transformed = MedImages.affine_transform_mi(med_im, M, MedImages.Linear_en)
                
                @test med_im_transformed isa MedImage
                @test size(med_im_transformed.voxel_data) == size(med_im.voxel_data)
                @test med_im_transformed.origin == med_im.origin
                @test med_im_transformed.spacing == med_im.spacing
                
                # Verify that some voxels changed
                @test !all(med_im_transformed.voxel_data .== med_im.voxel_data)
            end

            @testset "Interpolator Consistency" begin
                for interp in [MedImages.Nearest_neighbour_en, MedImages.Linear_en]
                    med_im_transformed = MedImages.affine_transform_mi(med_im, M, interp)
                    @test med_im_transformed isa MedImage
                end
            end
        end
    end
end
using .TestAffineTransform
