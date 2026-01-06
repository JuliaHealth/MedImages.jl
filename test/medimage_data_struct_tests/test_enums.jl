# test/medimage_data_struct_tests/test_enums.jl
# Tests for enum definitions in MedImage_data_struct module

using Test
using MedImages

# Import test infrastructure (conditionally include if not already defined)
if !isdefined(@__MODULE__, :TestHelpers)
    include(joinpath(@__DIR__, "..", "test_helpers.jl"))
    include(joinpath(@__DIR__, "..", "test_config.jl"))
end
using .TestHelpers
using .TestConfig

@testset "Enum Definition Tests" begin
    with_temp_output_dir("medimage_data_struct", "enums") do output_dir

        @testset "Image_type enum" begin
            @test MedImages.MedImage_data_struct.MRI_type isa MedImages.MedImage_data_struct.Image_type
            @test MedImages.MedImage_data_struct.PET_type isa MedImages.MedImage_data_struct.Image_type
            @test MedImages.MedImage_data_struct.CT_type isa MedImages.MedImage_data_struct.Image_type
        end

        @testset "Interpolator_enum" begin
            @test MedImages.Nearest_neighbour_en isa MedImages.MedImage_data_struct.Interpolator_enum
            @test MedImages.Linear_en isa MedImages.MedImage_data_struct.Interpolator_enum
            @test MedImages.B_spline_en isa MedImages.MedImage_data_struct.Interpolator_enum
        end

        @testset "Orientation_code enum" begin
            orientations = [
                MedImages.ORIENTATION_RPI,
                MedImages.ORIENTATION_LPI,
                MedImages.ORIENTATION_RAI,
                MedImages.ORIENTATION_LAI,
                MedImages.ORIENTATION_RPS,
                MedImages.ORIENTATION_LPS,
                MedImages.ORIENTATION_RAS,
                MedImages.ORIENTATION_LAS
            ]

            for orient in orientations
                @test orient isa MedImages.MedImage_data_struct.Orientation_code
            end

            # All should be unique
            @test length(unique(orientations)) == 8
        end

        @testset "Mode_mi enum" begin
            @test MedImages.MedImage_data_struct.pixel_array_mode isa MedImages.MedImage_data_struct.Mode_mi
            @test MedImages.MedImage_data_struct.spat_metadata_mode isa MedImages.MedImage_data_struct.Mode_mi
            @test MedImages.MedImage_data_struct.all_mode isa MedImages.MedImage_data_struct.Mode_mi
        end

        @testset "current_device_enum" begin
            @test MedImages.MedImage_data_struct.CPU_current_device isa MedImages.MedImage_data_struct.current_device_enum
            @test MedImages.MedImage_data_struct.CUDA_current_device isa MedImages.MedImage_data_struct.current_device_enum
            @test MedImages.MedImage_data_struct.AMD_current_device isa MedImages.MedImage_data_struct.current_device_enum
            @test MedImages.MedImage_data_struct.ONEAPI_current_device isa MedImages.MedImage_data_struct.current_device_enum
        end

        @testset "Image_subtype enum" begin
            subtypes = [
                MedImages.MedImage_data_struct.CT_subtype,
                MedImages.MedImage_data_struct.ADC_subtype,
                MedImages.MedImage_data_struct.DWI_subtype,
                MedImages.MedImage_data_struct.T1_subtype,
                MedImages.MedImage_data_struct.T2_subtype,
                MedImages.MedImage_data_struct.FLAIR_subtype,
                MedImages.MedImage_data_struct.FDG_subtype,
                MedImages.MedImage_data_struct.PSMA_subtype
            ]

            for subtype in subtypes
                @test subtype isa MedImages.MedImage_data_struct.Image_subtype
            end
        end
    end
end

@testset "Enum Comparison Tests" begin
    @testset "Interpolator enums are distinct" begin
        @test MedImages.Nearest_neighbour_en != MedImages.Linear_en
        @test MedImages.Linear_en != MedImages.B_spline_en
        @test MedImages.Nearest_neighbour_en != MedImages.B_spline_en
    end

    @testset "Image type enums are distinct" begin
        @test MedImages.MedImage_data_struct.CT_type != MedImages.MedImage_data_struct.PET_type
        @test MedImages.MedImage_data_struct.PET_type != MedImages.MedImage_data_struct.MRI_type
        @test MedImages.MedImage_data_struct.CT_type != MedImages.MedImage_data_struct.MRI_type
    end

    @testset "Orientation codes are distinct" begin
        @test MedImages.ORIENTATION_RAS != MedImages.ORIENTATION_LAS
        @test MedImages.ORIENTATION_RPI != MedImages.ORIENTATION_LPI
    end
end
