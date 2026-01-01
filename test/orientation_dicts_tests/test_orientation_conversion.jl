# test/orientation_dicts_tests/test_orientation_conversion.jl
# Tests for orientation dictionaries and conversion functions

using Test
using MedImages

# Import test infrastructure (conditionally include if not already defined)
if !isdefined(@__MODULE__, :TestHelpers)
    include(joinpath(@__DIR__, "..", "test_helpers.jl"))
    include(joinpath(@__DIR__, "..", "test_config.jl"))
end
using .TestHelpers
using .TestConfig

@testset "Orientation Dictionary Tests" begin
    with_temp_output_dir("orientation_dicts", "orientation_conversion") do output_dir

        @testset "string_to_orientation_enum dictionary" begin
            # All standard orientations should have mappings
            standard_orientations = ["RAS", "LAS", "RPI", "LPI", "RAI", "LAI", "RPS", "LPS"]

            for orient_str in standard_orientations
                @test haskey(MedImages.string_to_orientation_enum, orient_str)
                orient_enum = MedImages.string_to_orientation_enum[orient_str]
                @test orient_enum isa MedImages.MedImage_data_struct.Orientation_code
            end
        end

        @testset "orientation_enum_to_string dictionary" begin
            orientations = [
                MedImages.MedImage_data_struct.ORIENTATION_RAS,
                MedImages.MedImage_data_struct.ORIENTATION_LAS,
                MedImages.MedImage_data_struct.ORIENTATION_RPI,
                MedImages.MedImage_data_struct.ORIENTATION_LPI,
                MedImages.MedImage_data_struct.ORIENTATION_RAI,
                MedImages.MedImage_data_struct.ORIENTATION_LAI,
                MedImages.MedImage_data_struct.ORIENTATION_RPS,
                MedImages.MedImage_data_struct.ORIENTATION_LPS
            ]

            for orient_enum in orientations
                @test haskey(MedImages.orientation_enum_to_string, orient_enum)
                orient_str = MedImages.orientation_enum_to_string[orient_enum]
                @test orient_str isa String
                @test length(orient_str) == 3
            end
        end

        @testset "Bidirectional consistency" begin
            # Converting string -> enum -> string should give original string
            test_orientations = ["RAS", "LAS", "RPI", "LPI"]

            for orient_str in test_orientations
                orient_enum = MedImages.string_to_orientation_enum[orient_str]
                back_to_str = MedImages.orientation_enum_to_string[orient_enum]
                @test orient_str == back_to_str
            end
        end

        @testset "All enum values have string mappings" begin
            for orient in AVAILABLE_ORIENTATIONS
                @test haskey(MedImages.orientation_enum_to_string, orient)
            end
        end
    end
end

@testset "Orientation String Format Tests" begin
    @testset "Orientation strings are valid 3-letter codes" begin
        for (str, _) in MedImages.string_to_orientation_enum
            @test length(str) == 3
            @test all(c -> c in "RLAPIS", str)
        end
    end

    @testset "First letter indicates Left-Right axis" begin
        for (str, _) in MedImages.string_to_orientation_enum
            @test str[1] in ['R', 'L']
        end
    end

    @testset "Second letter indicates Anterior-Posterior axis" begin
        for (str, _) in MedImages.string_to_orientation_enum
            @test str[2] in ['A', 'P']
        end
    end

    @testset "Third letter indicates Superior-Inferior axis" begin
        for (str, _) in MedImages.string_to_orientation_enum
            @test str[3] in ['S', 'I']
        end
    end
end
