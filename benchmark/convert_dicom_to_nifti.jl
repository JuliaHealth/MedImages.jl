"""
DICOM to NIFTI Converter
Converts downloaded DICOM series to NIFTI format using MedImages.jl.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))  # Activate MedImages.jl environment

using MedImages
using Printf
using JSON3
using Statistics
using Dates
using DataFrames
using CSV

"""
    convert_dicom_to_nifti(dicom_dir::String, output_dir::String; modality::String="CT") -> Union{String, Nothing}

Convert a DICOM directory to NIFTI format.

# Arguments
- `dicom_dir`: Path to directory containing DICOM files
- `output_dir`: Directory to save NIFTI file
- `modality`: Modality type ("CT", "MR", or "PET")

# Returns
Path to generated NIFTI file, or nothing if conversion failed
"""
function convert_dicom_to_nifti(dicom_dir::String, output_dir::String; modality::String="CT")
    println("Converting: $dicom_dir")
    println("  Modality: $modality")

    try
        # Load DICOM using MedImages
        med_image = MedImages.load_image(dicom_dir, modality)

        # Print image info
        dims = size(med_image.voxel_data)
        spacing = med_image.spacing
        data_type = eltype(med_image.voxel_data)
        size_mb = prod(dims) * sizeof(data_type) / 1024^2

        @printf("  Dimensions: %d x %d x %d\n", dims...)
        @printf("  Spacing: %.3f x %.3f x %.3f mm\n", spacing...)
        @printf("  Data type: %s\n", data_type)
        @printf("  Size: %.1f MB\n", size_mb)
        @printf("  Value range: [%.1f, %.1f]\n", minimum(med_image.voxel_data), maximum(med_image.voxel_data))

        # Categorize by size
        max_dim = maximum(dims)
        if max_dim >= 1024
            size_category = "xlarge"
        elseif max_dim >= 512
            size_category = "large"
        else
            size_category = "medium"
        end

        println("  Size category: $size_category")

        # Create output directory
        output_subdir = joinpath(output_dir, size_category)
        mkpath(output_subdir)

        # Generate output filename
        base_name = basename(dicom_dir)
        nifti_path = joinpath(output_subdir, base_name)

        # Save as NIFTI
        println("  Saving to: $nifti_path.nii.gz")
        MedImages.create_nii_from_medimage(med_image, nifti_path)

        # Verify file was created
        nifti_file = nifti_path * ".nii.gz"
        if isfile(nifti_file)
            file_size = filesize(nifti_file) / 1024^2
            @printf("  NIFTI file size: %.1f MB\n", file_size)

            # Save metadata
            metadata = Dict(
                "dicom_dir" => dicom_dir,
                "nifti_file" => nifti_file,
                "dimensions" => dims,
                "spacing" => spacing,
                "origin" => med_image.origin,
                "modality" => modality,
                "data_type" => string(data_type),
                "size_mb" => size_mb,
                "value_range" => [minimum(med_image.voxel_data), maximum(med_image.voxel_data)],
                "size_category" => size_category,
                "conversion_date" => string(now())
            )

            metadata_file = nifti_path * "_metadata.json"
            open(metadata_file, "w") do f
                JSON3.write(f, metadata, allow_inf=true)
            end

            println("  Conversion successful\n")
            return nifti_file
        else
            @error "NIFTI file was not created" nifti_path=nifti_file
            return nothing
        end

    catch e
        @error "Conversion failed" dicom_dir=dicom_dir exception=e
        return nothing
    end
end

"""
    convert_all_dicom(dicom_root::String, nifti_root::String; modality::String="CT") -> Vector{String}

Convert all DICOM directories in root directory to NIFTI.

# Arguments
- `dicom_root`: Root directory containing DICOM directories
- `nifti_root`: Root directory for NIFTI output
- `modality`: Default modality (will try to infer from metadata if available)

# Returns
Vector of successfully converted NIFTI file paths
"""
function convert_all_dicom(dicom_root::String, nifti_root::String; modality::String="CT")
    println("="^80)
    println("DICOM to NIFTI Batch Converter")
    println("="^80)
    println("DICOM root: $dicom_root")
    println("NIFTI root: $nifti_root")
    println("Default modality: $modality")
    println()

    # Find all DICOM directories
    dicom_dirs = String[]
    for entry in readdir(dicom_root, join=true)
        if isdir(entry)
            # Check if contains DICOM files
            files = readdir(entry)
            if any(f -> endswith(f, ".dcm"), files)
                push!(dicom_dirs, entry)
            end
        end
    end

    println("Found $(length(dicom_dirs)) DICOM directories\n")

    if isempty(dicom_dirs)
        @warn "No DICOM directories found in $dicom_root"
        return String[]
    end

    # Try to load metadata to get actual modalities
    metadata_file = joinpath(dirname(dicom_root), "metadata.json")
    series_metadata = Dict{String, Any}()

    if isfile(metadata_file)
        println("Loading series metadata from $metadata_file")
        try
            metadata = JSON3.read(read(metadata_file, String))
            if haskey(metadata, :series)
                for s in metadata[:series]
                    if haskey(s, :SeriesInstanceUID)
                        uid_key = replace(string(s[:SeriesInstanceUID]), "." => "_")
                        series_metadata[uid_key] = s
                    end
                end
            end
            println("Loaded metadata for $(length(series_metadata)) series\n")
        catch e
            @warn "Failed to load metadata" exception=e
        end
    end

    # Convert each directory
    nifti_files = String[]

    for (i, dicom_dir) in enumerate(dicom_dirs)
        println("[$i/$(length(dicom_dirs))]")

        # Try to infer modality from metadata
        dir_name = basename(dicom_dir)
        series_mod = modality

        if haskey(series_metadata, dir_name)
            series_info = series_metadata[dir_name]
            if haskey(series_info, :Modality)
                series_mod = string(series_info[:Modality])
                println("  Inferred modality from metadata: $series_mod")
            end
        end

        # Convert
        nifti_file = convert_dicom_to_nifti(dicom_dir, nifti_root, modality=series_mod)

        if !isnothing(nifti_file)
            push!(nifti_files, nifti_file)
        end
    end

    # Summary
    println("="^80)
    println("Conversion Summary")
    println("="^80)
    println("Total DICOM directories: $(length(dicom_dirs))")
    println("Successful conversions: $(length(nifti_files))")
    println("Failed conversions: $(length(dicom_dirs) - length(nifti_files))")

    if !isempty(nifti_files)
        # Count by size category
        large_count = count(f -> contains(f, "/large/"), nifti_files)
        xlarge_count = count(f -> contains(f, "/xlarge/"), nifti_files)

        println("\nBy size:")
        println("  Large (512+): $large_count")
        println("  XLarge (1024+): $xlarge_count")

        # Calculate total size
        total_size = sum(filesize(f) for f in nifti_files) / 1024^3
        @printf("  Total size: %.2f GB\n", total_size)
    end

    return nifti_files
end

"""
    create_benchmark_catalog(nifti_root::String) -> DataFrame

Create a catalog of all converted NIFTI files for benchmarking.
"""
function create_benchmark_catalog(nifti_root::String)
    println("\nCreating benchmark catalog...")

    catalog = DataFrame(
        file=String[],
        size_category=String[],
        dimensions=Vector{Int}[],
        spacing=Vector{Float64}[],
        modality=String[],
        size_mb=Float64[]
    )

    for size_cat in ["large", "xlarge"]
        size_dir = joinpath(nifti_root, size_cat)
        if !isdir(size_dir)
            continue
        end

        for entry in readdir(size_dir, join=true)
            if endswith(entry, "_metadata.json")
                try
                    metadata = JSON3.read(read(entry, String))
                    nifti_file = metadata[:nifti_file]

                    if isfile(nifti_file)
                        push!(catalog, (
                            file=nifti_file,
                            size_category=size_cat,
                            dimensions=[metadata[:dimensions]...],
                            spacing=[metadata[:spacing]...],
                            modality=metadata[:modality],
                            size_mb=metadata[:size_mb]
                        ))
                    end
                catch e
                    @warn "Failed to read metadata" file=entry exception=e
                end
            end
        end
    end

    # Sort by size descending
    sort!(catalog, :size_mb, rev=true)

    catalog_file = joinpath(nifti_root, "benchmark_catalog.csv")
    CSV.write(catalog_file, catalog)

    println("Catalog saved to: $catalog_file")
    println("Total images: $(nrow(catalog))")

    return catalog
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    # Run as script
    dicom_root = "benchmark_data/dicom_raw"
    nifti_root = "benchmark_data/nifti"

    if !isdir(dicom_root)
        @error "DICOM directory not found: $dicom_root"
        @error "Please run download_tcia_data.jl first"
        exit(1)
    end

    # Convert all DICOM to NIFTI
    nifti_files = convert_all_dicom(dicom_root, nifti_root)

    # Create catalog
    if !isempty(nifti_files)
        catalog = create_benchmark_catalog(nifti_root)
        println("\nBenchmark Catalog:")
        println(catalog)
    end
end
