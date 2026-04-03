"""
TCIA Data Downloader
Downloads large medical imaging datasets from The Cancer Imaging Archive (TCIA) REST API v4.

API Documentation: https://wiki.cancerimagingarchive.net/x/NIIiAQ
"""

using HTTP
using JSON3
using ZipFile
using Printf
using Dates

# TCIA API v4 base URL
const TCIA_API_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v4"

"""
    json3_to_dict(obj) -> Dict

Convert JSON3 object to a regular Dict (JSON3 objects are immutable).
"""
function json3_to_dict(obj)
    result = Dict{Symbol, Any}()
    for key in keys(obj)
        result[key] = obj[key]
    end
    return result
end

"""
    get_collections() -> Vector{String}

Get list of all available collections from TCIA (no authentication required).
"""
function get_collections()
    url = "$TCIA_API_BASE/getCollectionValues"
    println("Querying TCIA collections...")

    try
        response = HTTP.get(url)
        collections = JSON3.read(String(response.body))
        println("Found $(length(collections)) collections")
        return collections
    catch e
        @error "Failed to get collections" exception=e
        rethrow(e)
    end
end

"""
    get_patients(collection::String) -> Vector

Get all patients in a specific collection.
"""
function get_patients(collection::String)
    url = "$TCIA_API_BASE/getPatient"
    params = Dict("Collection" => collection)

    println("Querying patients in collection: $collection")

    try
        response = HTTP.get(url, query=params)
        patients = JSON3.read(String(response.body))
        println("  Found $(length(patients)) patients")
        return patients
    catch e
        @error "Failed to get patients" collection=collection exception=e
        rethrow(e)
    end
end

"""
    get_series(collection::String, patient_id::String="") -> Vector{Dict}

Get all series in a collection, optionally filtered by patient ID.
Returns Vector of Dicts (converted from JSON3 for mutability).
"""
function get_series(collection::String, patient_id::String="")
    url = "$TCIA_API_BASE/getSeries"
    params = Dict("Collection" => collection)

    if !isempty(patient_id)
        params["PatientID"] = patient_id
        println("Querying series for patient $patient_id in collection $collection...")
    else
        println("Querying all series in collection $collection...")
    end

    try
        response = HTTP.get(url, query=params)
        body = String(response.body)

        # Handle empty response
        if isempty(strip(body)) || body == "[]"
            println("  Found 0 series")
            return Dict{Symbol, Any}[]
        end

        series_json = JSON3.read(body)

        # Convert to regular Dicts for mutability
        series = [json3_to_dict(s) for s in series_json]

        println("  Found $(length(series)) series")
        return series
    catch e
        @error "Failed to get series" collection=collection patient_id=patient_id exception=e
        rethrow(e)
    end
end

"""
    query_large_series(collection::String; min_slices::Int=300, modalities::Vector{String}=["CT", "MR"]) -> Vector{Dict}

Query for large series in a collection that meet size requirements.

# Arguments
- `collection`: TCIA collection name
- `min_slices`: Minimum number of images/slices (default: 300)
- `modalities`: List of modalities to include (default: ["CT", "MR"])

# Returns
Vector of series metadata (as Dicts) sorted by size (largest first)
"""
function query_large_series(collection::String; min_slices::Int=300, modalities::Vector{String}=["CT", "MR"])
    # Get all series in collection
    all_series = get_series(collection)

    if isempty(all_series)
        println("  Filtered to 0 large series (no series found)")
        return Dict{Symbol, Any}[]
    end

    # Filter for large series
    large_series = filter(all_series) do s
        haskey(s, :Modality) && haskey(s, :ImageCount) &&
        s[:Modality] in modalities &&
        s[:ImageCount] >= min_slices
    end

    println("  Filtered to $(length(large_series)) large series (>= $min_slices slices, modalities: $(join(modalities, ", ")))")

    # Sort by total size descending (use ImageCount as fallback if TotalSizeInBytes is 0)
    if !isempty(large_series)
        sort!(large_series, by=s -> begin
            size = get(s, :TotalSizeInBytes, 0)
            # If TotalSizeInBytes is 0, use ImageCount as proxy for size
            size > 0 ? size : get(s, :ImageCount, 0) * 512000  # ~500KB per image estimate
        end, rev=true)

        # Print top series info
        println("\nTop 5 largest series:")
        for (i, s) in enumerate(large_series[1:min(5, length(large_series))])
            size_mb = get(s, :TotalSizeInBytes, 0) / 1024^2
            image_count = get(s, :ImageCount, 0)
            @printf("  %d. %s: %s, %d images, %.1f MB\n",
                    i, get(s, :SeriesInstanceUID, "unknown"),
                    get(s, :Modality, "unknown"),
                    image_count,
                    size_mb > 0 ? size_mb : image_count * 0.5)  # Estimate ~0.5 MB per image
        end
    end

    return large_series
end

"""
    download_series(series_uid::String, output_dir::String) -> String

Download a DICOM series as a ZIP file and extract it.

# Arguments
- `series_uid`: SeriesInstanceUID to download
- `output_dir`: Directory to extract DICOM files

# Returns
Path to the extracted DICOM directory
"""
function download_series(series_uid::String, output_dir::String)
    url = "$TCIA_API_BASE/getImage"
    params = Dict("SeriesInstanceUID" => series_uid)

    # Create output directory if needed
    mkpath(output_dir)

    # Download ZIP file
    zip_path = joinpath(output_dir, "$(replace(series_uid, "." => "_")).zip")
    println("\nDownloading series: $series_uid")
    println("  Downloading to: $zip_path")

    try
        # Download with progress
        HTTP.download(url, zip_path, query=params)

        # Check file size
        file_size = filesize(zip_path) / 1024^2
        @printf("  Downloaded: %.1f MB\n", file_size)

        # Extract DICOM files
        dicom_dir = joinpath(output_dir, replace(series_uid, "." => "_"))
        println("  Extracting to: $dicom_dir")
        mkpath(dicom_dir)

        zarchive = ZipFile.Reader(zip_path)
        extracted_count = 0

        for file in zarchive.files
            # Skip directories
            if !endswith(file.name, "/")
                output_path = joinpath(dicom_dir, basename(file.name))
                write(output_path, read(file))
                extracted_count += 1
            end
        end
        close(zarchive)

        println("  Extracted $extracted_count DICOM files")

        # Remove ZIP file to save space
        rm(zip_path)

        return dicom_dir

    catch e
        @error "Failed to download/extract series" series_uid=series_uid exception=e
        # Clean up partial download
        isfile(zip_path) && rm(zip_path)
        rethrow(e)
    end
end

"""
    save_metadata(series_list::Vector, output_file::String)

Save series metadata to JSON file.
"""
function save_metadata(series_list::Vector, output_file::String)
    mkpath(dirname(output_file))

    metadata = Dict(
        "download_date" => string(now()),
        "tcia_api_version" => "v4",
        "series_count" => length(series_list),
        "series" => series_list
    )

    open(output_file, "w") do f
        JSON3.write(f, metadata, allow_inf=true)
    end

    println("Saved metadata to: $output_file")
end

"""
    download_large_datasets(;
        collections::Vector{String}=["LIDC-IDRI", "TCGA-LUAD"],
        output_dir::String="benchmark_data/dicom_raw",
        max_datasets::Int=5,
        min_slices::Int=300
    )

Main function to download large datasets from TCIA.

# Arguments
- `collections`: List of TCIA collections to search
- `output_dir`: Base directory for downloaded data
- `max_datasets`: Maximum number of datasets to download
- `min_slices`: Minimum slice count for "large" series

# Returns
Vector of paths to downloaded DICOM directories
"""
function download_large_datasets(;
    collections::Vector{String}=["LIDC-IDRI", "TCGA-LUAD"],
    output_dir::String="benchmark_data/dicom_raw",
    max_datasets::Int=5,
    min_slices::Int=300
)
    println("="^80)
    println("TCIA Large Dataset Downloader")
    println("="^80)
    println("Target collections: $(join(collections, ", "))")
    println("Minimum slices: $min_slices")
    println("Max datasets: $max_datasets")
    println("Output directory: $output_dir")
    println()

    all_large_series = Dict{Symbol, Any}[]

    # Query each collection
    for collection in collections
        println("\n" * "="^80)
        println("Collection: $collection")
        println("="^80)

        try
            large_series = query_large_series(collection, min_slices=min_slices)

            # Tag with collection name (now works because we use Dicts)
            for s in large_series
                s[:Collection] = collection
            end

            append!(all_large_series, large_series)
        catch e
            @warn "Failed to query collection" collection=collection exception=e
            continue
        end
    end

    # Check if we have any series
    if isempty(all_large_series)
        println("\n" * "="^80)
        println("No large datasets found matching criteria!")
        println("="^80)
        println("Suggestions:")
        println("  - Try different collections")
        println("  - Lower min_slices threshold")
        println("  - Check TCIA API availability")
        return String[]
    end

    # Sort all series by size (use ImageCount as proxy if TotalSizeInBytes is 0)
    sort!(all_large_series, by=s -> begin
        size = get(s, :TotalSizeInBytes, 0)
        size > 0 ? size : get(s, :ImageCount, 0) * 512000
    end, rev=true)

    # Select top N for download
    to_download = all_large_series[1:min(max_datasets, length(all_large_series))]

    println("\n" * "="^80)
    println("Selected $(length(to_download)) datasets for download")
    println("="^80)

    # Calculate total size (handle empty list and estimate if TotalSizeInBytes is 0)
    if !isempty(to_download)
        total_size_bytes = sum(begin
            size = get(s, :TotalSizeInBytes, 0)
            size > 0 ? size : get(s, :ImageCount, 0) * 512000
        end for s in to_download)
        total_size_gb = total_size_bytes / 1024^3
        @printf("Estimated download size: %.2f GB\n\n", total_size_gb)
    end

    # Download series
    downloaded_paths = String[]

    for (i, series) in enumerate(to_download)
        println("[$i/$(length(to_download))]")
        series_uid = get(series, :SeriesInstanceUID, "")

        if isempty(series_uid)
            @warn "Series missing UID, skipping" series=series
            continue
        end

        try
            dicom_dir = download_series(series_uid, output_dir)
            push!(downloaded_paths, dicom_dir)
        catch e
            @warn "Failed to download series, continuing..." series_uid=series_uid exception=e
            continue
        end

        # Be respectful to TCIA servers
        sleep(2)
    end

    # Save metadata
    if !isempty(to_download)
        metadata_file = joinpath(dirname(output_dir), "metadata.json")
        save_metadata(to_download, metadata_file)
    end

    println("\n" * "="^80)
    println("Download complete!")
    println("="^80)
    println("Downloaded: $(length(downloaded_paths)) datasets")
    println("Saved to: $output_dir")

    return downloaded_paths
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    # Run as script - removed QIN-HEADNECK as it seems to have API issues
    downloaded = download_large_datasets(
        collections=["LIDC-IDRI", "TCGA-LUAD"],
        output_dir="benchmark_data/dicom_raw",
        max_datasets=5,
        min_slices=300
    )

    if !isempty(downloaded)
        println("\nDownloaded datasets:")
        for (i, path) in enumerate(downloaded)
            println("  $i. $path")
        end
    else
        println("\nNo datasets were downloaded.")
        println("Try running with --synthetic flag for benchmarks: julia run_gpu_benchmarks.jl --synthetic")
    end
end
