module SUV_calc
using Dates
using ..MedImage_data_struct: MedImage, BatchedMedImage

export calculate_suv_factor

"""
    calculate_suv_factor(mi::MedImage)

Calculates the SUV factor for a single MedImage.
Returns Float64 or Nothing if calculation fails.
"""
function calculate_suv_factor(mi::MedImage)::Union{Float64, Nothing}
    return _calculate_suv_from_metadata(mi.metadata)
end

"""
    calculate_suv_factor(batched_image::BatchedMedImage)

Calculates SUV factors for a batch of images.
Returns a Vector of Union{Float64, Nothing}.
"""
function calculate_suv_factor(batched_image::BatchedMedImage)::Vector{Union{Float64, Nothing}}
    res = Vector{Union{Float64, Nothing}}(undef, length(batched_image.metadata))
    for i in 1:length(batched_image.metadata)
        res[i] = _calculate_suv_from_metadata(batched_image.metadata[i])
    end
    return res
end

function _calculate_suv_from_metadata(metadata::Dict)::Union{Float64, Nothing}
    try
        # 1. Get Patient Weight in grams
        weight_kg = get(metadata, "PatientWeight", nothing)
        if weight_kg === nothing
            return nothing
        end
        weight_g = Float64(weight_kg) * 1000.0

        # 2. Get Radionuclide Information
        radio_seq = get(metadata, "RadiopharmaceuticalInformationSequence", nothing)
        if radio_seq === nothing || isempty(radio_seq)
            return nothing
        end
        
        info = radio_seq[1]
        total_dose = get(info, "RadionuclideTotalDose", nothing)
        half_life = get(info, "RadionuclideHalfLife", nothing)
        inj_time_str = get(info, "RadiopharmaceuticalStartTime", nothing)
        
        if total_dose === nothing || half_life === nothing || inj_time_str === nothing
            return nothing
        end

        # 3. Get Scan/Acquisition Time
        # Prefer AcquisitionTime, fallback to SeriesTime
        scan_time_str = get(metadata, "AcquisitionTime", nothing)
        if scan_time_str === nothing
            scan_time_str = get(metadata, "SeriesTime", nothing)
        end
        
        if scan_time_str === nothing
            return nothing
        end

        # 4. Parse Times and calculate Delta t
        inj_time = parse_dicom_time(string(inj_time_str))
        scan_time = parse_dicom_time(string(scan_time_str))
        
        # Handle midnight crossover: if scan_time < inj_time, assume it's next day
        # Time difference in nanoseconds, convert to seconds
        delta_t = Float64((scan_time - inj_time).value) / 1e9
        if delta_t < 0
            delta_t += 86400.0 # Add 24 hours
        end

        # 5. Calculate Decay and SUV Factor
        decay = 2.0^(-delta_t / Float64(half_life))
        actual_dose = Float64(total_dose) * decay
        
        if actual_dose == 0
            return nothing
        end
        
        return weight_g / actual_dose
    catch
        return nothing
    end
end

function parse_dicom_time(t_str::AbstractString)
    # DICOM TM format: HHMMSS.ffffff (variable precision) or HHMMSS
    t_str = strip(t_str)

    try
        if contains(t_str, ".")
            parts = split(t_str, ".")
            main = parts[1]
            frac = parts[2]

            if length(main) == 6
                h = parse(Int, main[1:2])
                m = parse(Int, main[3:4])
                s = parse(Int, main[5:6])

                # Normalize frac to nanoseconds (9 digits)
                frac_len = length(frac)
                if frac_len > 9
                    frac = frac[1:9]
                else
                    frac = rpad(frac, 9, '0')
                end
                ns_total = parse(Int, frac)

                ms = div(ns_total, 1_000_000)
                rem_ns = rem(ns_total, 1_000_000)
                us = div(rem_ns, 1_000)
                ns = rem(rem_ns, 1_000)

                return Time(h, m, s, ms, us, ns)
            else
                 # Fallback for weird formats
                 return Time(t_str, dateformat"HHMMSS.s")
            end
        else
            if length(t_str) == 6
                return Time(t_str, dateformat"HHMMSS")
            else
                 # Partial times like HHMM?
                 return Time(t_str, dateformat"HHMMSS")
            end
        end
    catch
        # Last resort fallback
        return Time(t_str, dateformat"HHMMSS")
    end
end

"""
    calculate_suv_statistics(med_image::MedImage, mask::MedImage)

Calculates SUV statistics (mean and total SUV) for voxels selected by the binary mask.
Returns a NamedTuple: `(mean_suv=..., total_suv=...)`.
Throws an error if SUV factor cannot be calculated or dimensions mismatch.
The mask is treated as binary (any non-zero value is considered True).
If the mask is empty (no True voxels), returns (mean_suv=0.0, total_suv=0.0).
"""
function calculate_suv_statistics(med_image::MedImage, mask::MedImage)
    if size(med_image.voxel_data) != size(mask.voxel_data)
        throw(DimensionMismatch("Image and mask dimensions do not match"))
    end

    suv_factor = calculate_suv_factor(med_image)
    if suv_factor === nothing
        error("SUV calculation failed: Missing or invalid metadata for SUV calculation")
    end

    # Use view to avoid allocation if possible, or logical indexing
    # mask.voxel_data might be anything, assume it supports broadcasting
    mask_bool = mask.voxel_data .!= 0
    count = sum(mask_bool)

    if count == 0
        return (mean_suv=0.0, total_suv=0.0)
    end

    # Sum of pixel values in ROI
    # We can iterate or use logical indexing
    relevant_sum = sum(med_image.voxel_data[mask_bool])

    total_suv = relevant_sum * suv_factor
    mean_suv = total_suv / count

    return (mean_suv=mean_suv, total_suv=total_suv)
end

"""
    calculate_suv_statistics(batched_image::BatchedMedImage, mask::MedImage)

Calculates SUV statistics for each image in the batch using a single shared mask.
Returns a Vector of NamedTuples `(mean_suv=..., total_suv=...)`.
"""
function calculate_suv_statistics(batched_image::BatchedMedImage, mask::MedImage)
    img_size = size(batched_image.voxel_data)
    mask_size = size(mask.voxel_data)

    # Check if spatial dimensions match
    if img_size[1:3] != mask_size
        throw(DimensionMismatch("Image batch slice dimensions $(img_size[1:3]) and mask dimensions \$mask_size do not match"))
    end

    batch_size = img_size[4]
    suv_factors = calculate_suv_factor(batched_image)

    results = Vector{NamedTuple{(:mean_suv, :total_suv), Tuple{Float64, Float64}}}(undef, batch_size)

    mask_bool = mask.voxel_data .!= 0
    voxel_count = sum(mask_bool)

    for i in 1:batch_size
        factor = suv_factors[i]
        if factor === nothing
             error("SUV calculation failed for batch index \$i: Missing or invalid metadata")
        end

        if voxel_count == 0
            results[i] = (mean_suv=0.0, total_suv=0.0)
            continue
        end

        # Extract slice for current batch index
        # Using view for efficiency
        slice = view(batched_image.voxel_data, :, :, :, i)

        relevant_sum = sum(slice[mask_bool])

        total_suv = relevant_sum * factor
        mean_suv = total_suv / voxel_count

        results[i] = (mean_suv=mean_suv, total_suv=total_suv)
    end

    return results
end

"""
    calculate_suv_statistics(batched_image::BatchedMedImage, mask::BatchedMedImage)

Calculates SUV statistics for each image in the batch using a corresponding mask from the mask batch.
Returns a Vector of NamedTuples `(mean_suv=..., total_suv=...)`.
"""
function calculate_suv_statistics(batched_image::BatchedMedImage, mask::BatchedMedImage)
    img_size = size(batched_image.voxel_data)
    mask_size = size(mask.voxel_data)

    if img_size != mask_size
        throw(DimensionMismatch("Image batch dimensions \$img_size and mask batch dimensions \$mask_size do not match"))
    end

    batch_size = img_size[4]
    suv_factors = calculate_suv_factor(batched_image)

    results = Vector{NamedTuple{(:mean_suv, :total_suv), Tuple{Float64, Float64}}}(undef, batch_size)

    for i in 1:batch_size
        factor = suv_factors[i]
        if factor === nothing
             error("SUV calculation failed for batch index \$i: Missing or invalid metadata")
        end

        slice = view(batched_image.voxel_data, :, :, :, i)
        mask_slice = view(mask.voxel_data, :, :, :, i)

        mask_bool = mask_slice .!= 0
        count = sum(mask_bool)

        if count == 0
            results[i] = (mean_suv=0.0, total_suv=0.0)
            continue
        end

        relevant_sum = sum(slice[mask_bool])

        total_suv = relevant_sum * factor
        mean_suv = total_suv / count

        results[i] = (mean_suv=mean_suv, total_suv=total_suv)
    end

    return results
end

end
