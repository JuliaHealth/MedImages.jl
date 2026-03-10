module SUV_calc
using Dates
using ..MedImage_data_struct: MedImage, BatchedMedImage

export calculate_suv_factor

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

"""
    calculate_suv_factor(med_image::MedImage)

Calculates the SUV factor for a single image.
Returns a Union{Float64, Nothing}.
"""
function calculate_suv_factor(med_image::MedImage)::Union{Float64, Nothing}
    return _calculate_suv_from_metadata(med_image.metadata)
end

function _calculate_suv_from_metadata(meta::Dict{Any, Any})::Union{Float64, Nothing}
    # 1. Get Patient Weight in kg
    weight = get(meta, "PatientWeight", nothing)
    if weight === nothing
        println("DEBUG: Missing PatientWeight")
        return nothing
    end
    weight_kg = parse(Float64, string(weight))

    # 2. Extract Radiopharmaceutical Information
    radio_seq_arr = get(meta, "RadiopharmaceuticalInformationSequence", nothing)
    if radio_seq_arr === nothing || isempty(radio_seq_arr)
        println("DEBUG: Missing RadiopharmaceuticalInformationSequence")
        return nothing
    end
    # PyDicom sequences are usually arrays of dicts
    radio_seq = radio_seq_arr[1]

    inj_dose = get(radio_seq, "RadionuclideTotalDose", nothing)
    half_life = get(radio_seq, "RadionuclideHalfLife", nothing)
    if inj_dose === nothing || half_life === nothing
        println("DEBUG: Missing RadionuclideTotalDose or RadionuclideHalfLife")
        return nothing
    end
    
    inj_dose = parse(Float64, string(inj_dose))
    half_life = parse(Float64, string(half_life))

    inj_time = get(radio_seq, "RadiopharmaceuticalStartTime", nothing)
    if inj_time === nothing
        println("DEBUG: Missing RadiopharmaceuticalStartTime")
        return nothing
    end
    inj_time_str = string(inj_time)
    
    # 3. Determine Scan (Acquisition) Time
    scan_time_str = get(meta, "AcquisitionTime", get(meta, "SeriesTime", nothing))
    if scan_time_str === nothing
        println("DEBUG: Missing AcquisitionTime and SeriesTime")
        return nothing
    end
    scan_time_str = string(scan_time_str)

    # 4. Parse Times and Calculate Decay
    try
        t_inj = parse_dicom_time(inj_time_str)
        t_scan = parse_dicom_time(scan_time_str)

        s_inj = Dates.value(t_inj) / 1.0e9
        s_scan = Dates.value(t_scan) / 1.0e9

        delta_s = s_scan - s_inj
        if delta_s < 0
            delta_s += 24 * 3600
        end

        decayed_dose = inj_dose * exp(-log(2) * delta_s / half_life)

        if decayed_dose > 0
            return (weight_kg * 1000.0) / decayed_dose
        end
    catch e
        println("DEBUG: exception during time parsing or decay: ", e)
        return nothing
    end
    return nothing
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
