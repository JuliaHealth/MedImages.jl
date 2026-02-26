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
    # BatchedMedImage has metadata as Vector{Dict}
    # We map the calculation over the vector
    # But calculate_suv_factor(MedImage) expects a MedImage.
    # We can reconstruct temporary "MedImage-like" objects or refactor logic.
    # Refactoring logic is better to avoid allocations.

    # Actually, we can reuse the logic if we extract metadata for each.
    res = Vector{Union{Float64, Nothing}}(undef, length(batched_image.metadata))
    for i in 1:length(batched_image.metadata)
        # Create a mock wrapper or just pass metadata?
        # The function calculate_suv_factor(MedImage) accesses med_image.metadata.
        # Let's extract the core logic to a helper that takes Dict.
        res[i] = _calculate_suv_from_metadata(batched_image.metadata[i])
    end
    return res
end

# Internal helper to share logic
function _calculate_suv_from_metadata(meta::Dict{Any, Any})::Union{Float64, Nothing}
    # 1. Retrieve Patient Weight
    if !haskey(meta, "PatientWeight") || meta["PatientWeight"] === nothing
        # @warn "SUV calculation failed: Missing PatientWeight" # Reduce spam in loops
        return nothing
    end
    weight_g = Float64(meta["PatientWeight"]) * 1000.0

    # 2. Retrieve Radiopharmaceutical Information
    if !haskey(meta, "RadiopharmaceuticalInformationSequence") || isempty(meta["RadiopharmaceuticalInformationSequence"])
        return nothing
    end

    radio_seq = meta["RadiopharmaceuticalInformationSequence"][1]

    if !haskey(radio_seq, "RadionuclideTotalDose") || !haskey(radio_seq, "RadionuclideHalfLife") || !haskey(radio_seq, "RadiopharmaceuticalStartTime")
        return nothing
    end

    inj_dose_bq = Float64(radio_seq["RadionuclideTotalDose"])
    half_life_s = Float64(radio_seq["RadionuclideHalfLife"])
    inj_time_str = string(radio_seq["RadiopharmaceuticalStartTime"])

    # 3. Determine Scan (Acquisition) Time
    scan_time_str = get(meta, "AcquisitionTime", get(meta, "SeriesTime", nothing))
    if scan_time_str === nothing
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

        decay_factor = 2.0^(-delta_s / half_life_s)
        actual_dose = inj_dose_bq * decay_factor

        if actual_dose == 0
            return nothing
        end

        return weight_g / actual_dose

    catch e
        return nothing
    end
end

# Redirect MedImage call to helper to keep DRY
calculate_suv_factor(med_image::MedImage) = _calculate_suv_from_metadata(med_image.metadata)


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

end
