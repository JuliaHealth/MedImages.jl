module Normalization

using ..MedImage_data_struct: MedImage, BatchedMedImage
using Accessors
using Statistics
using ChainRulesCore

export z_score_normalize, min_max_normalize, apply_dicom_rescale
export histogram_match, nyul_train, nyul_transform

"""
    z_score_normalize(mi::MedImage; mask=nothing, eps=1e-8)
    z_score_normalize(bmi::BatchedMedImage; mask=nothing, eps=1e-8)

Performs Z-score normalization: (x - mean) / std.
If a mask is provided, statistics are calculated only over the masked region.
"""
function z_score_normalize(mi::MedImage; mask=nothing, eps=1e-8)
    data = mi.voxel_data
    if isnothing(mask)
        m = mean(data)
        s = std(data)
    else
        m = mean(data[mask])
        s = std(data[mask])
    end
    
    # Non-mutating broadcast for AD compatibility
    new_voxel_data = (data .- m) ./ (s + eps)
    
    new_mi = copy_with_new_data(mi, new_voxel_data)
    return new_mi
end

function z_score_normalize(bmi::BatchedMedImage; mask=nothing, eps=1e-8)
    data = bmi.voxel_data # (x, y, z, batch)
    
    batch_size = size(data)[end]
    
    # Map each volume in the batch
    new_volumes = map(1:batch_size) do i
        vol = data[:, :, :, i]
        if isnothing(mask)
            m = mean(vol)
            s = std(vol)
        else
            m = mean(vol[mask[:, :, :, i]])
            s = std(vol[mask[:, :, :, i]])
        end
        (vol .- m) ./ (s + eps)
    end
    
    # Concatenate back to 4D
    new_voxel_data = cat(new_volumes..., dims=4)
    
    new_bmi = copy_with_new_data(bmi, new_voxel_data)
    return new_bmi
end

"""
    min_max_normalize(mi::MedImage; range=(0.0, 1.0), eps=1e-8)
    min_max_normalize(bmi::BatchedMedImage; range=(0.0, 1.0), eps=1e-8)

Scales intensities to the specified range [min_val, max_val].
"""
function min_max_normalize(mi::MedImage; range=(0.0, 1.0), eps=1e-8)
    data = mi.voxel_data
    min_d = minimum(data)
    max_d = maximum(data)
    
    new_voxel_data = (data .- min_d) ./ (max_d - min_d + eps) .* (range[2] - range[1]) .+ range[1]
    
    return copy_with_new_data(mi, new_voxel_data)
end

function min_max_normalize(bmi::BatchedMedImage; range=(0.0, 1.0), eps=1e-8)
    data = bmi.voxel_data
    batch_size = size(data)[end]
    
    new_volumes = map(1:batch_size) do i
        vol = data[:, :, :, i]
        min_d = minimum(vol)
        max_d = maximum(vol)
        (vol .- min_d) ./ (max_d - min_d + eps) .* (range[2] - range[1]) .+ range[1]
    end
    
    new_voxel_data = cat(new_volumes..., dims=4)
    return copy_with_new_data(bmi, new_voxel_data)
end

"""
    apply_dicom_rescale(mi::MedImage; slope=nothing, intercept=nothing)

Applies DICOM Rescale Slope and Intercept: output = pixel_value * slope + intercept.
"""
function apply_dicom_rescale(mi::MedImage; slope=nothing, intercept=nothing)
    s = isnothing(slope) ? get(mi.metadata, "RescaleSlope", 1.0) : slope
    i = isnothing(intercept) ? get(mi.metadata, "RescaleIntercept", 0.0) : intercept
    
    new_voxel_data = mi.voxel_data .* s .+ i
    return copy_with_new_data(mi, new_voxel_data)
end

"""
    histogram_match(source::AbstractArray, target::AbstractArray)

Differentiable implementation of histogram matching.
Uses sorting-based matching.
"""
function histogram_match(source::AbstractArray, target::AbstractArray)
    s_flat = reshape(source, :)
    t_flat = reshape(target, :)
    
    s_sorted = sort(s_flat)
    t_sorted = sort(t_flat)
    
    # If target has different length, interpolate
    if length(s_sorted) != length(t_sorted)
        indices = range(1, length(t_sorted), length=length(s_sorted))
        t_interp = interpolate_sorted(t_sorted, indices)
    else
        t_interp = t_sorted
    end
    
    # Map back. Use sortperm and indexing instead of mutation and .=
    p = sortperm(s_flat)
    rev_p = invperm(p)
    matched_flat = t_interp[rev_p]
    
    return reshape(matched_flat, size(source))
end

function interpolate_sorted(vals, indices)
    # Basic linear interpolation for sorted values
    lower = floor.(Int, indices)
    upper = ceil.(Int, indices)
    frac = indices .- lower
    
    # Clamp to avoid bounds error
    lower = clamp.(lower, 1, length(vals))
    upper = clamp.(upper, 1, length(vals))
    
    return (1 .- frac) .* vals[lower] .+ frac .* vals[upper]
end

# Support for MedImage
function histogram_match(source_mi::MedImage, target_mi::MedImage)
    new_voxel_data = histogram_match(source_mi.voxel_data, target_mi.voxel_data)
    return copy_with_new_data(source_mi, new_voxel_data)
end


"""
    nyul_train(images::Vector{MedImage}; percentiles=1:99)

Trains Nyul's landmaks from a set of images.
Returns (percentiles, standard_landmarks).
"""
function nyul_train(images::Vector{MedImage}; percentiles=1:99)
    all_landmarks = []
    for mi in images
        data = mi.voxel_data
        # Typically ignore values below a threshold (e.g., background)
        valid_data = data[data .> 0.0]
        if isempty(valid_data)
            valid_data = data # Fallback if all 0
        end
        push!(all_landmarks, quantile(reshape(valid_data, :), percentiles ./ 100.0))
    end
    standard_landmarks = mean(all_landmarks)
    return (percentiles, standard_landmarks)
end

"""
    nyul_transform(mi::MedImage, train_results)

Applies Nyul's landmaks.
"""
function nyul_transform(mi::MedImage, train_results)
    percentiles, standard_landmarks = train_results
    data = mi.voxel_data
    
    valid_data = data[data .> 0.0]
    if isempty(valid_data)
        valid_data = data
    end
    source_landmarks = quantile(reshape(valid_data, :), percentiles ./ 100.0)
    
    new_voxel_data = piecewise_linear_transform(data, source_landmarks, standard_landmarks)
    return copy_with_new_data(mi, new_voxel_data)
end

function piecewise_linear_transform(data, source_landmarks, target_landmarks)
    # Piecewise linear mapping
    # To handle the mapping efficiently, we can pre-calculate the slope and intercept for each interval
    
    slopes = (target_landmarks[2:end] .- target_landmarks[1:end-1]) ./ 
             (source_landmarks[2:end] .- source_landmarks[1:end-1] .+ 1e-8)
    intercepts = target_landmarks[1:end-1] .- slopes .* source_landmarks[1:end-1]

    function map_val(x)
        if x <= source_landmarks[1]
            return target_landmarks[1]
        elseif x >= source_landmarks[end]
            return target_landmarks[end]
        end
        # Find interval
        i = searchsortedfirst(source_landmarks, x) - 1
        i = clamp(i, 1, length(slopes))
        return slopes[i] * x + intercepts[i]
    end

    return map_val.(data)
end

# Helper to deepcopy and update voxel data
function copy_with_new_data(mi::MedImage, new_data)
    return @set mi.voxel_data = new_data
end

function copy_with_new_data(bmi::BatchedMedImage, new_data)
    return @set bmi.voxel_data = new_data
end

end # module
