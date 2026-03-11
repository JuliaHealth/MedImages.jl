using MedImages
using MLUtils
using Statistics
using Random

function load_patient_data(patient_dir::String; target_size=(64, 64, 64))
    spect_recon_path = joinpath(patient_dir, "SPECT_DATA", "SPECT_Recon_WholeBody.nii.gz")
    ct_path = joinpath(patient_dir, "SPECT_DATA", "CT.nii.gz")
    dose_path = joinpath(patient_dir, "SPECT_DATA", "Dosemap.nii.gz")

    if !isfile(spect_recon_path) || !isfile(ct_path) || !isfile(dose_path)
        @warn "Missing files in $patient_dir"
        return nothing
    end

    # Load images
    println("  Loading SPECT...")
    spect = load_image(spect_recon_path, "PET")
    println("  Loading CT...")
    ct = load_image(ct_path, "CT")
    println("  Loading Dosemap...")
    dose = load_image(dose_path, "PET")

    println("  Raw sizes: SPECT=$(size(spect.voxel_data)) CT=$(size(ct.voxel_data)) Dose=$(size(dose.voxel_data))")

    # 1. Resample SPECT to target spacing/size
    spect_extent = size(spect.voxel_data) .* spect.spacing
    target_spacing = (Float64(spect_extent[1] / target_size[1]),
                      Float64(spect_extent[2] / target_size[2]),
                      Float64(spect_extent[3] / target_size[3]))

    println("  Resampling SPECT to spacing $target_spacing...")
    spect_res = resample_to_spacing(spect, target_spacing, Linear_en)

    # 2. Resample CT and Dosemap to match the resampled SPECT grid
    #    resample_to_image(fixed, moving, interp) -> resamples moving to match fixed
    println("  Resampling CT to SPECT grid...")
    ct_res = resample_to_image(spect_res, ct, Linear_en)
    println("  Resampling Dosemap to SPECT grid...")
    dose_res = resample_to_image(spect_res, dose, Linear_en)

    println("  Resampled sizes: SPECT=$(size(spect_res.voxel_data)) CT=$(size(ct_res.voxel_data)) Dose=$(size(dose_res.voxel_data))")

    # 3. Extract voxel data and crop/pad to exact target_size
    function get_sized_data(im, sz)
        data = Float32.(im.voxel_data)
        actual = size(data)
        # Crop to at most sz in each dimension
        r1 = 1:min(actual[1], sz[1])
        r2 = 1:min(actual[2], sz[2])
        r3 = 1:min(actual[3], sz[3])
        cropped = data[r1, r2, r3]
        # Pad with zeros if smaller
        if size(cropped) != sz
            padded = zeros(Float32, sz...)
            padded[1:size(cropped,1), 1:size(cropped,2), 1:size(cropped,3)] .= cropped
            return padded
        end
        return cropped
    end

    spect_data = get_sized_data(spect_res, target_size)
    ct_data = get_sized_data(ct_res, target_size)
    dose_data = get_sized_data(dose_res, target_size)

    # 4. Normalization
    # SPECT: normalize to [0, 1]
    spect_max = maximum(abs.(spect_data))
    if spect_max > 0
        spect_data ./= spect_max
    end

    # Dosemap: normalize to [0, 1]
    dose_max = maximum(abs.(dose_data))
    if dose_max > 0
        dose_data ./= dose_max
    end

    # CT: clip HU to [-1000, 2000] then normalize to [0, 1]
    ct_data = clamp.(ct_data, -1000f0, 2000f0)
    ct_min = minimum(ct_data)
    ct_data .-= ct_min
    ct_max = maximum(ct_data)
    if ct_max > 0
        ct_data ./= ct_max
    end

    println("  Normalized ranges: SPECT=[$(minimum(spect_data)), $(maximum(spect_data))], CT=[$(minimum(ct_data)), $(maximum(ct_data))], Dose=[$(minimum(dose_data)), $(maximum(dose_data))]")
    return spect_data, ct_data, dose_data
end

function create_lu_data_loader(base_dir::String; num_samples=8, target_size=(64, 64, 64), batchsize=2)
    patient_dirs = [joinpath(base_dir, d) for d in readdir(base_dir) if isdir(joinpath(base_dir, d))]

    # Filter for dirs that have SPECT_DATA
    patient_dirs = filter(d -> isdir(joinpath(d, "SPECT_DATA")), patient_dirs)

    if isempty(patient_dirs)
        error("No patient data found in $base_dir")
    end

    # Limit to num_samples
    n = min(length(patient_dirs), num_samples)
    patient_dirs = patient_dirs[1:n]

    W, H, D = target_size
    X = zeros(Float32, W, H, D, 2, n)
    Y = zeros(Float32, W, H, D, 1, n)

    loaded = 0
    println("Loading $n patients from $base_dir...")
    for i in 1:n
        println("Patient $i/$(n): $(basename(patient_dirs[i]))")
        data = load_patient_data(patient_dirs[i]; target_size=target_size)
        if isnothing(data)
            continue
        end
        spect, ct, dose = data
        X[:, :, :, 1, i] .= spect
        X[:, :, :, 2, i] .= ct
        Y[:, :, :, 1, i] .= dose
        loaded += 1
        println("  ✓ Loaded successfully")
    end
    println("Successfully loaded $loaded/$n patients.")

    return DataLoader((X, Y); batchsize=batchsize, shuffle=true)
end
