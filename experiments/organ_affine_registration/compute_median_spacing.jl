using MedImages
using MedImages.Load_and_save
using Statistics
using ProgressMeter

function compute_median_spacing(dataset_path="/home/jm/project_ssd/MedImages.jl/test_data/dataset_PET")
    # Find all NIfTI files
    find_cmd = `find $dataset_path -name "*.nii.gz"`
    all_files = readlines(pipeline(find_cmd))
    
    if isempty(all_files)
        println("No valid subjects found.")
        return
    end
    
    # Sample up to 100 files
    sample_size = min(100, length(all_files))
    sampled_files = all_files[1:sample_size]
    
    spacings = []
    
    @showprogress "Reading spacings... " for path in sampled_files
        try
            img = load_image(path, "PET")
            push!(spacings, img.spacing)
        catch e
            # Skip errors
        end
    end
    
    if isempty(spacings)
        println("No valid spacings found.")
        return
    end
    
    sx = [s[1] for s in spacings]
    sy = [s[2] for s in spacings]
    sz = [s[3] for s in spacings]
    
    med_spacing = (median(sx), median(sy), median(sz))
    println("\nMedian Spacing for Dataset (Sampled $sample_size): $med_spacing")
    return med_spacing
end

if abspath(PROGRAM_FILE) == @__FILE__
    compute_median_spacing()
end
