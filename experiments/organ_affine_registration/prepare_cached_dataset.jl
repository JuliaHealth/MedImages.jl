using MedImages
using MedImages.Load_and_save
using MedImages.Spatial_metadata_change
using MedImages.MedImage_data_struct
using HDF5
using ProgressMeter

function prepare_cache(n_subjects=16)
    lu_path = "/home/jm/project_ssd/MedImages.jl/test_data/dataset_Lu"
    subjects = filter(pat -> startswith(pat, "FDM"), readdir(lu_path))
    if length(subjects) < n_subjects
        n_subjects = length(subjects)
    end
    subjects = subjects[1:n_subjects]
    
    h5_path = "/media/jm/hddData/projects_new/MedImages.jl/experiments/organ_affine_registration/cache_fdm_16.h5"
    
    h5open(h5_path, "w") do h5
        @showprogress "Caching subjects... " for pat in subjects
            try
                # Paths
                img_path = joinpath(lu_path, pat, "SPECT_DATA", "SPECT_Recon_WholeBody.nii.gz")
                seg_path = joinpath(lu_path, pat, "TOTAL_SEG", "seg.nii.gz")
                
                if !isfile(img_path) || !isfile(seg_path)
                    println("Skipping $pat, files missing.")
                    continue
                end
                
                # Load
                img = load_image(img_path, "PET")
                seg = load_image(seg_path, "CT") # Segments are often saved as CT (int)
                
                # Reorient to RAS
                img_ras = change_orientation(img, ORIENTATION_RAS)
                seg_ras = change_orientation(seg, ORIENTATION_RAS)
                
                # Create Group
                g = create_group(h5, pat)
                
                # Write voxel data (Float32 to match pipeline)
                g["image"] = Float32.(img_ras.voxel_data)
                g["segmentation"] = Float32.(seg_ras.voxel_data)
                
                # Write spatial metadata
                attributes(g)["origin"] = collect(img_ras.origin)
                attributes(g)["spacing"] = collect(img_ras.spacing)
                attributes(g)["direction"] = collect(img_ras.direction)
                
            catch e
                println("Error processing $pat: $e")
            end
        end
    end
    println("Cache created at $h5_path")
end

prepare_cache(16)
