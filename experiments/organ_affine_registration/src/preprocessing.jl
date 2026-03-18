module Preprocessing

using MedImages
using MedImages.MedImage_data_struct
using Statistics
using LinearAlgebra
using CUDA, KernelAbstractions, LuxCUDA

export preprocess_organ_data, OrganMetadata

struct OrganMetadata
    id::Int
    barycenter::Tuple{Float32, Float32, Float32}
    max_radius::Float32
end

"""
    preprocess_organ_data(atlas_mask::MedImage, gold_standard_mask::MedImage; max_points=512)

Prepares Atlas and Gold Standard segmentation data for the affine registration network.

# Steps
1.  **Organ Identification**: Finds all unique organ IDs in the Atlas mask.
2.  **Point Extraction**: For each organ in the Atlas, extracts voxel coordinates.
3.  **Deterministic Downsampling**:
    -   If points > `max_points`, selects a subset using a fixed stride to ensure deterministic behavior.
    -   If points < `max_points`, pads the tensor with `(-1, -1, -1)`.
4.  **Metadata Computation**: Calculates the Barycenter and Maximum Radius for each organ in the Gold Standard mask.
5.  **One-Hot Encoding**: Converts the Gold Standard mask into a multi-channel probability volume for differentiable interpolation.

# Returns
- `points_tensor`: `Array{Float32, 3}` of shape `(3, max_points, num_organs)`.
- `organ_metadata`: `Vector{OrganMetadata}` containing ID, Barycenter, and Radius.
- `gold_standard_onehot`: `Array{Float32, 4}` of shape `(X, Y, Z, num_organs)`.
"""
function preprocess_organ_data(atlas_mask::MedImage, gold_standard_mask::MedImage; max_points=512)
    # 1. Identify unique organs (integers > 0)
    # We assume atlas and gold standard have matching organ IDs
    atlas_data = atlas_mask.voxel_data
    organ_ids = sort(unique(filter(x -> x > 0, Array(atlas_data)))) # Ensure unique on CPU for simplicity
    num_organs = length(organ_ids)

    # Check if we are on GPU
    data_on_gpu = hasmethod(parent, (typeof(atlas_data),)) && parent(atlas_data) isa CuArray || atlas_data isa CuArray
    # Or more simply:
    data_on_gpu = (KernelAbstractions.get_backend(atlas_data) isa LuxCUDA.CUDABackend)

    if num_organs == 0
        error("No organs found in atlas mask")
    end

    # 2. Initialize structures
    # points_tensor: (3 coordinates, 512 points, Num_Organs)
    points_tensor = fill(-1.0f0, (3, max_points, num_organs))

    organ_meta_list = Vector{OrganMetadata}(undef, num_organs)

    # 3. Process each organ in Atlas
    for (i, oid) in enumerate(organ_ids)
        # Process on GPU if data is there
        if data_on_gpu
            mask = (atlas_data .== Int32(oid))
            indices_gpu = findall(mask)
            n_points = length(indices_gpu)
            
            if n_points == 0
                organ_meta_list[i] = OrganMetadata(Int(oid), (0f0,0f0,0f0), 0f0)
                continue
            end
            
            # Deterministic Downsampling on CPU (once we have indices)
            # Fetch only the indices we need to avoid massive CPU transfer if many points
            if n_points > max_points
                sel_indices = round.(Int, range(1, stop=n_points, length=max_points))
                # Indexing into CuArray with indices is allowed
                subset_indices = Array(indices_gpu[sel_indices])
            else
                subset_indices = Array(indices_gpu)
            end
            
            # Fill Tensor
            for (p_idx, idx) in enumerate(subset_indices)
                points_tensor[1, p_idx, i] = Float32(idx[1])
                points_tensor[2, p_idx, i] = Float32(idx[2])
                points_tensor[3, p_idx, i] = Float32(idx[3])
            end
        else
            # CPU Path
            indices = findall(x -> x == oid, atlas_data)
            if isempty(indices)
                organ_meta_list[i] = OrganMetadata(Int(oid), (0f0,0f0,0f0), 0f0)
                continue
            end
            
            n_points = length(indices)
            if n_points > max_points
                sel_indices = round.(Int, range(1, stop=n_points, length=max_points))
                subset = indices[sel_indices]
            else
                subset = indices
            end
            
            for (p_idx, idx) in enumerate(subset)
                points_tensor[1, p_idx, i] = Float32(idx[1])
                points_tensor[2, p_idx, i] = Float32(idx[2])
                points_tensor[3, p_idx, i] = Float32(idx[3])
            end
        end
    end

    # 4. Process Gold Standard for Metadata
    gold_data = gold_standard_mask.voxel_data
    dims = size(gold_data)

    # One-Hot Volume: (X, Y, Z, Num_Organs)
    # We can use Float32 for interpolation
    gold_onehot = zeros(Float32, dims[1], dims[2], dims[3], num_organs)

    for (i, oid) in enumerate(organ_ids)
        # Create one-hot channel
        # On GPU, use broadcasting for performance
        if data_on_gpu
            gold_onehot[:, :, :, i] .= Float32.(gold_data .== Int32(oid))
            
            mask = (gold_data .== Int32(oid))
            n = sum(mask)
            if n > 0
                sx, sy, sz = size(gold_data)
                # Use grids for barycenter
                x_grid = CuArray(Float32.(collect(1:sx)))
                y_grid = CuArray(Float32.(collect(1:sy)))
                z_grid = CuArray(Float32.(collect(1:sz)))
                
                sum_x = sum(Float32.(mask) .* reshape(x_grid, :, 1, 1))
                sum_y = sum(Float32.(mask) .* reshape(y_grid, 1, :, 1))
                sum_z = sum(Float32.(mask) .* reshape(z_grid, 1, 1, :))
                
                center = (sum_x/n, sum_y/n, sum_z/n)
                
                # Max Radius
                # Broadcasted distance calculation
                # (x-cx)^2 + (y-cy)^2 + (z-cz)^2
                dist_sq_vol = (((reshape(x_grid, :, 1, 1) .- center[1]).^2) .+ 
                               ((reshape(y_grid, 1, :, 1) .- center[2]).^2) .+ 
                               ((reshape(z_grid, 1, 1, :) .- center[3]).^2)) .* Float32.(mask)
                max_radius = sqrt(maximum(dist_sq_vol))
                
                organ_meta_list[i] = OrganMetadata(Int(oid), center, Float32(max_radius))
            else
                organ_meta_list[i] = OrganMetadata(Int(oid), (0f0,0f0,0f0), 1.0f0)
            end
        else
            # CPU path (Original)
            indices = findall(x -> x == oid, gold_data)
            if isempty(indices)
                 organ_meta_list[i] = OrganMetadata(Int(oid), (0f0,0f0,0f0), 0f0)
                 continue
            end
            for idx in indices
                gold_onehot[idx, i] = 1.0f0
            end
            coords = [Float32.((idx[1], idx[2], idx[3])) for idx in indices]
            sum_x = sum(c[1] for c in coords); sum_y = sum(c[2] for c in coords); sum_z = sum(c[3] for c in coords)
            n = length(coords)
            center = (sum_x/n, sum_y/n, sum_z/n)
            max_dist_sq = 0.0f0
            for p in coords
                d2 = sum((p .- center).^2)
                if d2 > max_dist_sq; max_dist_sq = d2; end
            end
            organ_meta_list[i] = OrganMetadata(Int(oid), center, sqrt(max_dist_sq))
        end
    end

    return points_tensor, organ_meta_list, gold_onehot
end

end # module
