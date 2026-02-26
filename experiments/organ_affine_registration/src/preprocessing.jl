module Preprocessing

using MedImages
using MedImages.MedImage_data_struct
using Statistics
using LinearAlgebra

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
    organ_ids = sort(unique(filter(x -> x > 0, atlas_data)))
    num_organs = length(organ_ids)

    if num_organs == 0
        error("No organs found in atlas mask")
    end

    # 2. Initialize structures
    # points_tensor: (3 coordinates, 512 points, Num_Organs)
    points_tensor = fill(-1.0f0, (3, max_points, num_organs))

    organ_meta_list = Vector{OrganMetadata}(undef, num_organs)

    # 3. Process each organ in Atlas
    for (i, oid) in enumerate(organ_ids)
        # Extract indices (CartesianIndex)
        indices = findall(x -> x == oid, atlas_data)

        if isempty(indices)
            # Should not happen based on unique check, but safety first
            organ_meta_list[i] = OrganMetadata(oid, (0f0,0f0,0f0), 0f0)
            continue
        end

        # Convert to Float32 tuples (1-based index)
        points_float = [Float32.((idx[1], idx[2], idx[3])) for idx in indices]

        # Compute Barycenter (Atlas) - used for initialization logic if needed,
        # but here we focus on processing points.
        # Actually, the prompt says "calculate Gold Standard Barycenters and Max Radii".
        # But we also need points from Atlas.

        # Deterministic Downsampling
        n_points = length(points_float)
        if n_points > max_points
            # Stride based selection
            # We want exactly max_points? Or at most?
            # Prompt: "fill those with -1" -> implies fixed size buffer.
            # "select every k-th point"
            step = n_points / max_points
            selected_indices = [Int(ceil(j * step)) for j in 1:max_points]
            # Clamp to be safe
            selected_indices = clamp.(selected_indices, 1, n_points)
            # Ensure unique if possible? With ceil, might repeat last one.
            # Better: range(1, n_points, length=max_points)
            selected_indices = round.(Int, range(1, stop=n_points, length=max_points))

            subset = points_float[selected_indices]
        else
            subset = points_float
        end

        # Fill Tensor
        for (p_idx, p) in enumerate(subset)
            points_tensor[1, p_idx, i] = p[1]
            points_tensor[2, p_idx, i] = p[2]
            points_tensor[3, p_idx, i] = p[3]
        end
        # Remaining are already -1.0f0
    end

    # 4. Process Gold Standard for Metadata
    gold_data = gold_standard_mask.voxel_data
    dims = size(gold_data)

    # One-Hot Volume: (X, Y, Z, Num_Organs)
    # We can use Float32 for interpolation
    gold_onehot = zeros(Float32, dims[1], dims[2], dims[3], num_organs)

    for (i, oid) in enumerate(organ_ids)
        # Create one-hot channel
        # We can optimize this by iterating volume once if needed, but per-organ is clear
        indices = findall(x -> x == oid, gold_data)

        if isempty(indices)
             # Warning: Gold standard missing organ present in Atlas?
             # Set metadata to 0
             organ_meta_list[i] = OrganMetadata(oid, (0f0,0f0,0f0), 0f0)
             continue
        end

        # Fill One-Hot
        for idx in indices
            gold_onehot[idx, i] = 1.0f0
        end

        # Compute Barycenter
        coords = [Float32.((idx[1], idx[2], idx[3])) for idx in indices]
        # Tuple mean is not standard in Statistics for simple divide
        # Calculate manually
        sum_x = sum(c[1] for c in coords)
        sum_y = sum(c[2] for c in coords)
        sum_z = sum(c[3] for c in coords)
        n = length(coords)
        center = (sum_x/n, sum_y/n, sum_z/n)

        # Compute Max Radius
        # Max distance from center to any voxel in the organ
        max_dist_sq = 0.0f0
        for p in coords
            d2 = sum((p .- center).^2)
            if d2 > max_dist_sq
                max_dist_sq = d2
            end
        end
        max_radius = sqrt(max_dist_sq)

        organ_meta_list[i] = OrganMetadata(oid, center, max_radius)
    end

    return points_tensor, organ_meta_list, gold_onehot
end

end # module
