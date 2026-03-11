using HDF5

h5_path = "/mnt/vast-kisski/projects/ovgu_medicine_llm/ollama_data/dataset_unified_for_registration.h5"

fid = h5open(h5_path, "r")
train_list = read(fid["train_list"])

println("Checking $(length(train_list)) subjects...")

missing_image = []
for pat_id in train_list
    if !haskey(fid, pat_id)
        println("Patient not found in file: $pat_id")
        continue
    end
    grp = fid[pat_id]
    if !haskey(grp, "image")
        push!(missing_image, (pat_id, keys(grp)))
    end
end

if isempty(missing_image)
    println("All patients have 'image' key.")
else
    println("Found $(length(missing_image)) patients missing 'image' key:")
    for (id, ks) in missing_image
        println("- $id: $ks")
    end
end

close(fid)
