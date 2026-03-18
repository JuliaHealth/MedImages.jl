using HDF5

hdf5_path = "/mnt/vast-kisski/projects/ovgu_medicine_llm/ollama_data/dataset_unified_for_registration.h5"
if !isfile(hdf5_path)
    println("File not found: $hdf5_path")
    exit()
end

println("Opening $hdf5_path...")
f = h5open(hdf5_path, "r")
try
    root_keys = keys(f)
    println("Root keys: ", root_keys)
    
    if "train_list" in root_keys
        train_list = read(f["train_list"])
        println("Num subjects in train_list: ", length(train_list))
        
        if length(train_list) > 0
            first_pat = train_list[1]
            println("First patient in list: ", first_pat)
            if haskey(f, first_pat)
                println("Keys for $first_pat: ", keys(f[first_pat]))
            else
                println("Patient $first_pat NOT found in HDF5 root!")
                # Print a few available keys to see format
                println("First 5 root keys: ", root_keys[1:min(5, end)])
            end
        end
    else
        println("KEY 'train_list' NOT FOUND in root!")
    end
finally
    close(f)
end
