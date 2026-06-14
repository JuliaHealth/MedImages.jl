using Serialization, ComponentArrays, CUDA, Lux
checkpoint_path = "data/checkpoints/UDE_NO_APPROX_64/model_best_UDE_NO_APPROX_64.jls"
println("Loading $checkpoint_path...")
θ = deserialize(checkpoint_path)
println("Type of θ: ", typeof(θ))
if θ isa ComponentArray
    println("Keys: ", keys(θ))
    # Recursive display of keys for nested structures
    function print_keys(obj, indent="")
        if obj isa ComponentArray || obj isa NamedTuple
            try
                for k in keys(obj)
                    val = getproperty(obj, k)
                    println(indent, k, " (", typeof(val), ")")
                    print_keys(val, indent * "  ")
                end
            catch e
                # Maybe not indexable
            end
        end
    end
    print_keys(θ)
else
    println("θ is not a ComponentArray")
    println(θ)
end
