# debug_load.jl
try
    using MedImages
    println("SUCCESS")
catch e
    println("ERROR TYPE: ", typeof(e))
    println("ERROR: ", e)
    Base.display_error(stderr, e, catch_backtrace())
end
