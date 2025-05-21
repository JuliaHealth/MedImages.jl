using Documenter, DocumenterVitepress

makedocs(; 
    sitename = "MedImages.jl", 
    authors = "Jakub Mitura <jakub.mitura14@gmail>, Divyansh Goyal <divital2004@gmail.com>, Jan Zubik",
    format=DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/JuliaHealth/MedImages.jl", 
        devbranch = "main",
        devurl = "dev",
    ),
    warnonly = true,
    draft = false,
    source = "src",
    build = "build",
    pages=[
        "Manual" => [
            "Get Started" => "manual/get_started.md",
            "Code" => "manual/code_example.md"
        ],
        "Developers' documentation" => [
            "Image Registration" => "devs/image_registration.md"
        ],
        "api" => "api.md"
        ],
)

# This is the critical part that creates the version structure
DocumenterVitepress.deploydocs(;
    repo = "github.com/JuliaHealth/MedImages.jl", 
    devbranch = "main",
    push_preview = true,
)
