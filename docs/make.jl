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
        "Home" => "index.md",
        "Manual" => [
            "Getting Started" => "manual/get_started.md",
            "Tutorials" => "manual/tutorials.md",
            "Code Examples" => "manual/code_example.md",
            "Coordinate Systems" => "manual/coordinate_systems.md"
        ],
        "Reference" => [
            "API Reference" => "api.md",
            "Data Structures" => "reference/data_structures.md"
        ],
        "Developers" => [
            "Image Registration" => "devs/image_registration.md"
        ]
    ],
)

# This is the critical part that creates the version structure
DocumenterVitepress.deploydocs(;
    repo = "github.com/JuliaHealth/MedImages.jl",
    devbranch = "main",
    push_preview = true,
)
