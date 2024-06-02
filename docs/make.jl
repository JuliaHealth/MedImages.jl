using MedImage3D
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(MedImage3D, :DocTestSetup, :(using MedImage3D); recursive=true)

makedocs(;
    modules = [MedImage3D],
    repo = Remotes.GitHub("JuliaHealth", "MedImage.jl"),
    authors = "Jakub-Mitura <jakubmitura14@gmail.com>, Divyansh-Goyal <divital2004@gmail.com> and contributors",
    sitename = "MedImage3D.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo="https://github.com/JuliaHealth/MedImage.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => "tutorials.md",
        "Contributing" => "contributing.md"
    ],
)

deploydocs(;
    repo = "github.com/JuliaHealth/MedImage.jl",
    target = "build", # this is where Vitepress stores its output
    devbranch = "main",
    branch = "gh-pages",
    push_preview = true
)