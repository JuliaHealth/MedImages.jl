using MedImages
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(MedImages, :DocTestSetup, :(using MedImages); recursive=true)

makedocs(;
    modules=[MedImages],
    repo=Remotes.GitHub("JuliaHealth", "MedImage.jl"),
    authors="Jakub-Mitura <jakubmitura14@gmail.com>, Divyansh-Goyal <divital2004@gmail.com> and contributors",
    sitename="MedImages.jl",
    format=DocumenterVitepress.MarkdownVitepress(
        repo="https://github.com/JuliaHealth/MedImage.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => "tutorials.md",
        "Contributing" => "contributing.md"
    ],
)

deploydocs(;
    repo="github.com/JuliaHealth/MedImage.jl",
    target="build", # this is where Vitepress stores its output
    devbranch="main",
    branch="gh-pages",
    push_preview=true
)