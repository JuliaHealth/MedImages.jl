using MedImages
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(MedImages, :DocTestSetup, :(using MedImages); recursive=true)

pgs = [ "Home" => "index.md",
        "Tutorials" => "tutorials.md",
        "Contributing" => "contributing.md"
       ]

fmt = DocumenterVitepress.MarkdownVitepress(
      repo = "https://github.com/Juliahealth/MedImages.jl",)


makedocs(;
    modules=[MedImages],
    repo=Remotes.GitHub("JuliaHealth", "MedImages.jl"),
    authors="Jakub-Mitura <jakubmitura14@gmail.com>, Divyansh-Goyal <divital2004@gmail.com>, Jan-Zubik and contributors",
    sitename="MedImages.jl",
    format=fmt,
    pages=pgs
   )    
deploydocs(;
    repo="github.com/JuliaHealth/MedImages.jl",
    target="build", # this is where Vitepress stores its output
    devbranch="main",
    branch="gh-pages",
    push_preview=true
)
