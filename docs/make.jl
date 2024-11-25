using LogitSamplers
using Documenter

DocMeta.setdocmeta!(LogitSamplers, :DocTestSetup, :(using LogitSamplers); recursive=true)

makedocs(;
    modules=[LogitSamplers],
    authors="murrellb <murrellb@gmail.com> and contributors",
    sitename="LogitSamplers.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/LogitSamplers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/LogitSamplers.jl",
    devbranch="main",
)
