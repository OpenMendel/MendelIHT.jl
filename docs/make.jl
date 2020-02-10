using Documenter, MendelIHT

makedocs(
    doctest = false, #will set to true later
    format = Documenter.HTML(),
    sitename = "MendelIHT",
    authors = "Benjamin Chu, Kevin Keys",
    clean = true,
    pages = [
        "Home"            => "index.md",
        "What is IHT?"    => "man/introduction.md",
        "Getting Started" => "man/getting_started.md",
        "Examples"        => "man/examples.md",
        "Contributing"    => "man/contributing.md",
    ]
)

deploydocs(
    repo   = "github.com/OpenMendel/MendelIHT.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing,
)
