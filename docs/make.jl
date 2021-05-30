using Documenter, MendelIHT

makedocs(
    doctest = false, #will set to true later
    format = Documenter.HTML(),
    sitename = "MendelIHT",
    authors = "Benjamin Chu, Kevin Keys",
    clean = true,
    pages = [
        "Home"            => "index.md",
        "Getting Started" => "man/getting_started.md",
        "Examples"        => "man/examples.md",
        "Mathematical Details" => "man/math.md",
        "Contributing"    => "man/contributing.md",
        "FAQ"             => "man/FAQ.md",
        "API"             => "man/api.md",
    ]
)

deploydocs(
    repo   = "github.com/OpenMendel/MendelIHT.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing,
)
