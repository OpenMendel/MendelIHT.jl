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
        "Contributing"    => "man/contributing.md",
    ]
)

deploydocs(
    repo   = "github.com/biona001/MendelIHT.jl.git",
    target = "build",
  	deps   = nothing,
  	make   = nothing,
)