using Documenter, MendelIHT

makedocs(
    format = :html,
    sitename = "MendelIHT",
    modules = [MendelIHT],
    authors = "Benjamin Chu, Kevin Keys",
)

deploydocs(
    repo   = "https://github.com/biona001/MendelIHT.jl.git",
    target = "build"
)