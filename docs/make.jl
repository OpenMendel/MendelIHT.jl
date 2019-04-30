using Documenter, MendelIHT

makedocs(
    format = Documenter.HTML(),
    sitename = "MendelIHT",
    authors = "Benjamin Chu, Kevin Keys",
    clean = true,
    branch = "gh-pages",
    devbranch = "master",
    devurl = "dev",
    page = [
        "index.md"
    ]
)

deploydocs(
    repo   = "https://github.com/biona001/MendelIHT.jl.git",
    target = "build"
)