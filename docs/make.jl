using Documenter, MendelIHT

makedocs(
    format = Documenter.HTML(),
    sitename = "MendelIHT",
    authors = "Benjamin Chu, Kevin Keys",
    clean = true,
    pages = [
		"Home" => "index.md"
    ]
)

deploydocs(
    repo   = "https://github.com/biona001/MendelIHT.jl.git",
    target = "build",
    branch = "gh-pages",
  	deps   = nothing,
  	make   = nothing
)