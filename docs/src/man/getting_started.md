
# Getting started

In this section, we outline the basic procedure to analyze your GWAS data with MendelIHT. 

## Installation

Download and install [Julia](https://julialang.org/downloads/). Within Julia, copy and paste the following:
```
using Pkg
Pkg.add(PackageSpec(url="https://github.com/OpenMendel/SnpArrays.jl.git"))
Pkg.add(PackageSpec(url="https://github.com/OpenMendel/VCFTools.jl.git"))
Pkg.add(PackageSpec(url="https://github.com/OpenMendel/MendelIHT.jl.git"))
```
`MendelIHT.jl` supports Julia 1.5+ for Mac, Linux, and window machines. A few features are disabled for windows users, and users will be warned when trying to use them.

## Typical Workflow

1. Run `cross_validate()` to determine best sparsity level (k).
2. Run `iht` on optimal `k`.

We believe the best way to learn is through examples. Head over to the example section on the left to see these steps in action. 

## Parallel computing

For large datasets, one can run cross validation in parallel. Assuming you have $N$ cores, one can load $N$ processors by
```julia
using Distributed
addprocs(4) # 4 processors
@everywhere using MendelIHT
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)
```
Note by default, BLAS runs with multiple threads, so the command `BLAS.set_num_threads(1)` sets the number of BLAS threads to 1, avoiding [oversubscription](https://ieeexplore.ieee.org/document/5470434)
