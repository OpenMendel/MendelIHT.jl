
# Getting started

In this section, we outline the basic procedure to analyze your GWAS data with MendelIHT. 

## Installation

Download and install [Julia](https://julialang.org/downloads/). Within Julia, copy and paste the following:
```
using Pkg
pkg"add https://github.com/OpenMendel/SnpArrays.jl"
pkg"add https://github.com/OpenMendel/MendelIHT.jl"
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
@everywhere begin
    using MendelIHT
    using LinearAlgebra
    BLAS.set_num_threads(1)
end
```
Note by default, BLAS runs with multiple threads, so the command `BLAS.set_num_threads(1)` sets the number of BLAS threads to 1, avoiding [oversubscription](https://ieeexplore.ieee.org/document/5470434)

## Running from command line as script

If you don't want to run MendelIHT.jl in a Julia session (e.g. you want to run batch jobs on a cluster), you can do so by putting the code below in a Julia file. For example, in order to run with 8 cores, create a file called `iht.jl` which contains:

```julia
# place these code in a file called iht.jl
using Distributed
addprocs(4) # use 4 cores
@everywhere using MendelIHT

# setup code goes here
plinkfile = ARGS[1]     # 1st command line argument (plink file location)
covariates = ARGS[2]    # 2nd command line argument (covariate file location)
path = 5:5:100          # test k = 5, 10, 15, ... 100

# run MendelIHT: first cross validate for best k, then run IHT using best k
mses = cross_validate(plinkfile, Normal, covariates=covariates, path=path)
iht_result = iht(plinkfile, Normal, k=path[argmin(mses)])
```

Then in the terminal you can do:
```shell
julia iht.jl plinkfile covariates.txt
```
You should get progress printed to your terminal and have `cviht.summary.txt`, `iht.summary.txt`, and `iht.beta.txt` files saved to your local directory
