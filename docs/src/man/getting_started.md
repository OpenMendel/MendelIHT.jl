
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

1. Run [cross_validate](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.cross_validate) or [cv_iht](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.cv_iht) to determine best sparsity level (k).
2. Run [iht](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.iht) or [fit_iht](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.fit_iht) on optimal `k` determined from cross validation. 

We believe the best way to learn is through examples. Head over to the example section on the left to see these steps in action. 

## Parallel computing

Assuming you have 4 cores, one can load 4 processors by

!!! note
    If you prefer to use the environment variable you can set it as follows in
    Bash (Linux/macOS):
    ```bash
    export JULIA_NUM_THREADS=4
    ```
    C shell on Linux/macOS, CMD on Windows:
    ```bash
    set JULIA_NUM_THREADS=4
    ```
    Powershell on Windows:
    ```powershell
    $env:JULIA_NUM_THREADS=4
    ```
    Note that this must be done *before* starting Julia.

Also, the command `BLAS.set_num_threads(1)` is generally recommended to set the number of BLAS threads to 1, avoiding [oversubscription](https://ieeexplore.ieee.org/document/5470434)

## Running from command line as script

If you don't want to run MendelIHT.jl in a Julia session (e.g. you want to run batch jobs on a cluster), you can do so by putting the code below in a Julia file. For example, in order to run with 8 cores, create a file called `iht.jl` which contains:

```julia
# place these code in a file called iht.jl
using MendelIHT

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
