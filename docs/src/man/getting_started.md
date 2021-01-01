
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

## Wrapper Functions

Most users will use the following wrapper functions, which automatically handles everything. The user only has to specify where the PLINK files (and possibly the phenotype/covariate files) are located. 

```@docs
  iht
```

```@docs
  cross_validate
```

## Core Functions

For advanced users, one can also run IHT regression or cross-validation directly. For cross validation, we generally recommend using `cv_iht`. This function cycles through the testing sets sequentially and fits different sparsity models in parallel. For larger problems (e.g. UK Biobank sized), one can instead choose to run `cv_iht_distribute_fold`. This function fits different sparsity models sequentially but initializes all training/testing model in parallel, which consumes more memory (see below). The later strategy allows one to distribute different sparsity parameters to different computers, achieving greater parallel power. 

```@docs
  fit
```   

```@docs
  cv_iht
```   

```@docs
  cv_iht_distribute_fold
```   

!!! note 

    **Do not** delete intermediate files with random file names created by `cv_iht` and `cv_iht_distribute_fold` (windows users will be instructed to manually do so via print statements). These are memory-mapped files necessary for cross validation. For `cv_iht`, **you must have `x` GB of free space and RAM on your hard disk** where `x` is your `.bed` file size. For `cv_iht_distribute_fold`, you must have enough RAM and disk space to fit all `q` training datasets simultaneously, each of which typically requires `(q - 1)/q * x` GB. 

## Specifying Groups and Weights

When you have group and weight information, you input them as optional arguments in `L0_reg` and `cv_iht`. The weight vector is a vector of Float64, while the group vector is a vector of integers. For instance,

```Julia
    g = #import group vector
    w = #import weight vector
    J = length(unique(g)) # specify number of non-zero groups
    result = L0_reg(x, xbm, z, y, J, k, d(), l, group=g, weight=w)
```

## Simulation Utilities

MendelIHT provides some simulation utilities that help users explore the function and capabilities of iterative hard thresholding. 

```@docs
  simulate_random_snparray
```

```@docs
  simulate_correlated_snparray
```

!!! note
    Simulating a SnpArray with $n$ subjects and $p$ SNPs requires up to $4np$ bits of RAM. Make sure you have enough RAM before simulating very large SnpArrays.

```@docs
  simulate_random_response
```

!!! note
    For negative binomial and gamma, the link function must be LogLink. For Bernoulli, the probit link seems to work better than logitlink when used in `cv_iht` or `L0_reg`. 

```@docs
  make_bim_fam_files
```

## Other Useful Functions

MendelIHT additionally provides useful utilities that may be of interest to a few advanced users. 

```@docs
  iht_run_many_models
```

```@docs
  maf_weights
```
