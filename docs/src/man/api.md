
# API

## Wrapper Functions

Most users will use the following wrapper functions. Users specify location of PLINK files and possibly the phenotype/covariate files. These functions will soon be updated to support VCF and BGEN formats.

```@docs
  iht
  cross_validate
```

## Core Functions

Users can also use the `fit_iht` and `cv_iht` functions directly. One must import genotypes via [SnpArrays.jl](https://github.com/OpenMendel/SnpArrays.jl) and phenotypes/covariates using Julia's standard routine. 

```@docs
  fit_iht
  cv_iht
```

## Specifying Groups and Weights

When you have group and weight information, you input them as optional arguments in `fit_iht` and `cv_iht`. The weight vector is a vector of `Float64`, while the group vector is a vector of `Int`. For instance,

```Julia
    g = #import group vector
    w = #import weight vector
    ng = length(unique(g)) # specify number of non-zero groups
    result = fit_iht(y, x, z; J=ng, k=10, d=Normal(), l=IdentityLink(), group=g, weight=w)
```

## Simulation Utilities

For complex simulations, please use [TraitSimulation.jl](https://github.com/OpenMendel/TraitSimulation.jl). 

MendelIHT provides very naive simulation utilities, which were written before [TraitSimulation.jl](https://github.com/OpenMendel/TraitSimulation.jl) was developed.

```@docs
  simulate_random_snparray
  simulate_correlated_snparray
```

!!! note
    Simulating a SnpArray with $n$ subjects and $p$ SNPs requires up to $2np$ bits of RAM. 

```@docs
  simulate_random_response
```

!!! note
    For negative binomial and gamma, the link function must be LogLink. 

```@docs
  make_bim_fam_files
```

## Other Useful Functions

MendelIHT additionally provides useful utilities that may be of interest to a few advanced users. 

```@docs
  iht_run_many_models
  pve
```
