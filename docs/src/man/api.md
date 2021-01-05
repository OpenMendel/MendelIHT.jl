
# API

## Wrapper Functions

Most users will use the following wrapper functions, which automatically handles everything. The user only has to specify where the PLINK files (and possibly the phenotype/covariate files) are located. 

```@docs
  iht
  cross_validate
```

## Core Functions

For advanced users, one can also run IHT regression or cross-validation directly. For cross validation, we generally recommend using `cv_iht`. This function cycles through the testing sets sequentially and fits different sparsity models in parallel. For larger problems (e.g. UK Biobank sized), one can instead choose to run `cv_iht_distribute_fold`. This function fits different sparsity models sequentially but initializes all training/testing model in parallel, which consumes more memory (see below). The later strategy allows one to distribute different sparsity parameters to different computers, achieving greater parallel power. 

```@docs
  fit_iht
  cv_iht
  cv_iht_distribute_fold
```

!!! note 

    **Do not** delete intermediate files with random file names created by `cv_iht` and `cv_iht_distribute_fold` (windows users will be instructed to manually do so via print statements). These are memory-mapped files necessary for cross validation. For `cv_iht`, **you must have `x` GB of free space and RAM on your hard disk** where `x` is your `.bed` file size. For `cv_iht_distribute_fold`, you must have enough RAM and disk space to fit all `q` training datasets simultaneously, each of which typically requires `(q - 1)/q * x` GB. 

## Specifying Groups and Weights

When you have group and weight information, you input them as optional arguments in `fit_iht` and `cv_iht`. The weight vector is a vector of `Float64`, while the group vector is a vector of `Int`. For instance,

```Julia
    g = #import group vector
    w = #import weight vector
    ng = length(unique(g)) # specify number of non-zero groups
    result = fit_iht(y, x, z; J=ng, k=10, d=Normal(), l=IdentityLink(), group=g, weight=w)
```

## Simulation Utilities

MendelIHT provides some simulation utilities that help users explore the function and capabilities of iterative hard thresholding. 

```@docs
  simulate_random_snparray
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
