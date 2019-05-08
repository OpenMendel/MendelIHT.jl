# Mendel - Iterative Hard Thresholding

*A modern approach to analyze data from a Genome Wide Association Studies (GWAS)*


!!! note

    Please see the [OpenMendel paper](https://link.springer.com/article/10.1007/s00439-019-02001-z) for a review of GWAS statistics and why we chose [Julia](https://docs.julialang.org/en/v1/) as the programming language for this project. Pay attention to sections *Handling SNP data* and *Iterative hard thresholding* because they are especially relevant in this package.

## Package Feature

+ Analyze large GWAS datasets intuitively.
+ Built-in parallel computing routines for `q-fold` cross-validation.
+ Fits a variety of generalized linear models with any choice of link function.
+ Computation directly on raw genotype files.
+ Ability to include non-genetic covariates.
+ Optional acceleration (debias) step to dramatically improve speed.
+ Ability to explicitly incorporate weights for predictors.
+ Ability to enforce within and between group sparsity. 
+ Excellent flexibility to handle different data structures and complements well with other Julia packages.

## Manual Outline

```@contents
Pages = [
    "man/getting_started.md",
    "man/examples.md",
    "man/contributing.md",
]
Depth = 2
```
