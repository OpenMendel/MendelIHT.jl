# Mendel - Iterative Hard Thresholding

*A modern approach to analyze data from a Genome Wide Association Studies (GWAS)*


!!! note

    Please see the [OpenMendel paper](https://link.springer.com/article/10.1007/s00439-019-02001-z) for a review of GWAS statistics and why we chose [Julia](https://docs.julialang.org/en/v1/) as the programming language for this project. Pay attention to sections *Handling SNP data* and *Iterative hard thresholding* because they are especially relevant in this package.

## Package Feature

+ Analyze large GWAS datasets intuitively.
+ Built-in support for [PLINK binary files](https://www.cog-genomics.org/plink/1.9/input#bed) via [SnpArrays.jl](https://github.com/OpenMendel/SnpArrays.jl) and [VCF files](https://en.wikipedia.org/wiki/Variant_Call_Format) via [VCFTools.jl](https://github.com/OpenMendel/VCFTools.jl).
+ Out-of-the-box parallel computing routines for `q-fold` cross-validation.
+ Fits a variety of generalized linear models with any choice of link function.
+ Computation directly on raw genotype files.
+ Efficient handlings for non-genetic covariates.
+ Optional acceleration (debias) step to dramatically improve speed.
+ Ability to explicitly incorporate weights for predictors.
+ Ability to enforce within and between group sparsity. 
+ Naive genotype imputation. 
+ Estimates nuisance parameter for negative binomial regression using Newton or MM algorithm. 
+ Excellent flexibility to handle different data structures and complements well with other Julia packages.

Read [our paper](https://www.biorxiv.org/content/10.1101/697755v2) for more detail.

## Manual Outline

```@contents
Pages = [
    "man/getting_started.md",
    "man/examples.md",
    "man/contributing.md",
]
Depth = 2
```
