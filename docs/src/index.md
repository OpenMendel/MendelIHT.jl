# Iterative Hard Thresholding

*A modern approach to analyze data from a Genome Wide Association Studies (GWAS)*


!!! note

    Please see the [OpenMendel paper](https://link.springer.com/article/10.1007/s00439-019-02001-z) for a review of GWAS statistics and why we chose [Julia](https://docs.julialang.org/en/v1/) as the programming language for this project. Pay attention to sections *Handling SNP data* and *Iterative hard thresholding* because they are especially relevant in this package.

## Package Feature

+ Analyze large GWAS datasets intuitively.
+ Automatically distributed (parallel) `q-fold` cross-validation routines.
+ Ability to fit a large range of generalized linear models with any choice of link function.
+ Computation directly on raw genotype files.
+ Ability to include non-genetic covariates.
+ Optional acceleration step to dramatically improve speed by setting `debias = true`.
+ Ability to incorporate prior weights for predictors.
+ Ability to enforce within and between group sparsity. 
+ Easily adjustable convergence criteria, max iteration count, and max backtracking.

## Manual Outline

```@contents
    pages = [
		"Home" => "index.md"
        "Getting Started" => "man/getting_started.md"
    ]
```
