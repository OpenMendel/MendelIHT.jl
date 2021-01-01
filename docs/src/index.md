# Mendel - Iterative Hard Thresholding

*A modern approach to analyze data from a Genome Wide Association Studies (GWAS)*

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

Read [our paper](https://doi.org/10.1093/gigascience/giaa044) for more detail.

## Supported GLM models and Link functions

MendelIHT borrows distribution and link functions implementationed in [GLM.jl](http://juliastats.github.io/GLM.jl/stable/) and [Distributions.jl](https://juliastats.github.io/Distributions.jl/stable/).

| Distribution | Canonical Link | Status |
|:---:|:---:|:---:|
| Normal | IdentityLink | $\checkmark$ |
| Bernoulli | LogitLink |$\checkmark$ |
| Poisson | LogLink |  $\checkmark$ |
| NegativeBinomial | LogLink |  $\checkmark$ |
| Gamma | InverseLink | experimental |
| InverseGaussian | InverseSquareLink | experimental |

Examples of these distributions in their default value is visualized in [this post](https://github.com/JuliaStats/GLM.jl/issues/289).

### Available link functions

    CauchitLink
    CloglogLink
    IdentityLink
    InverseLink
    InverseSquareLink
    LogitLink
    LogLink
    ProbitLink
    SqrtLink

## Manual Outline

```@contents
Pages = [
    "man/getting_started.md",
    "man/math.md",
    "man/examples.md",
    "man/contributing.md",
]
Depth = 2
```
