# MendelIHT

**Iterative hard thresholding -** *a multiple regression approach to analyze data from a Genome Wide Association Studies (GWAS)*

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://OpenMendel.github.io/MendelIHT.jl/latest) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://OpenMendel.github.io/MendelIHT.jl/stable) | [![build Actions Status](https://github.com/OpenMendel/MendelIHT.jl/workflows/CI/badge.svg)](https://github.com/OpenMendel/MendelIHT.jl/actions) [![CI (Julia nightly)](https://github.com/openmendel/mendeliht.jl/workflows/JuliaNightly/badge.svg)](https://github.com/OpenMendel/MendelIHT.jl/actions/workflows/JuliaNightly.yml)| [![codecov](https://codecov.io/gh/OpenMendel/MendelIHT.jl/branch/master/graph/badge.svg?token=YyPqiFpIM1)](https://codecov.io/gh/OpenMendel/MendelIHT.jl) |

## Installation

Download and install [Julia](https://julialang.org/downloads/). Within Julia, copy and paste the following:
```
using Pkg
pkg"add MendelIHT"
```
This package supports Julia `v1.6`+ for Mac, Linux, and window machines. 

## Documentation

+ [**Latest**](https://OpenMendel.github.io/MendelIHT.jl/latest/)
+ [**Stable**](https://OpenMendel.github.io/MendelIHT.jl/stable/)

## Quick Start

The following uses data under the `data` directory. PLINK files are stored in `normal.bed`, `normal.bim`, `normal.fam`. 

```julia
# load package
using MendelIHT
dir = normpath(MendelIHT.datadir()) * "/"

# select k SNPs in PLINK file, Gaussian phenotypes
result = iht(dir * "normal", 9, Normal) # run IHT with k = 9
result = iht(dir * "normal", 10, Normal, covariates=dir*"covariates.txt") # separately include covariates, k = 10
result = iht(dir * "normal", 10, Normal, covariates=dir*"covariates.txt", phenotypes=dir*"phenotypes.txt") # phenotypes are stored separately

# run cross validation to determine best k
mses = cross_validate(dir * "normal", Normal, path=1:20) # test k = 1, 2, ..., 20
mses = cross_validate(dir * "normal", Normal, path=[1, 5, 10, 15, 20]) # test k = 1, 5, 10, 15, 20
mses = cross_validate(dir * "normal", Normal, path=1:20, covariates=dir*"covariates.txt") # separately include covariates
mses = cross_validate(dir * "normal", Normal, path=1:20, covariates=dir*"covariates.txt", phenotypes=dir*"phenotypes.txt") # if phenotypes are in separate file

# Multivariate IHT for multiple quantitative phenotypes
result = iht(dir * "multivariate", 10, MvNormal, phenotypes=[6, 7]) # phenotypes stored in 6th and 7th column of .fam file
result = iht(dir * "multivariate", 10, MvNormal, phenotypes=dir*"multivariate.phen") # phenotypes stored separate file

# other distributions for single trait analysis (no test data available)
result = iht("datafile", 10, Bernoulli) # logistic regression with k = 10
result = iht("datafile", 10, Poisson) # Poisson regression with k = 10
result = iht("datafile", 10, NegativeBinomial, est_r=:Newton) # Negative Binomial regression + nuisnace parameter estimation
```

Please see our latest [documentation](https://OpenMendel.github.io/MendelIHT.jl/latest/) for more detail. 

## Citation and Reproducibility:

For univariate analysis, please cite our [paper](https://academic.oup.com/gigascience/article/9/6/giaa044/5850823?searchresult=1) as

```
@article{chu2020iterative,
  title={{Iterative hard thresholding in genome-wide association studies: Generalized linear models, prior weights, and double sparsity}},
  author={Chu, Benjamin B and Keys, Kevin L and German, Christopher A and Zhou, Hua and Zhou, Jin J and Sobel, Eric M and Sinsheimer, Janet S and Lange, Kenneth},
  journal={GigaScience},
  volume={9},
  number={6},
  pages={giaa044},
  year={2020},
  publisher={Oxford University Press}
}
```
In the `figures` subfolder, one can find all the code to reproduce the figures and tables in our paper. 

For multivariate analysis, please cite our [paper](https://www.biorxiv.org/content/10.1101/2021.08.04.455145v2) as
```
@article{chu2021multivariate,
  title={Multivariate Genomewide Association Analysis with IHT},
  author={Chu, Benjamin B and Ko, Seyoon and Zhou, Jin J and Zhou, Hua and Sinsheimer, Janet S and Lange, Kenneth L},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
In the `manuscript` subfolder, one can find all the code to reproduce the figures and tables in our paper. 

## Bug fixes and user support

If you encounter a bug or need user support, please open a new issue on Github. Please provide as much detail as possible for bug reports, ideally a sequence of reproducible code that lead to the error.

PRs and feature requests are welcomed!
