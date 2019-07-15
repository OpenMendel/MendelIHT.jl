# MendelIHT

*A modern approach to analyze data from a Genome Wide Association Studies (GWAS)*

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://biona001.github.io/MendelIHT.jl/latest) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://biona001.github.io/MendelIHT.jl/stable) | [![Build Status](https://travis-ci.org/biona001/MendelIHT.jl.svg?branch=master)](https://travis-ci.org/biona001/MendelIHT.jl) [![Build status](https://ci.appveyor.com/api/projects/status/s7dxx48g1ol9hqi0?svg=true)](https://ci.appveyor.com/project/biona001/mendeliht-jl) | [![Coverage Status](https://coveralls.io/repos/github/biona001/MendelIHT.jl/badge.svg)](https://coveralls.io/github/biona001/MendelIHT.jl)  [![codecov](https://codecov.io/gh/biona001/MendelIHT.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/biona001/MendelIHT.jl)

## Installation

Start Julia, press `]` to enter package manager mode, and type the following (after `pkg>`):
```
(v1.0) pkg> add https://github.com/OpenMendel/SnpArrays.jl
(v1.0) pkg> add https://github.com/biona001/MendelIHT.jl
```
The order of installation is important!

## Documentation

+ [**Latest**](https://biona001.github.io/MendelIHT.jl/latest/)
+ [**Stable**](https://biona001.github.io/MendelIHT.jl/stable/)

## Why Iterative Hard Thresholding for GWAS? 

Because it is (figures taken from our manuscript, which is to be published):

### Accurate

![](https://github.com/biona001/MendelIHT.jl/blob/master/figures/accuracy.png)

### Reliable

![](https://github.com/biona001/MendelIHT.jl/blob/master/figures/iht_lasso_marginal.png)

* Marginal tests indicate a traditional GWAS analysis from a SNP-by-SNP association test.

## Citation:

A preprint of our paper is available on [bioRxiv](https://www.biorxiv.org/content/10.1101/697755v1). If you use `MendelIHT.jl` in an academic manuscript, please cite:

```
Benjamin B. Chu, Kevin L. Keys, Janet S. Sinsheimer, and Kenneth Lange. Multivariate GWAS: Generalized Linear Models, Prior Weights, and Double Sparsity. bioRxiv doi:10.1101/697755
```
