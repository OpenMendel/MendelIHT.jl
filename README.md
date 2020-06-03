# MendelIHT

**Iterative hard thresholding -** *a multiple regression approach to analyze data from a Genome Wide Association Studies (GWAS)*

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://OpenMendel.github.io/MendelIHT.jl/latest) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://OpenMendel.github.io/MendelIHT.jl/stable) | [![Build Status](https://travis-ci.org/OpenMendel/MendelIHT.jl.svg?branch=master)](https://travis-ci.org/OpenMendel/MendelIHT.jl) | [![Coverage Status](https://coveralls.io/repos/github/OpenMendel/MendelIHT.jl/badge.svg?branch=master)](https://coveralls.io/github/OpenMendel/MendelIHT.jl?branch=master)  

## Installation

Start Julia, press `]` to enter package manager mode, and type the following (after `pkg>`):
```
(v1.0) pkg> add https://github.com/OpenMendel/SnpArrays.jl
(v1.0) pkg> add https://github.com/OpenMendel/MendelSearch.jl
(v1.0) pkg> add https://github.com/OpenMendel/MendelBase.jl
(v1.0) pkg> add https://github.com/OpenMendel/MendelIHT.jl
```
The order of installation is important!

## Documentation

+ [**Latest**](https://OpenMendel.github.io/MendelIHT.jl/latest/)
+ [**Stable**](https://OpenMendel.github.io/MendelIHT.jl/stable/)

## Video Introduction

[![Video Introduction to MendelIHT.jl](https://github.com/OpenMendel/MendelIHT.jl/blob/master/figures/video_intro.png)](https://www.youtube.com/watch?v=UPIKafShwFw)

## Citation and Reproducibility:

See our [paper](https://academic.oup.com/gigascience/article/9/6/giaa044/5850823?searchresult=1) for algorithmic details. If you use `MendelIHT.jl`, please cite:

```
Benjamin B Chu, Kevin L Keys, Christopher A German, Hua Zhou, Jin J Zhou, Eric M Sobel, Janet S Sinsheimer, Kenneth Lange, Iterative hard thresholding in genome-wide association studies: Generalized linear models, prior weights, and double sparsity, GigaScience, Volume 9, Issue 6, June 2020, giaa044, https://doi.org/10.1093/gigascience/giaa044
```

In the `figures` subfolder, one can find all the code to reproduce the figures and tables in our preprint. 

## Bug fixes and user support

If you encounter a bug or need user support, please open a new issue on Github. Please provide as much detail as possible for bug reports, ideally a sequence of reproducible code that lead to the error.

PRs and feature requests are welcomed!
