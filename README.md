# MendelIHT

**Iterative hard thresholding -** *a multiple regression approach to analyze data from a Genome Wide Association Studies (GWAS)*

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://OpenMendel.github.io/MendelIHT.jl/latest) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://OpenMendel.github.io/MendelIHT.jl/stable) | [![Build Status](https://travis-ci.org/OpenMendel/MendelIHT.jl.svg?branch=master)](https://travis-ci.org/OpenMendel/MendelIHT.jl) | [![Coverage Status](https://coveralls.io/repos/github/OpenMendel/MendelIHT.jl/badge.svg?branch=master)](https://coveralls.io/github/OpenMendel/MendelIHT.jl?branch=master)  

## Installation

Download and install [Julia](https://julialang.org/downloads/). Within Julia, copy and paste the following:
```
using Pkg
Pkg.add(PackageSpec(url="https://github.com/OpenMendel/SnpArrays.jl.git"))
Pkg.add(PackageSpec(url="https://github.com/OpenMendel/VCFTools.jl.git"))
Pkg.add(PackageSpec(url="https://github.com/OpenMendel/MendelIHT.jl.git"))
```
This package supports Julia `v1.5`+.

## Documentation

+ [**Latest**](https://OpenMendel.github.io/MendelIHT.jl/latest/)
+ [**Stable**](https://OpenMendel.github.io/MendelIHT.jl/stable/)

## Citation and Reproducibility:

See our [paper](https://academic.oup.com/gigascience/article/9/6/giaa044/5850823?searchresult=1) for algorithmic details. If you use `MendelIHT.jl`, please cite:

```
@article{mendeliht,
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

In the `figures` subfolder, one can find all the code to reproduce the figures and tables in our preprint. 

## Bug fixes and user support

If you encounter a bug or need user support, please open a new issue on Github. Please provide as much detail as possible for bug reports, ideally a sequence of reproducible code that lead to the error.

PRs and feature requests are welcomed!
