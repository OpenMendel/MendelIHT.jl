# MendelIHT

*A modern approach to analyze data from a Genome Wide Association Studies (GWAS)*

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://biona001.github.io/MendelIHT.jl/latest) | [![Build Status](https://travis-ci.org/biona001/MendelIHT.jl.svg?branch=master)](https://travis-ci.org/biona001/MendelIHT.jl) | [![codecov](https://codecov.io/gh/biona001/MendelIHT.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/biona001/MendelIHT.jl) [![Coverage Status](https://coveralls.io/repos/github/biona001/MendelIHT.jl/badge.svg?branch=master)](https://coveralls.io/github/biona001/MendelIHT.jl?branch=master) 

## Installation

Start Julia, press `]` to enter package manager mode, and type the following (after `pkg>`):
```
(v1.0) pkg> add https://github.com/OpenMendel/SnpArrays.jl
(v1.0) pkg> add https://github.com/biona001/MendelIHT.jl
```
The order of installation is important!

## Documentation

+ [**Latest**](https://biona001.github.io/MendelIHT.jl/latest/)

## Why Iterative Hard Thresholding for GWAS? 

Because it is (figures taken from our manuscript, which is to be published):

### Fast & Memory Efficient

Benchmark results on 1 million SNPs and various sample size:
![](https://github.com/biona001/MendelIHT.jl/blob/master/figures/benchmark/yes_debias.png)

### Accurate

![](https://github.com/biona001/MendelIHT.jl/blob/master/figures/accuracy.png)

### Reliable

![](https://github.com/biona001/MendelIHT.jl/blob/master/figures/iht_lasso_marginal.png)

* Marginal tests indicate a traditional GWAS analysis from a SNP-by-SNP association test.

## Citation:

If you use `MendelIHT.jl` in an academic manuscript, please cite:

```
Zhou, Hua, et al. "OpenMendel: a cooperative programming project for statistical genetics." Human genetics (2019): 1-11.
```

Bibtex:

```
@article{zhou2019openmendel,
  title={OpenMendel: a cooperative programming project for statistical genetics},
  author={Zhou, Hua and Sinsheimer, Janet S and Bates, Douglas M and Chu, Benjamin B and German, Christopher A and Ji, Sarah S and Keys, Kevin L and Kim, Juhyun and Ko, Seyoon and Mosher, Gordon D and others},
  journal={Human genetics},
  pages={1--11},
  year={2019},
  publisher={Springer}
}
```
