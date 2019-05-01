# MendelIHT

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://biona001.github.io/PACKAGE_NAME.jl/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://biona001.github.io/MendelIHT.jl/dev) | 
[![Build Status](https://travis-ci.org/biona001/MendelIHT.jl.svg?branch=master)](https://travis-ci.org/biona001/MendelIHT.jl) | 
[![Coverage Status](https://coveralls.io/repos/github/OpenMendel/SnpArrays.jl/badge.svg?branch=master)](https://coveralls.io/github/OpenMendel/SnpArrays.jl?branch=master) [![codecov](https://codecov.io/gh/OpenMendel/SnpArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/OpenMendel/SnpArrays.jl) |
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://biona001.github.io/PACKAGE_NAME.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://biona001.github.io/MendelIHT.jl/dev)

*A modern approach to analyze data from a Genome Wide Association Studies (GWAS)*

## Installation

IHT.jl supports Julia 1.0 and 1.1, but is currently an unregistered package. To install, press `]` to enter package manager mode, and then install the following packages:

```
(v1.0) pkg> add https://github.com/OpenMendel/SnpArrays.jl
(v1.0) pkg> add https://github.com/OpenMendel/MendelSearch.jl
(v1.0) pkg> add https://github.com/OpenMendel/MendelBase.jl
(v1.0) pkg> add https://github.com/biona001/IHT.jl
```

The order of installation is important!

## Documentation

+ [**STABLE**](https://biona001.github.io/MendelIHT.jl/stable)

Alterantively, a brief tutorial can be found in the [OpenMendel Tutorials](https://github.com/OpenMendel/Tutorials/blob/master/IHT/Mendel_IHT_tutorial.ipynb). For more advanced functionalities (e.g. doubly sparse projections, prior weightings), please see the [figures folder](https://github.com/biona001/MendelIHT.jl/tree/master/figures) which illustrates some of these functions. 

## Use Caution:

**Missing Genotype:**
The current implementation of MendelIHT assumes *there are no missing genotypes* since it uses linear algebra functions defined in [`SnpArrays.jl`](https://openmendel.github.io/SnpArrays.jl/latest/man/snparray/#linear-algebra-with-snparray). Therefore, you must first impute missing genotypes *before* you use MendelIHT. `SnpArrays.jl` offer some naive imputation strategy, but otherwise, we recommend using [Option 23 of Mendel](http://www.genetics.ucla.edu/software/mendel). 

**Parallel Computation:**
IHT enjoys built-in parallelism for cross validation routines. Users should ensure to (1) NOT spawn more workers than the number of available CPU cores, and (2) NOT remove auxiliary files (e.g. `train.bed`) that will be produced during cross validation. These files will be removed in the end. 

## Citation:

This is a work in progress, so at the moment please cite the general OpenMendel paper:

```
Zhou, Hua, et al. "OpenMendel: a cooperative programming project for statistical genetics." Human genetics (2019): 1-11.
```


See the [figures folder](https://github.com/biona001/MendelIHT.jl/tree/master/figures) for figures and code for reproducing them.