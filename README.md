# MendelIHT

A Julia module that implements the (normalized) [iterative hard thresholding algorithm](http://eprints.soton.ac.uk/142499/1/BD_NIHT09.pdf) (IHT) of Blumensath and Davies for GWAS data. IHT performs [feature selection](https://en.wikipedia.org/wiki/Feature_selection) akin to [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)). Code for GLM models and structured regression is under development.

## Installation

IHT.jl supports Julia 1.0 and 1.1, but is currently an unregistered package. To install, press `]` to enter package manager mode, and then install the following packages:

```
(v1.0) pkg> add https://github.com/OpenMendel/SnpArrays.jl
(v1.0) pkg> add https://github.com/OpenMendel/MendelSearch.jl
(v1.0) pkg> add https://github.com/OpenMendel/MendelBase.jl
(v1.0) pkg> add https://github.com/biona001/IHT.jl
```

The order of installation is important!

## Tutorials

A brief tutorial can be found in the [OpenMendel Tutorials](https://github.com/OpenMendel/Tutorials/blob/master/IHT/Mendel_IHT_tutorial.ipynb). For more advanced functionalities (e.g. doubly sparse projections, prior weightings), please see the [figures folder](https://github.com/biona001/MendelIHT.jl/tree/master/figures) which illustrates some of these functions. 

## MISSING GENOTYPE:

The current implementation of MendelIHT assumes *there are no missing genotypes* since it uses linear algebra functions defined in [`SnpArrays.jl`](https://openmendel.github.io/SnpArrays.jl/latest/man/snparray/#linear-algebra-with-snparray). Therefore, you must first impute missing genotypes *before* you use MendelIHT. `SnpArrays.jl` offer some naive imputation strategy, but otherwise, we recommend using [Option 23 of Mendel](http://www.genetics.ucla.edu/software/mendel). 

## Citation:

This is a work in progress. See the [figures folder](https://github.com/biona001/MendelIHT.jl/tree/master/figures) for figures and code for reproducing them.