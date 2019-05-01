# MendelIHT

*A modern approach to analyze data from a Genome Wide Association Studies (GWAS)*

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://biona001.github.io/MendelIHT.jl/latest) | [![Build Status](https://travis-ci.org/biona001/MendelIHT.jl.svg?branch=master)](https://travis-ci.org/biona001/MendelIHT.jl) | [![codecov](https://codecov.io/gh/biona001/MendelIHT.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/biona001/MendelIHT.jl) [![Coverage Status](https://coveralls.io/repos/github/biona001/MendelIHT.jl/badge.svg?branch=master)](https://coveralls.io/github/biona001/MendelIHT.jl?branch=master) 

## Installation

Copy and paste the following in Julia (the order of installation is important):

```
using Pkg
Pkg.add("https://github.com/OpenMendel/SnpArrays.jl")
Pkg.add("https://github.com/OpenMendel/MendelSearch.jl")
Pkg.add("https://github.com/OpenMendel/MendelBase.jl")
Pkg.add("https://github.com/biona001/MendelIHT.jl")
```

## Documentation

+ [**Latest**](https://biona001.github.io/MendelIHT.jl/latest/)

Alterantively, a brief tutorial can be found in the [OpenMendel Tutorials](https://github.com/OpenMendel/Tutorials/blob/master/IHT/Mendel_IHT_tutorial.ipynb). For more advanced functionalities (e.g. doubly sparse projections, prior weightings), please see the [figures folder](https://github.com/biona001/MendelIHT.jl/tree/master/figures) which illustrates some of these functions. 

## Use Caution:

**Missing Genotype:**
The current implementation of MendelIHT assumes *there are no missing genotypes* since it uses linear algebra functions defined in [`SnpArrays.jl`](https://openmendel.github.io/SnpArrays.jl/latest/man/snparray/#linear-algebra-with-snparray). Therefore, you must first impute missing genotypes *before* you use MendelIHT. `SnpArrays.jl` offer some naive imputation strategy, but otherwise, we recommend using [Option 23 of Mendel](http://www.genetics.ucla.edu/software/mendel). 

**Parallel Computation:**
IHT enjoys built-in parallelism for cross validation routines. Users should ensure to (1) NOT spawn more workers than the number of available CPU cores, and (2) NOT remove auxiliary files (e.g. `train.bed`) that will be produced during cross validation. These files will be removed in the end. 

## Citation:

If you use MendelIHT.jl in an academic manuscript, please cite:

Zhou, Hua, et al. "OpenMendel: a cooperative programming project for statistical genetics." Human genetics (2019): 1-11.

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
