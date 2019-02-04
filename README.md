# IHT

A Julia module that implements the (normalized) [iterative hard thresholding algorithm](http://eprints.soton.ac.uk/142499/1/BD_NIHT09.pdf) (IHT) of Blumensath and Davies.
IHT performs [feature selection](https://en.wikipedia.org/wiki/Feature_selection) akin to [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics))- or [MCP](http://arxiv.org/pdf/1002.4734.pdf)-penalized regression using a greedy selection approach.

## Installation

IHT.jl supports Julia 1.0. but is currently an unregistered package. To install, press `]` to enter package manager mode, and then install the following packages:

```
(v1.0) pkg> add https://github.com/OpenMendel/SnpArrays.jl/
(v1.0) pkg> add https://github.com/OpenMendel/MendelBase.jl
(v1.0) pkg> add https://github.com/biona001/IHT.jl
```

The order of installation is important!

## Tutorials

Detailed tutorials on the functionalities of IHT can be found inside the [docs](https://github.com/biona001/IHT.jl/tree/master/docs) folder. Open Mendel users should read the MendelIHT tutorial for a more focused demonstration. 

## MISSING GENOTYPE:

The current implementation of MendelIHT assumes *there are no missing genotypes* since it uses linear algebra functions defined in [`SnpArrays.jl`](https://openmendel.github.io/SnpArrays.jl/latest/man/snparray/#linear-algebra-with-snparray). Therefore, you must first impute missing genotypes *before* you use MendelIHT. `SnpArrays.jl` offer some naive imputation strategy, but otherwise, we recommend using [Option 23 of Mendel](http://www.genetics.ucla.edu/software/mendel). 
