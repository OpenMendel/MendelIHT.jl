# IHT

A Julia module that implements the (normalized) [iterative hard thresholding algorithm](http://eprints.soton.ac.uk/142499/1/BD_NIHT09.pdf) (IHT) of Blumensath and Davies.
IHT performs [feature selection](https://en.wikipedia.org/wiki/Feature_selection) akin to [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics))- or [MCP](http://arxiv.org/pdf/1002.4734.pdf)-penalized regression using a greedy selection approach.

## Installation

IHT.jl is not registered in `METADATA`. 
It depends on other unregistered packages:

1. [PLINK.jl](https://github.com/klkeys/PLINK.jl)
2. [SnpArrays.jl](https://github.com/OpenMendel/SnpArrays.jl)
3. [Search.jl](https://github.com/OpenMendel/Search.jl)
4. [MendelBase.jl](https://github.com/OpenMendel/MendelBase.jl)

At the Julia REPL, execute

    Pkg.clone("https://github.com/klkeys/PLINK.jl.git")
    Pkg.clone("https://github.com/OpenMendel/SnpArrays.jl.git")
    Pkg.clone("https://github.com/OpenMendel/Search.jl.git")
    Pkg.clone("https://github.com/OpenMendel/MendelBase.jl.git")
    Pkg.clone("https://github.com/biona001/IHT.jl.git")

The order of installation is important!

## Tutorials

Detailed tutorials on the functionalities of IHT can be found inside the [docs](https://github.com/biona001/IHT.jl/tree/master/docs) folder. Open Mendel users should read the MendelIHT tutorial for a more focused demonstration. 

## GWAS users

Run MendelIHT on GWAS data using the following command:

    using IHT
    MendelIHT("gwas 1 Control.txt")

IHT.jl uses `control file` as inputs to specify all input arguments, similar to most [Open Mendel](https://openmendel.github.io/) packages. However, advanced users are encouraged to work directly with function calls such as `cv_iht` and `L0_reg` for greater flexibility. More comprehensive tutorials can be found in the docs folder. 

**MISSING GENOTYPE:** The current implementation of MendelIHT assumes *there are no missing genotypes* since it uses linear algebra functions defined in [`SnpArrays.jl`](https://openmendel.github.io/SnpArrays.jl/latest/man/snparray/#linear-algebra-with-snparray). Therefore, you must first impute missing genotypes *before* you use MendelIHT. `SnpArrays.jl` offer some naive imputation strategy, but otherwise, we recommend using [Option 23 of Mendel](http://www.genetics.ucla.edu/software/mendel). 

## IHT on Numerical Data

Given a data matrix `x`, a continuous response `y`, and a number `k` of desired predictors, we run IHT with the simple command

    output = L0_reg(x, y, k)

Here `output` is an `IHTResults` container object with the following fields:

* `loss` is the optimal loss function value (minimum residual sum of squares)
* `iter` is the number of iterations until convergence
* `time` is the time spent in computations
* `beta` is the vector of the optimal statistical model.

IHT.jl also facilitates crossvalidation for the best model size.
We perform _q_-fold crossvalidation via

    cv_output = cv_iht(x, y)

where `cv_output` is an `IHTCrossvalidationResults` container object with the following fields:

* `mses` contains the mean squared errors for each model size.
* `b` contains the estimated coefficients at the optimal statistical model.
* `bidx` contains the indices of the predictors in the best crossvalidated model.
* `k` is the best crossvalidated model size.

Important optimal arguments to `cv_iht` include:

* `path`, an `Int` vector containing the path of model sizes to test. Defaults to `collect(1:p)`, with `p = size(x,2)`.
* `q`, the number of crossvalidation folds to use. The default depends on the Julia variable `Sys.CPU_CORES`, but it usually equals one of 3, 4, or 5.
* `folds`, a `DenseVector{Int}` object to assign data to each fold.
* `pids`, the `Int` vector of process IDs to which we distribute computations. The default is to include all available processes via `pids = procs()`.
* `refit`, a `Bool` to determine whether or not to refit the model. The default is `refit = true`.

To fix the fold for each element of `y`, the user must pass a prespecified `Int` vector to `folds`.
For example, to perform 3-fold crossvalidation on a vector `y` with 30 elements, `folds` should be a 30-element vector with ten `1`s, ten `2`s, and ten `3`s:

    n = 30
    q = 3
    k = div(n,q)
    y = randn(n)
    folds = ones(Int, n)
    folds[k+1:2k] = 2
    folds[2k+1:end] = 3

IHT.jl can perform simple random assignment of folds using a subroutine from RegressionTools.jl:

    folds = RegressionTools.cv_get_folds(y, q) # use the phenotype vector...
    folds = Regressiontools.cv_get_folds(n, q) # ...or use the length of that vector

## GPU acceleration

IHT.jl interfaces with the GPU accelerator from PLINK.jl.
The GPU accelerator farms the calculation of the gradient to a GPU,
which greatly improves computational performance.
`L0_reg` needs to know where the GPU kernels are stored.
PLINK.jl preloads the 64-bit kernels into a variable `PLINK.gpucode64`.

    output   = L0_reg(x::BEDFile, y, k, PLINK.gpucode64)

PLINK.jl ships with kernels for `Float32` (32-bit) and `Float64` (64-bit) arrays.
The corresponding kernel files are housed in `PLINK.gpucode64` and `PLINK.gpucode32`.
These are the only kinds of arrays supported by IHT.jl.
Use of `Float32` arithmetic yields faster execution times but may suffer from numerical underflow.
IHT.jl enforces the same precision for both `x` and `y`; mixing single- and double-precision arrays is not allowed.

IHT.jl performs crossvalidation with GPUs via

    cv_output = cv_iht(xfile, covfile, yfile, PLINK.gpucode64)

Crossvalidation with GPUs is a complicated topic.
To summarize, IHT farms an entire copy of the data to each host process.
For IHT.jl, a host process corresponds to a single crossvalidation fold, so `q`-fold crossvalidation will usually entail `q` processes.
OpenCL memory constraints dictate that each process should have its own copy of the data on the GPU device.
**NOTA BENE:** IHT.jl currently makes no effort to ensure that the GPU contains sufficient memory for _q_ copies of the data.
Users must consider device memory limits when calling `cv_iht` with GPUs.
Exceeding device memory can yield cryptic OpenCL errors regarding unallocatable buffers.
