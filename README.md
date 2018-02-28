# IHT

A Julia module that implements the (normalized) [iterative hard thresholding algorithm](http://eprints.soton.ac.uk/142499/1/BD_NIHT09.pdf) (IHT) of Blumensath and Davies.
IHT performs [feature selection](https://en.wikipedia.org/wiki/Feature_selection) akin to [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics))- or [MCP](http://arxiv.org/pdf/1002.4734.pdf)-penalized regression using a greedy selection approach.

## Basic use

Given a data matrix `x`, a continuous response `y`, and a number `k` of desired predictors, we run IHT with the simple command

    output = L0_reg(x, y, k)

Here `output` is an `IHTResults` container object with the following fields:

* `loss` is the optimal loss function value (minimum residual sum of squares)
* `iter` is the number of iterations until convergence
* `time` is the time spent in computations
* `beta` is the vector of the optimal statistical model.

IHT.jl also facilitates crossvalidation for the best model size. 
Given a vector `modelsizes` of model sizes to test,
we perform _q_-fold crossvalidation via

    cv_output = cv_iht(x, y)

where `cv_output` is an `IHTCrossvalidationResults` container object with the following fields:

* `mses` contains the mean squared errors for each model size
* `b` contains the estimated coefficients at the optimal statistical model
* `bidx` contains the indices of the predictors in the best crossvalidated model
* `k` is the best crossvalidated model size.

Important optimal arguments to `cv_iht` include 

* `path`, an `Int` vector containing the path of model sizes to test. Defaults to `collect(1:p)`, with `p = size(x,2)`. 
* `q`, the number of crossvalidation folds to use. The default is depends on the Julia variable `CPU_CORES`, but it usually defaults to 5. 
* `folds`, a `DenseVector{Int}` object to assign data to each fold
* `pids`, the `Int` vector of process IDs to which we distribute computations
* `refit`, a `Bool` to determine whether or not to refit the model. 

To fix the folds, pass a prespecified integer vector to `folds`.

## GWAS

IHT.jl interfaces with [PLINK.jl](https://github.com/klkeys/PLINK.jl) to enable feature selection over [GWAS](https://en.wikipedia.org/wiki/Genome-wide_association_study) data in [PLINK binary format](http://pngu.mgh.harvard.edu/~purcell/plink/data.shtml#bed).
The interface to `L0_reg` is unchanged:

    output = L0_reg(x::BEDFile, y, k)

However, the call to `cv_iht` changes in order to accomodate parallel computing.
For the most part, the data **must** be stored in a binary format (PLINK binary or standard binary). 
The exception is the matrix of nongenetic covariates, which is read from a delimited text file. 
Instead of using variables from the REPL, `cv_iht` uses file paths to the data stored in file:

    cv_output = cv_iht(xfile.bed, covfile.txt, yfile.bin)

Here `xfile` points to the genotype data, `covfile` points to the covariates, and `yfile` points to the response vector. Note the file extensions! They are important for correct initialization of the data. 
See the documentation of PLINK.jl for details about the `BEDFile` object.

## GPU acceleration

IHT.jl interfaces with the GPU accelerator from PLINK.jl.
The GPU accelerator farms the calculation of the gradient to a GPU,
which greatly improves computational performance.
`L0_reg` needs to know where the GPU kernels are stored.
PLINK.jl preloads the kernels into a variable `PLINK.gpucode64`.

    output   = L0_reg(x::BEDFile, y, k, PLINK.gpucode64)

PLINK.jl ships with kernels for `Float64` and `Float64` arrays.
The corresponding kernel files are housed in `PLINK.gpucode64` and `PLINK.gpucode32`.
These are the only kinds of arrays supported by IHT.jl.
Use of `Float32` arithmetic yields faster execution times but may suffer from numerical underflow.

Crossvalidation with GPUs is a complicated topic. 
IHT farms an entire copy of the data to each host process. 
OpenCL memory constraints dictate that each process should have its own copy of the data on the device. 
Then IHT.jl performs crossvalidation with GPUs via

    cv_output = cv_iht(xfile, covfile, yfile, PLINK.gpucode64)

**NOTA BENE:** IHT.jl currently makes no effort to ensure that the GPU contains sufficient memory for _q_ copies of the data.
Users are urged to consider device memory limits when calling `cv_iht`.
Exceeding device memory can yield cryptic OpenCL errors regarding unallocatable buffers.
