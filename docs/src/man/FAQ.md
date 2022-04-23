
# FAQ

If you do not find your problem here, or the provided solution does not solve your problem, please file an issue on [GitHub](https://github.com/OpenMendel/MendelIHT.jl/issues). 

## First-time Performance

In a fresh Julia session, the first time any function gets called will take a *long* time because the code has to be compiled on the spot. For instance, compare



```julia
@time using MendelIHT
```

    140.065958 seconds (179.19 M allocations: 8.491 GiB, 2.39% gc time, 84.11% compilation time)



```julia
@time using MendelIHT
```

      0.000156 seconds (161 allocations: 12.859 KiB)


Fortunately, for large problems, compilation time becomes negligible. 

## Memory requirement?

For binary PLINK files, `MendelIHT.jl` uses [SnpArrays.jl's SnpLinAlg](https://openmendel.github.io/SnpArrays.jl/latest/#Linear-Algebra) for linear algebra. This data structure uses memory mapping and computes directly on raw genotype file, described in our [multivariate paper](https://www.biorxiv.org/content/10.1101/2021.08.04.455145v2.abstract). As such, it requires roughly $2np$ bits of *virtual memory* and much less physical memory to store. For UK biobank with 500k samples and 500k SNPs, this is roughly 62GB of virtual memory and a couple of GB of RAM. Thus, for binary PLINK files, we can easily fit IHT on large dataset, including those that are too large to fit in RAM.

In addition to storing the above matrix in memory, IHT also enables `memory_efficient=false` by default. This means IHT will additionally store a $n \times k$ matrix in double precision, where $k$ is the sparsity level. This require $64nk$ bits of RAM. For cross validation routines that test multiple different $k_1, ..., k_q$ values, $t$ of them must co-exist in memory where $t$ is the number of threads. Thus for large samples such as the UK Biobank data, it is possible that holding these "sparse" matrices will require more memory than holding compressed PLINK files. In such cases, specifying `memory_efficient=true` will prevent allocating these intermediate sparse matrices, but computation is roughly 1.5-2x slower. 

For BGEN and VCF files, `MendelIHT.jl` imports genotypes into double precision matrices which require $64np$ bits of RAM. For 500k samples and 500k SNPs, this requires 2 TB of RAM. Thus, IHT does not work on extremely large VCF and BGEN files. If you have these large VCF/BGEN files, one can convert it to binary PLINK format before running IHT.

## How to run code in parallel?

If Julia is started with multiple threads (e.g. `julia --threads 4`), `MendelIHT.jl` will automatically run your code in parallel. 

+ [How to start Julia with multiple threads](https://docs.julialang.org/en/v1/manual/multi-threading/#Starting-Julia-with-multiple-threads).
+ Execute `Threads.nthreads()` within Julia to check if multiple thread is enabled
+ On HPC clusters, it is often helpful to `ssh` into the node running IHT and explicitly check CPU usage by the `top` or `htop` command. 

!!! note

    When Julia is started with multiple threads, make sure to set the number of BLAS threads to 1 (i.e. `using LinearAlgebra; BLAS.set_num_threads(1)`. This avoids oversubscription. 

## Phenotype quality control

Our software assumes phenotypes have been properly quality controlled. For instance

+ There no are missing values (except possibly for Gaussian traits where we impute them with the mean). 
+ when running sparse linear regression, phenotypes should be approximately Gaussian
+ when running Poisson regression, phenotypes should have approximately equal mean and variance
+ when running multivariate traits, phenotypes are normalized

...etc.

Execute your judicious judgement!

## When to standardize phenotypes?

Only multivariate Guassian traits should be standardize to mean 0 variance 1. This ensures that mean squared error in cross-validation among traits are comparable, so the tuning process is driven by all traits.

For single trait analysis, standardization is not necessary.

## When to standardize covariates?

**Always** standardize your covariates (genetic and non-genetic) to mean 0 variance 1. This ensures sparsity is enforced equally on all predictors. 

For binary PLINK files (.bed/.bim/.fam) standardization is automatic. When using wrapper functions [cross_validate()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.cross_validate) and [iht()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.iht), non-genetic covariate will also be automatically standardized. However using internal functions [fit_iht()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.fit_iht) and [cv_iht()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.cv_iht) bypasses standardization and is generally recommended only if wrapper functions do not work for your purposes. 

## How to enforce sparsity on non-genetic covariates?

The `zkeep` parameter will allow non-genetic covariates to be subject to selection. Say you have 5 covariates, and you want to always keep the first 3 in the model but possibly set the last 2 to zero. You can do


```julia
zkeep = trues(5)
zkeep[4:5] .= false
zkeep

# now input zkeep as keyword argument for the wrapper or core functions, e.g. 
# iht(plinkfile, k, d, zkeep=zkeep)
```




    5-element BitVector:
     1
     1
     1
     0
     0



Note `zkeep` is a `BitVector` and not a `Vector{Bool}`. 

## Missing data?

In general, any sample or covariate with large proportion of missing (e.g. >10%) should be excluded. But our software does have a few built-in mechanisms for handling them.

**Phenotypes:** Gaussian phenotypes can be internally imputed with the mean. Binary/count phenotypes cannot be imputed.

**Genotypes:** All genotypes can be imputed with the mean. This is the default when using wrapper functions [iht()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.iht) and [cross_validate()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.cross_validate)

**Nongenetic covariates**: These cannot be imputed. Please impute them before running IHT.

## Keyword arguments?

Julia supports 2 types of "optional" arguments. Optional arguments specified before semicolon `;` can be directly inputted. Optional arguments specified after semicolon `;` needs to be explicitly inputted as `varname = x`. For instance, 


```julia
function add(a::Int, b::Int=1; c::Int=2)
    return a + b + c 
end
@show add(0)             # 0 + b + c using default value for b, c
@show add(0, 3)          # 0 + b + c using b = 3 and default value for c
@show add(0, 5, c=10);   # 0 + b + c using b = 5 and c = 10
```

    add(0) = 3
    add(0, 3) = 5
    add(0, 5, c = 10) = 15


## Will IHT work on sequence/imputed data?

If someone can test this out and tell us, that would be extremely helpful.
