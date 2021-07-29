
# FAQ

If you do not find your problem here, or the provided solution does not solve your problem, please file an issue on [GitHub](https://github.com/OpenMendel/MendelIHT.jl/issues). 

## First-time Performance

In a fresh Julia session, the first time any function gets called will take a *long* time because the code has to be compiled on the spot. For instance, compare



```julia
@time using MendelIHT
```

      4.493948 seconds (9.04 M allocations: 564.071 MiB, 8.22% gc time)



```julia
@time using MendelIHT
```

      0.020589 seconds (32.81 k allocations: 1.886 MiB, 99.54% compilation time)


The first call was 200 times slower than the second time! Fortunately, for large problems, compilation time becomes negligible. 

## How to run code in parallel?

If Julia is started with multiple threads (e.g. `julia --threads 4`), `MendelIHT.jl` will automatically run your code in parallel. 

+ [How to start Julia with multiple threads](https://docs.julialang.org/en/v1/manual/multi-threading/#Starting-Julia-with-multiple-threads).
+ Execute `Threads.nthreads()` within Julia to check if multiple thread is enabled

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

For binary PLINK files (.bed/.bim/.fam) standardization is automatic. When using wrapper functions [cross_validate()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.cross_validate) and [iht()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.iht), non-genetic covariate will also be automatically standardized. However using internal functions [fit_iht()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.iht) and [cv_iht()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.cv_iht) bypasses standardization and is generally recommended only if wrapper functions do not work for your purposes. 

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

**Genotypes:** All genotypes can be imputed with the mean. 

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
