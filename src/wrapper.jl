# TODO: Handle missing phenotypes (-9 or NA)
# TODO: Autimatic write to output file
# TODO: VCF read

"""
    iht(plinkfile, k, kwargs...)

Runs IHT with sparsity level `k`. Example:

```julia
result = iht("plinkfile", 10)
```

# Arguments
- `plinkfile`: A `String` for input PLINK file name (without `.bim/.bed/.fam` suffixes)
- `k`: An `Int` for sparsity parameter = number of none-zero coefficients
- `d`: Distribution of phenotypes. Specify `Normal` for quantitative traits,
    `Bernoulli` for binary traits, `Poisson` or `NegativeBinomial` for
    count traits, and `MvNormal` for multiple quantitative traits. 

# Optional Arguments
- `phenotypes`: Phenotype file name (`String`), an integer, or vector of integer. Integer(s)
    coresponds to the column(s) of `.fam` file that stores phenotypes (default 6). 
    We recognize missing phenotypes as `NA` or `-9`. For quantitative traits
    (univariate or multivariate), missing phenotypes are imputed with the mean. Binary
    and count phenotypes cannot be imputed. Phenotype files are read using `readdlm` function
    in Julia base. We require each subject's phenotype to occupy a different row. The file
    should not include a header line. Each row should be listed in the same order as in
    the PLINK. 
- `covariates`: Covariate file name. Default is nothing (i.e. ""), where an intercept
    term will be automatically included. If `covariates` file specified, it will be 
    read using `readdlm` function in Julia base. We require the covariate file to be
    comma separated, and not include a header line. Each row should be listed in the
    same order as in the PLINK. The first column should be all 1s to indicate an
    intercept. All other columns will be standardized to mean 0 variance 1. 
- `summaryfile`: Output file name for saving IHT's summary statistics. Default
    `iht.summary.txt`.
- `betafile`: Output file name for saving IHT's estimated genotype effect sizes. 
    Default `iht.beta.txt`. 
- All arguments available in [`fit_iht`](@ref)
"""
function iht(
    plinkfile::AbstractString,
    k::Int,
    d::UnionAll;
    phenotypes::Union{AbstractString, Int, AbstractVector{Int}} = 6,
    covariates::AbstractString = "",
    summaryfile::AbstractString = "iht.summary.txt",
    betafile::AbstractString = "iht.beta.txt",
    kwargs...
    )
    # read genotypes
    snpdata = SnpArrays.SnpData(plinkfile)
    xla = SnpLinAlg{Float64}(snpdata.snparray, model=ADDITIVE_MODEL, 
        center=true, scale=true, impute=true)

    # read phenotypes
    y = parse_phenotypes(snpdata, phenotypes, d())

    # read and standardize covariates 
    z = covariates == "" ? ones(size(xla, 1)) : 
        parse_covariates(covariates, standardize=true)
    is_multivariate(y) && (z = convert(Matrix{Float64}, Transpose(z)))

    # run IHT
    if is_multivariate(y)
        result = fit_iht(y, Transpose(xla), z, k=k; kwargs...)
    else
        l = d == NegativeBinomial ? LogLink() : canonicallink(d()) # link function
        result = fit_iht(y, xla, z, k=k, d=d(), l=l; kwargs...)
    end

    # save results
    open(summaryfile, "w") do io
        show(io, result)
    end
    is_multivariate(y) ? writedlm(betafile, result.beta') : writedlm(betafile, result.beta)

    @show result

    return result
end

# adhoc constructor for empty MvNormal distribution
MvNormal() = MvNormal(Float64[])

"""
    parse_phenotypes(x, col::Union{Int, AbstractVector{Int}}, ::Distribution)

Reads phenotypes to numeric array. If `x` is a `SnpData`, columns `col` of the `.fam`
file will be parsed as phenotypes. Otherwise, will read `x` as comma-separated text
file where each sample occupies a row. We recognize missing phenotypes as `NA` or
`-9`. For quantitative traits (univariate or multivariate), missing phenotypes are
imputed with the mean. Binary and count phenotypes cannot be imputed. 
"""
function parse_phenotypes end

function parse_phenotypes(x::SnpData, col::AbstractVector{Int}, ::MvNormal)
    n = x.people
    r = length(col) # number of traits
    y = Matrix{Float64}(undef, r, n)
    offset = 5

    # impute missing phenotypes "-9" by mean of observed phenotypes
    missing_idx = Int[]
    for c in col
        empty!(missing_idx)
        s = 0.0
        for i in 1:n
            if phenotype_is_missing(x.person_info[i, c])
                y[c - offset, i] = 0.0
                push!(missing_idx, i)
            else
                y[c - offset, i] = parse(Float64, x.person_info[i, c])
                s += y[c - offset, i]
            end
        end
        avg = s / (n - length(missing_idx))
        for i in missing_idx
            y[c - offset, i] = avg
        end
    end
    return y
end

function parse_phenotypes(x::SnpData, col::Int, ::Normal)
    n = x.people
    y = Vector{Float64}(undef, n)

    # impute missing phenotypes by mean of observed phenotypes
    missing_idx = Int[]
    s = 0.0
    for i in 1:n
        if phenotype_is_missing(x.person_info[i, col])
            y[i] = 0.0
            push!(missing_idx, i)
        else
            y[i] = parse(Float64, x.person_info[i, col])
            s += y[i]
        end
    end
    avg = s / (n - length(missing_idx))
    for i in missing_idx
        y[i] = avg
    end
    return y
end

function parse_phenotypes(x::SnpData, col::Int, ::UnivariateDistribution)
    n = x.people
    y = Vector{Float64}(undef, n)

    # missing phenotypes NOT allowed for binary/count phenotypes
    for i in 1:n
        if phenotype_is_missing(x.person_info[i, col])
            error("Missing phenotype detected for sample $i. Automatic phenotype " * 
                "imputation are only possible for quantitative traits. Please " * 
                "exclude them or impute phenotypes first. ")
        else
            y[i] = parse(Float64, x.person_info[i, col])
        end
    end
    return y
end

function parse_phenotypes(x::SnpData, pheno_filename::AbstractString, d)
    y = readdlm(pheno_filename, ',', Float64)
    if is_multivariate(y)
        y = convert(Matrix{Float64}, Transpose(y))
    else
        y = dropdims(y, dims=2)
    end
    return y
end

"""
    parse_covariates(x::AbstractString; standardize::Bool=true)

Reads a comma separated text file `x`. Each row should be a sample ordered the 
same as in the plink file. The first column should be array of 1 (representing
intercept). Each covariate should be comma separated. If `standardize=true`, 
columns 2 and beyond will be normalized to mean 0 variance 1. 
"""
function parse_covariates(x::AbstractString; standardize::Bool=true)
    z = readdlm(x, ',', Float64)
    standardize!(@view(z[:, 2:end]))
    return z
end

function phenotype_is_missing(s::AbstractString)
    return s == "-9" || s == "NA"
end

"""
    cross_validate(phenotypes, plinkfile, covariates, path, kwargs...)

Runs cross-validation to determinal optimal sparsity level `k`. Sparsity levels
is specified in `path. Example:

```julia
mses = cross_validate("phenotypes.txt", "plinkfile", "covariates.txt", 1:20)
mses = cross_validate("phenotypes.txt", "plinkfile", "covariates.txt", [1, 10, 20]) # alternative syntax
```

# Phenotypes
Phenotypes are read using `readdlm` function in Julia base. We require each 
subject's phenotype to occupy a different row. The file should not include a header
line. Each row should be listed in the same order as in the PLINK. 

# Covariate file
Covariates are read using `readdlm` function in Julia base. We require the
covariate file to be comma separated, and not include a header line. Each row
should be a sample listed in the same order as in the PLINK. The first column
should be all 1s to indicate an intercept. All other columns will be automatically
standardized to mean 0 variance 1.

# Arguments
- `phenotypes`: A `String` for phenotype file name
- `plinkfile`: A `String` for input PLINK file name (without `.bim/.bed/.fam` suffixes)
- `covariates`: A `String` for covariate file name
- `path`: Different sparsity levels. Can be an integer range (default 1:20) or vector of integers. 

# Optional Arguments
- `phenotypes`: Phenotype file name (`String`), an integer, or vector of integer. Integer(s)
    coresponds to the column(s) of `.fam` file that stores phenotypes (default 6). 
    We recognize missing phenotypes as `NA` or `-9`. For quantitative traits
    (univariate or multivariate), missing phenotypes are imputed with the mean. Binary
    and count phenotypes cannot be imputed. Phenotype files are read using `readdlm` function
    in Julia base. We require each subject's phenotype to occupy a different row. The file
    should not include a header line. Each row should be listed in the same order as in
    the PLINK. 
- `covariates`: Covariate file name. Default is nothing (i.e. ""), where an intercept
    term will be automatically included. If `covariates` file specified, it will be 
    read using `readdlm` function in Julia base. We require the covariate file to be
    comma separated, and not include a header line. Each row should be listed in the
    same order as in the PLINK. The first column should be all 1s to indicate an
    intercept. All other columns will be standardized to mean 0 variance 1. 
- `summaryfile`: Output file name for saving IHT's cross validation summary statistics.
    Default `cviht.summary.txt`.
- `path`: Different values of `k` that should be tested. Default `1:20`
- `q`: Number of cross validation folds. Larger means more accurate and more computationally
    intensive. Should be larger 2 and smaller than 10. Default 5. 
- All arguments available in [`cv_iht`](@ref)
"""
function cross_validate(
    plinkfile::AbstractString,
    d::UnionAll;
    phenotypes::Union{AbstractString, Int, AbstractVector{Int}} = 6,
    covariates::AbstractString = "",
    cv_summaryfile::AbstractString = "cviht.summary.txt",
    path::AbstractVector{<:Integer} = 1:20,
    q::Int = 5,
    kwargs...
    )
    start_time = time()
    snpdata = SnpArrays.SnpData(plinkfile)
    x = SnpLinAlg{Float64}(snpdata.snparray, model=ADDITIVE_MODEL, center=true,
        scale=true, impute=true)

    # read phenotypes
    y = parse_phenotypes(snpdata, phenotypes, d())

    # read and standardize covariates 
    z = covariates == "" ? ones(size(x, 1)) : 
        parse_covariates(covariates, standardize=true)
    is_multivariate(y) && (z = convert(Matrix{Float64}, Transpose(z)))

    # run cross validation
    if is_multivariate(y)
        mse = cv_iht(y, Transpose(x), z, path=path, q=q; kwargs...)
    else
        l = d == NegativeBinomial ? LogLink() : canonicallink(d()) # link function
        mse = cv_iht(y, x, z, path=path, q=q, d=d(), l=l; kwargs...)
    end

    # save results
    open(cv_summaryfile, "w") do io
        k = path[argmin(mse)]
        print_cv_results(io, mse, path, k)
        end_time = time() - start_time
        println(io, "Total cross validation time = $end_time seconds")
    end

    return mse
end
