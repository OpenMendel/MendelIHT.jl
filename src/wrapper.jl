# TODO: Handle missing phenotypes (-9 or NA)
# TODO: Autimatic write to output file
# TODO: VCF read

"""
    iht(plinkfile, k, kwargs...)

Runs IHT with sparsity level `k`. Example:

```julia
result = iht("plinkfile", 10)
```

# Phenotypes
Will use column(s) specified in `col` of the `.fam` file for phenotype values. 
For instance, `col = 6` (default) means we will use the `6`th column of the `.fam` file as 
phenotype. We recognize missing phenotypes as `NA` or `-9`. For quantitative traits
(univariate or multivariate), missing phenotypes are imputed with the mean. Binary
and count phenotypes cannot be imputed. 

# Other covariates
An intercept will be automatically included as the only non-genetic covariate. 
If you have more covariates, see the 3 argument `iht` function.

# Arguments
- `plinkfile`: A `String` for input PLINK file name (without `.bim/.bed/.fam` suffixes)
- `k`: An `Int` for sparsity parameter = number of none-zero coefficients

# Optional Arguments
- `col`: Column of `.fam` file that stores phenotype. Can be integer (for 
    univariate analysis) or vector of integers (multivariate analysis). 
    Default is 6.
- `d`: Distribution of phenotypes. Use `Normal()` for quantitative traits,
    `Bernoulli()` for binary traits, `Poisson()` or `NegativeBinomial()` for
    count traits, and `MvNormal` for multivariate traits. 
- All arguments available in [`fit_iht`](@ref)
"""
function iht(
    plinkfile::AbstractString,
    k::Int;
    summaryfile::AbstractString = "iht.summary.txt",
    betafile::AbstractString = "iht.beta.txt",
    col::Union{Int, AbstractVector{Int}}=6,
    d::Distribution = length(col) > 1 ? MvNormal(Float64[]) : Normal(),
    kwargs...
    )
    snpdata = SnpArrays.SnpData(plinkfile)
    y = parse_phenotypes(snpdata, col, d)
    xla = SnpLinAlg{Float64}(snpdata.snparray, model=ADDITIVE_MODEL, 
        center=true, scale=true, impute=true)

    # run IHT
    if is_multivariate(y)
        result = fit_iht(y, Transpose(xla), k=k, d=d; kwargs...)
    else
        result = fit_iht(y, xla, k=k, d=d; kwargs...)
    end

    # save results
    open(summaryfile, "w") do io
        show(io, result)
    end
    writedlm(betafile, result.beta)

    return result
end

"""
    parse_phenotypes(x::SnpData, col::Union{Int, AbstractVector{Int}}, ::Distribution)

Reads phenotypes from columns `col` of a `SnpData`. We recognize missing phenotypes
as `NA` or `-9`. For quantitative traits (univariate or multivariate), missing
phenotypes are imputed with the mean. Binary and count phenotypes cannot be imputed. 
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

function phenotype_is_missing(s::AbstractString)
    return s == "-9" || s == "NA"
end

"""
    iht(plinkfile, covariates, k, kwargs...)

Runs IHT with sparsity level `k`, with additional covariates stored separately. Example:

```julia
result = iht("plinkfile", "covariates.txt", 10)
```

# Phenotypes
Will use column(s) specified in `col` of the `.fam` file for phenotype values. 
For instance, `col = 6` (default) means we will use the `6`th column of the `.fam` file as 
phenotype. We recognize missing phenotypes as `NA` or `-9`. For quantitative traits
(univariate or multivariate), missing phenotypes are imputed with the mean. Binary
and count phenotypes cannot be imputed. 

# Covariate file
Covariates are read using `readdlm` function in Julia base. We require the
covariate file to be comma separated, and not include a header line. Each row
should be a sample listed in the same order as in the PLINK. The first column
should be all 1s to indicate an intercept. All other columns will be automatically
standardized to mean 0 variance 1. 

# Arguments
- `plinkfile`: A `String` for input PLINK file name (without `.bim/.bed/.fam` suffixes)
- `covariates`: A `String` for covariate file name
- `k`: An `Int` for sparsity parameter = number of none-zero coefficients

# Optional Arguments
All arguments available in [`fit_iht`](@ref)
"""
function iht(
    plinkfile::AbstractString,
    covariates::AbstractString,
    k::Int;
    col::Union{Int, AbstractVector{Int}}=6,
    d::Distribution = length(col) > 1 ? MvNormal(Float64[]) : Normal(),
    kwargs...
    )
    snpdata = SnpArrays.SnpData(plinkfile)
    y = parse_phenotypes(snpdata, col, d)
    xla = SnpLinAlg{Float64}(snpdata.snparray, model=ADDITIVE_MODEL, 
        center=true, scale=true, impute=true)
    
    # read and standardize covariates 
    z = readdlm(covariates, ',', Float64)
    standardize!(@view(z[:, 2:end]))
    is_multivariate(y) && (z = convert(Matrix{Float64}, Transpose(z)))

    if is_multivariate(y)
        return fit_iht(y, Transpose(xla), z, k=k; kwargs...)
    else
        return fit_iht(y, xla, z, k=k, d=d; kwargs...)
    end
end

"""
    iht(phenotypes, plinkfile, covariates, k, kwargs...)

Runs IHT with sparsity level `k`, where both phenotypes and additional covariates
are stored separately. Example:

```julia
result = iht("phenotypes.txt", "plinkfile", "covariates.txt", 10)
```

# Phenotypes
Phenotypes are read using `readdlm` function in Julia base. We require each 
subject's phenotype to occupy a different row. The file should not include a header
line. Each row should be listed in the same order as in the PLINK. For multivariate
traits, different phenotypes should be comma separated. All phenotypes will be read
into memory but only phenotypes specified in columns `col` will be analyzed (jointly). 

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
- `k`: An `Int` for sparsity parameter = number of none-zero coefficients

# Optional Arguments
All arguments available in [`fit_iht`](@ref)
"""
function iht(
    phenotypes::AbstractString,
    plinkfile::AbstractString,
    covariates::AbstractString,
    k::Int;
    col::Union{Int, AbstractVector{Int}}=6,
    kwargs...
    )
    offset = 5
    snpdata = SnpArrays.SnpData(plinkfile)
    xla = SnpLinAlg{Float64}(snpdata.snparray, model=ADDITIVE_MODEL, 
        center=true, scale=true, impute=true)

    # read phenotypes
    y = readdlm(phenotypes, ',', Float64)
    if is_multivariate(y)
        y = convert(Matrix{Float64}, Transpose(y[:, col .- offset]))
    else
        y = dropdims(y, dims=2)
    end

    # read and standardize covariates 
    z = readdlm(covariates, ',', Float64)
    standardize!(@view(z[:, 2:end]))
    is_multivariate(y) && (z = convert(Matrix{Float64}, Transpose(z)))

    if is_multivariate(y)
        return fit_iht(y, Transpose(xla), z, k=k; kwargs...)
    else
        return fit_iht(y, xla, z, k=k; kwargs...)
    end
end

"""
    cross_validate(plinkfile, path, kwargs...)

Runs cross-validation to determinal optimal sparsity level `k`. Sparsity levels
is specified in `path. Example:

```julia
mses = cross_validate("plinkfile", 1:20)
mses = cross_validate("plinkfile", [1, 2, 3, 4, 5]) # alternative syntax
```

# Phenotypes and other covariates
Will use 6th column of `.fam` file for phenotype values and will automatically
include an intercept as the only non-genetic covariate. 

# Arguments
- `plinkfile`: A `String` for input PLINK file name (without `.bim/.bed/.fam` suffixes)
- `path`: Different sparsity levels. Can be an integer range (default 1:20) or vector of integers. 

# Optional Arguments
All arguments available in [`cv_iht`](@ref)
"""
function cross_validate(
    plinkfile::AbstractString,
    path::AbstractVector{<:Integer};
    col::Union{Int, AbstractVector{Int}}=6,
    d::Distribution = length(col) > 1 ? MvNormal(Float64[]) : Normal(),
    kwargs...
    )
    snpdata = SnpArrays.SnpData(plinkfile)
    x = SnpLinAlg{Float64}(snpdata.snparray, model=ADDITIVE_MODEL, center=true, scale=true)
    y = parse_phenotypes(snpdata, col, d)
    return cv_iht(y, x, path=path; kwargs...)
end

"""
    cross_validate(plinkfile, covariates, path, kwargs...)

Runs cross-validation to determinal optimal sparsity level `k`. Sparsity levels
is specified in `path. Example:

```julia
mses = cross_validate("plinkfile", "covariates.txt", 1:20)
mses = cross_validate("plinkfile", "covariates.txt", [1, 10, 20]) # alternative syntax
```

# Phenotypes
Will use 6th column of `.fam` file for phenotype values. 

# Covariate file
Covariates are read using `readdlm` function in Julia base. We require the
covariate file to be comma separated, and not include a header line. Each row
should be a sample listed in the same order as in the PLINK. The first column
should be all 1s to indicate an intercept. All other columns will be automatically
standardized to mean 0 variance 1.

# Arguments
- `plinkfile`: A `String` for input PLINK file name (without `.bim/.bed/.fam` suffixes)
- `covariates`: A `String` for covariate file name
- `path`: Different sparsity levels. Can be an integer range (default 1:20) or vector of integers. 

# Optional Arguments
All arguments available in [`cv_iht`](@ref)
"""
function cross_validate(
    plinkfile::AbstractString,
    covariates::AbstractString,
    path::AbstractVector{<:Integer};
    col::Union{Int, AbstractVector{Int}}=6,
    d::Distribution = length(col) > 1 ? MvNormal(Float64[]) : Normal(),
    kwargs...
    )
    snpdata = SnpArrays.SnpData(plinkfile)
    x = SnpLinAlg{Float64}(snpdata.snparray, model=ADDITIVE_MODEL, center=true, scale=true)
    y = parse_phenotypes(snpdata, col, d)
    
    # read and standardize covariates 
    z = readdlm(covariates, ',', Float64)
    standardize!(@view(z[:, 2:end])) 

    @assert size(z, 1) == size(x, 1) "$(size(z, 1)) samples detected in " * 
        "covariate file but $(size(x, 1)) samples detected in PLINK file."

    return cv_iht(y, x, z, path=path; kwargs...)
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
All arguments available in [`cv_iht`](@ref)
"""
function cross_validate(
    phenotypes::AbstractString,
    plinkfile::AbstractString,
    covariates::AbstractString,
    path::AbstractVector{<:Integer};
    kwargs...
    )
    snpdata = SnpArrays.SnpData(plinkfile)
    x = SnpLinAlg{Float64}(snpdata.snparray, model=ADDITIVE_MODEL, center=true, scale=true)

    # read phenotypes
    y = vec(readdlm(phenotypes, ',', Float64)) 

    # read and standardize covariates 
    z = readdlm(covariates, ',', Float64)
    standardize!(@view(z[:, 2:end])) 

    @assert size(y, 1) == size(z, 1) == size(x, 1) "Dimension mismatch: " * 
        "detected $(size(y, 1)) phenotypes, $(size(z, 1)) sample covariates, " * 
        "and $(size(x, 1)) sample genotypes."

    return cv_iht(y, x, z, path=path; kwargs...)
end
