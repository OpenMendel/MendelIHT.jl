# TODO: Handle missing phenotypes (-9 or NA)
# TODO: IHT signature and timings
# TODO: Autimatic write to output file

"""
    iht(plinkfile, k, kwargs...)

Runs IHT with sparsity level `k`. Example:

```julia
result = iht("plinkfile", 10)
```

# Phenotypes and other covariates
Will use 6th column of `.fam` file for phenotype values and will automatically
include an intercept as the only non-genetic covariate. Current there should
be NO missing phenotypes. 

# Arguments
- `plinkfile`: A `String` for input PLINK file name (without `.bim/.bed/.fam` suffixes)
- `k`: An `Int` for sparsity parameter = number of none-zero coefficients

# Optional Arguments
- `col`: Column of `.fam` file that stores phenotype. Can be integer (for 
    univariate analysis) or vector of integers (multivariate analysis). 
    Default is 6. 
All arguments available in [`fit_iht`](@ref)
"""
function iht(
    plinkfile::AbstractString,
    k::Int;
    col::Union{Int, AbstractVector{Int}}=6,
    kwargs...
    )
    snpdata = SnpArrays.SnpData(plinkfile)
    y = parse.(Float64, snpdata.person_info.phenotype)
    xla = SnpLinAlg{Float64}(snpdata.snparray, model=ADDITIVE_MODEL, 
        center=true, scale=true, impute=true)
    return fit_iht(y, xla, k=k; kwargs...)
end

"""
    parse_phenotypes(x::SnpData, col::AbstractVector{Int})

Reads phenotypes from columns `col` of a `SnpData` for multivariate analysis. 
Missing phenotypes are imputed as the mean.
"""
function parse_phenotypes(x::SnpData, col::AbstractVector{Int})
    n = x.people
    r = length(col) # number of traits
    y = Matrix{Float64}(undef, r, n)

    # impute missing phenotypes "-9" by mean of observed phenotypes
    missing_idx = Int[]
    for c in col
        fill!(missing_idx, false)
        s = 0.0
        for i in 1:n
            if phenotype_is_missing(x.person_info[i, c])
                y[c, i] = 0.0
                push!(missing_idx, i)
            else
                y[c, i] = parse(Float64, x.person_info[i, c])
                s += y[c, i]
            end
        end
        avg = s / (n - length(missing_idx))
        for i in missing_idx
            y[c, i] = avg
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
Will use 6th column of `.fam` file for phenotype values. 

# Other covariates
Covariates are read using `readdlm` function in Julia base. We require the
covariate file to be comma separated, and not include a header line. Each row
should be listed in the same order as in the PLINK. The first column should be
all 1s to indicate an intercept. All other columns will be standardized to mean
0 variance 1. 

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
    kwargs...
    )
    snpdata = SnpArrays.SnpData(plinkfile)
    y = parse.(Float64, snpdata.person_info.phenotype)
    xla = SnpLinAlg{Float64}(snpdata.snparray, model=ADDITIVE_MODEL, 
        center=true, scale=true, impute=true)
    
    # read and standardize covariates 
    z = readdlm(covariates, ',', Float64)
    standardize!(@view(z[:, 2:end])) 

    @assert size(z, 1) == size(xla, 1) "$(size(z, 1)) samples detected in " * 
        "covariate file but $(size(xla, 1)) samples detected in PLINK file."

    return fit_iht(y, xla, z, k=k; kwargs...)
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
line. Each row should be listed in the same order as in the PLINK. 

# Other covariates
Covariates are read using `readdlm` function in Julia base. We require the
covariate file to be comma separated, and not include a header line. Each row
should be listed in the same order as in the PLINK. The first column should be
all 1s to indicate an intercept. All other columns will be standardized to mean
0 variance 1. 

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
    kwargs...
    )
    snpdata = SnpArrays.SnpData(plinkfile)
    xla = SnpLinAlg{Float64}(snpdata.snparray, model=ADDITIVE_MODEL, 
        center=true, scale=true, impute=true)

    # read phenotypes
    y = vec(readdlm(phenotypes, ',', Float64)) 

    # read and standardize covariates 
    z = readdlm(covariates, ',', Float64)
    standardize!(@view(z[:, 2:end])) 

    @assert size(y, 1) == size(z, 1) == size(xla, 1) "Dimension mismatch: " * 
        "detected $(size(y, 1)) phenotypes, $(size(z, 1)) sample covariates, " * 
        "and $(size(xla, 1)) sample genotypes."

    return fit_iht(y, xla, z, k=k; kwargs...)
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
    kwargs...
    )
    snpdata = SnpArrays.SnpData(plinkfile)
    x = SnpLinAlg{Float64}(snpdata.snparray, model=ADDITIVE_MODEL, center=true, scale=true)
    y = parse.(Float64, snpdata.person_info.phenotype)
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

# Other covariates
Covariates are read using `readdlm`@ref function in Julia base. We require the
covariate file to be comma separated, and not include a header line. Each row
should be listed in the same order as in the PLINK. The first column should be
all 1s to indicate an intercept. All other columns will be standardized to mean
0 variance 1. 

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
    kwargs...
    )
    snpdata = SnpArrays.SnpData(plinkfile)
    x = SnpLinAlg{Float64}(snpdata.snparray, model=ADDITIVE_MODEL, center=true, scale=true)
    y = parse.(Float64, snpdata.person_info.phenotype)
    
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

# Other covariates
Covariates are read using `readdlm` function in Julia base. We require the
covariate file to be comma separated, and not include a header line. Each row
should be listed in the same order as in the PLINK. The first column should be
all 1s to indicate an intercept. All other columns will be standardized to mean
0 variance 1. 

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
