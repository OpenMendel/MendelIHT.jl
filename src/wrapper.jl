"""
    iht(plinkfile::AbstractString, k::Int, kwargs...)

Runs IHT with sparsity level `k`. 

# Phenotypes and other covariates
Will use 6th column of `.fam` file for phenotype values and will automatically
include an intercept as the only non-genetic covariate. 

# Arguments
- `plinkfile`: Input PLINK files (without `.bim/.bed/.fam` suffixes)
- `k`: Sparsity parameter = number of none-zero coefficients
"""
function iht(
    plinkfile::AbstractString,
    k::Int;
    kwargs...
    )
    snpdata = SnpArrays.SnpData(plinkfile)
    y = parse.(Float64, snpdata.person_info.phenotype)
    xla = SnpLinAlg{Float64}(snpdata.snparray, model=ADDITIVE_MODEL, 
        center=true, scale=true, impute=true)
    return fit(y, xla, k=k, kwargs...)
end

"""
    iht(plinkfile::AbstractString, covariates::AbstractString, k::Int, kwargs...)

Runs IHT with sparsity level `k`, with additional covariates stored separately.

# Phenotypes
Will use 6th column of `.fam` file for phenotype values. 

# Other covariates
Covariates are read using `readdlm` function in Julia base. We require the
covariate file to be comma separated, and not include a header line. Each row
should be listed in the same order as in the PLINK. The first column should be
all 1s to indicate an intercept. All other columns will be standardized to mean
0 variance 1. 

# Arguments
- `plinkfile`: Input PLINK files (without `.bim/.bed/.fam` suffixes)
- `k`: Sparsity parameter = number of none-zero coefficients
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

    return fit(y, xla, z, k=k, kwargs...)
end

"""
    iht(phenotypes, plinkfile, covariates, k::Int, kwargs...)

Runs IHT with sparsity level `k`, where both phenotypes and additional covariates
are stored separately.

# Phenotypes
Phenotypes are read using `readdlm` function in Julia base. We require the
phenotype file to be comma separated (if running multivariate response), and not
include a header line. Each row should be listed in the same order as in the PLINK. 

# Other covariates
Covariates are read using `readdlm` function in Julia base. We require the
covariate file to be comma separated, and not include a header line. Each row
should be listed in the same order as in the PLINK. The first column should be
all 1s to indicate an intercept. All other columns will be standardized to mean
0 variance 1. 

# Arguments
- `plinkfile`: Input PLINK files (without `.bim/.bed/.fam` suffixes)
- `k`: Sparsity parameter = number of none-zero coefficients
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
    y = readdlm(phenotypes, ',', Float64)

    # read and standardize covariates 
    z = readdlm(covariates, ',', Float64)
    standardize!(@view(z[:, 2:end])) 

    @assert size(y, 1) == size(z, 1) == size(xla, 1) "Dimension mismatch: " * 
        "detected $(size(y, 1)) phenotypes, $(size(z, 1)) sample covariates, " * 
        "and $(size(xla, 1)) sample genotypes."

    return fit(y, xla, z, k=k, kwargs...)
end

"""
    cv_iht(plinkfile::AbstractString, path::AbstractVector{<:Integer}, kwargs...)

Runs cross-validation to determinal optimal sparsity level `k`. Sparsity levels
is specified in `path. 

# Phenotypes and other covariates
Will use 6th column of `.fam` file for phenotype values and will automatically
include an intercept as the only non-genetic covariate. 

# Arguments
- `plinkfile`: Input PLINK files (without `.bim/.bed/.fam` suffixes)
- `path`: Different sparsity levels. Can be a range (default 1:20) or vector of integers. 
"""
function cv_iht(
    plinkfile::AbstractString,
    path::AbstractVector{<:Integer};
    kwargs...
    )
    snpdata = SnpArrays.SnpData(plinkfile)
    x = snpdata.snparray
    y = parse.(Float64, snpdata.person_info.phenotype)
    return cv_iht(y, x, path=path, kwargs...)
end

