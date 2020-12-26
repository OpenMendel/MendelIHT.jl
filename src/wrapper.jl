"""
    iht(plinkfile::AbstractString, k::Int, kwargs...)

Wrapper function for running IHT of sparsity `k`. Will use 6th column of `.fam`
file for phenotype values and will automatically include an intercept. 

# Arguments
- `plinkfile`: Input PLINK files (without `.bim/.bed/.fam` suffixes)
- `k`: Sparsity parameter = number of none-zero coefficients

# TODO
- non-genetic covariates
"""
function iht(
    plinkfile::AbstractString,
    k::Int;
    kwargs...
    )
    snpdata = SnpArrays.SnpData(plinkfile)
    y = parse.(Float64, snpdata.person_info.phenotype)
    xla = SnpLinAlg{Float64}(snpdata.snparray, model=ADDITIVE_MODEL, 
        center=true, scale=true)
    return fit(y, xla, k=k, kwargs...)
end

"""
    cv_iht(plinkfile::AbstractString, path::AbstractVector{<:Integer}, kwargs...)

Wrapper function for cross-validating sparsity `k`. Sparsity levels should be 
specified in `path. Will use 6th column of `.fam` file for phenotype values and
will automatically include an intercept. 

# Arguments
- `plinkfile`: Input PLINK files (without `.bim/.bed/.fam` suffixes)
- `path`: Different sparsity levels. Can be a range (default 1:20) or vector of integers. 

# TODO
- non-genetic covariates
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
