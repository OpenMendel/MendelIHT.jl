"""
    iht(plinkfile, k; kwargs...)

Wrapper function for running IHT of sparsity `k`. 

# Arguments
- `plinkfile`: Input PLINK files (without `.bim/.bed/.fam` suffixes)
- `k`: Number of non-zero `βⱼ`s (counting intercept and non-genetic covariates)

# TODO
- handle cross validation
- non-genetic covariates
- group and weight inputs
"""
function iht(
    plinkfile :: AbstractString,
    k         :: Int;
    J         :: Int = 1,
    d         :: UnivariateDistribution = Normal(),
    l         :: Link = IdentityLink(),
    use_maf   :: Bool = false, 
    debias    :: Bool = false,
    verbose   :: Bool = true,
    tol       :: Float64 = 0.0001,
    max_iter  :: Int = 100,            # maximum IHT iterations
    max_step  :: Int = 5,              # maximum backtracking for each iteration
    )
    # import genotype and phenotype data
    snpdata = SnpArrays.SnpData(plinkfile)
    x = snpdata.snparray
    y = parse.(Float64, snpdata.person_info.phenotype)

    # non-genetic covariates
    z = ones(size(x, 1))

    # run IHT
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true)
    return fit(x, xla, z, y, J, k, d, l, use_maf=use_maf, debias=debias, 
        verbose=verbose, tol=tol, max_iter=max_iter, max_step=max_step)
end
