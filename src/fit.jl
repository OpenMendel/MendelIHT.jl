"""
    fit_iht(y, x, z; k=10, d = Normal(), l=IdentityLink(),...)

Fits a model on design matrix (genotype data) `x`, response (phenotype) `y`, 
and non-genetic covariates `z` on a specific sparsity parameter `k`. Only predictors in 
`x` will be subject to sparsity constraint, unless `zkeep` keyword is specified. 

# Arguments:
+ `y`: Phenotype vector or matrix. Should be an `Array{T, 1}` (single traits) or
    `Array{T, 2}` (multivariate Gaussian traits). For multivariate traits, each 
    column of `y` should be a sample. 
+ `x`: Genotype matrix (an `Array{T, 2}` or `SnpLinAlg`). For univariate
    analysis, samples are rows of `x`. For multivariate analysis, samples are
    columns of `x` (i.e. input `Transpose(x)` for `SnpLinAlg`)
+ `z`: Matrix of non-genetic covariates of type `Array{T, 2}` or `Array{T, 1}`.
    For univariate analysis, sample covariates are rows of `z`. For multivariate
    analysis, sample covariates are columns of `z`. If this is not specified, an
    intercept term will be included automatically. If `z` is specified, make sure
    the first column (row) is all 1s to represent the intercept. 

# Optional Arguments:
+ `k`: Number of non-zero predictors. Can be a constant or a vector (for group IHT). 
+ `J`: The number of maximum groups (set as 1 if no group infomation available)
+ `d`: Distribution of phenotypes. Specify `Normal()` for quantitative traits,
    `Bernoulli()` for binary traits, `Poisson()` or `NegativeBinomial()` for
    count traits, and `MvNormal()` for multiple quantitative traits. 
+ `l`: A link function. The recommended link functions are `l=IdentityLink()` for
    quantitative traits, `l=LogitLink()` for binary traits, `l=LogLink()` for Poisson
    distribution, and `l=Loglink()` for NegativeBinomial distribution. For multivariate
    analysis, the choice of link does not matter. 
+ `group`: vector storing (non-overlapping) group membership
+ `weight`: vector storing vector of weights containing prior knowledge on each SNP
+ `zkeep`: BitVector determining whether non-genetic covariates in `z` will be subject 
    to sparsity constraint. `zkeep[i] = true` means covariate `i` will NOT be projected.
    Note covariates forced in the model are not subject to sparsity constraint `k`. 
+ `est_r`: Symbol (`:MM`, `:Newton` or `:None`) to estimate nuisance parameters for negative binomial regression
+ `use_maf`: boolean indicating whether we want to scale projection with minor allele frequencies (see paper)
+ `debias`: boolean indicating whether we debias at each iteration
+ `verbose`: boolean indicating whether we want to print intermediate results
+ `tol`: used to track convergence
+ `max_iter`: is the maximum IHT iteration for a model to converge. Defaults to 200, or 100 for cross validation
+ `min_iter`: is the minimum IHT iteration before checking for convergence. Defaults to 5.
+ `max_step`: is the maximum number of backtracking per IHT iteration. Defaults 3
+ `io`: An `IO` object for displaying intermediate results. Default `stdout`.
+ `init_beta`: Whether to initialize beta values to univariate regression values. 
    Currently only Gaussian traits can be initialized. Default `false`. 
- `memory_efficient`: If `true,` it will cause ~1.1 times slow down but one only
    needs to store the genotype matrix (requiring 2np bits for PLINK binary files
    and `8np` bytes for other formats). If `memory_efficient=false`, one also need
    to store a `8nk` byte matrix where `k` is the sparsity levels. 

# Output 
+ An `IHTResult` (for single-trait analysis) or `mIHTResult` (for multivariate analysis).

# Group IHT
If `k` is a constant, then each group will have the same sparsity level. To run doubly 
sparse IHT with varying group sparsities, construct `k` to be a vector where `k[i]`
indicates the max number of predictors for group `i`. 
"""
function fit_iht(
    y         :: AbstractVecOrMat{T},
    x         :: AbstractMatrix{T},
    z         :: AbstractVecOrMat{T};
    k         :: Union{Int, Vector{Int}} = 10,
    J         :: Int = 1,
    d         :: Distribution = size(y, 2) > 1 ? MvNormal(T[]) : Normal(),
    l         :: Link = IdentityLink(),
    group     :: AbstractVector{Int} = Int[],
    weight    :: AbstractVector{T} = T[],
    zkeep     :: BitVector = trues(size(y, 2) > 1 ? size(z, 1) : size(z, 2)),
    est_r     :: Symbol = :None,
    use_maf   :: Bool = false, 
    debias    :: Bool = false,
    verbose   :: Bool = true,          # print informative things to stdout
    tol       :: T = convert(T, 1e-4), # tolerance for tracking convergence
    max_iter  :: Int = 200,            # maximum IHT iterations
    min_iter  :: Int = 5,              # minimum IHT iterations
    max_step  :: Int = 3,              # maximum backtracking for each iteration
    io        :: IO = stdout,
    init_beta :: Bool = false,
    memory_efficient::Bool=true
    ) where T <: Float

    verbose && print_iht_signature(io)

    # first handle errors
    @assert J ≥ 0        "Value of J (max number of groups) must be nonnegative!\n"
    @assert max_iter ≥ 0 "Value of max_iter must be nonnegative!\n"
    @assert max_step ≥ 0 "Value of max_step must be nonnegative!\n"
    @assert tol > eps(T) "Value of global tol must exceed machine precision!\n"
    checky(y, d) # make sure response data y is in the form compatible with specified GLM
    check_group(k, group) # make sure sparsity parameter `k` is reasonable. 
    !(typeof(d) <: NegativeBinomial) && est_r != :None && 
        error("Only negative binomial regression currently supports nuisance parameter estimation")
    typeof(x) <: AbstractSnpArray && error("x is a SnpArray! Please convert it to a SnpLinAlg first!")
    check_data_dim(y, x, z)
    if typeof(x) <: SnpLinAlg 
        x.center || error("x is not centered! Please construct SnpLinAlg{Float64}(::SnpArray, center=true, scale=true)")
        x.scale || @warn("x is not scaled! We highly recommend `scale=true` in `SnpLinAlg` constructor")
        x.impute || @warn("x does not have impute flag! We highly recommend `impute=true` in `SnpLinAlg` constructor")
    end

    # initialize IHT variable
    v = initialize(x, z, y, J, k, d, l, group, weight, est_r, init_beta, zkeep, 
        verbose=verbose, memory_efficient=memory_efficient)

    # print information
    verbose && print_parameters(io, k, d, l, use_maf, group, debias, tol, max_iter, min_iter)

    # fit IHT model
    tot_time, best_logl, mm_iter = fit_iht!(v, debias=debias, verbose=verbose,
        tol=tol, max_iter=max_iter, min_iter=min_iter, max_step=max_step, io=io)

    # compute phenotype's proportion of variation explained
    σ2 = pve(v)

    return IHTResult(tot_time, best_logl, mm_iter, σ2, v)
end

function fit_iht(
    y::AbstractVecOrMat{T},
    x::AbstractMatrix{T};
    kwargs...
    ) where T 
    z = is_multivariate(y) ? ones(T, 1, size(y, 2)) : ones(T, length(y))
    return fit_iht(y, x, z; kwargs...)
end

"""
    fit_iht!(v; kwargs...)

Fits a IHT variable `v`. 

# Arguments:
+ `v`: A properly initialized `mIHTVariable` or `IHTVariable`. Users should run [`fit_iht`](@ref)

# Optional Arguments:
+ `debias`: boolean indicating whether we debias at each iteration
+ `verbose`: boolean indicating whether we want to print results if model does not converge.
+ `tol`: used to track convergence
+ `max_iter`: is the maximum IHT iteration for a model to converge. Defaults to 200, or 100 for cross validation
+ `max_step`: is the maximum number of backtracking. Since l0 norm is not convex, we have no ascent guarantee
+ `io`: An `IO` object for displaying intermediate results. Default `stdout`.
"""
function fit_iht!(
    v         :: Union{mIHTVariable{T, M}, IHTVariable{T, M}};
    debias    :: Bool = false,
    verbose   :: Bool = true,          # print informative things
    tol       :: T = convert(T, 1e-4), # tolerance for tracking convergence
    max_iter  :: Int = 200,            # maximum IHT iterations
    min_iter  :: Int = 5,              # minimum IHT iterations
    max_step  :: Int = 3,              # maximum backtracking for each iteration
    io        :: IO = stdout
    ) where {T <: Float, M}

    #start timer
    start_time = time()

    # initialize constants
    mm_iter     = 0                 # number of iterations 
    tot_time    = 0.0               # compute time *within* fit!
    next_logl   = typemin(T)        # loglikelihood
    best_logl   = typemin(T)        # best loglikelihood achieved
    η_step      = 0                 # counts number of backtracking steps for η

    # Begin 'iterative' hard thresholding algorithm
    for iter in 1:max_iter

        # notify and return current model if maximum iteration exceeded
        if iter ≥ max_iter
            best_logl = save_prev!(v, next_logl, best_logl)
            save_best_model!(v)
            mm_iter  = iter
            tot_time = time() - start_time
            verbose && printstyled(io, "Did not converge after $max_iter " * 
                "iterations! IHT run time was " * string(tot_time) *
                " seconds\n", color=:red)
            break
        end

        # save values from previous iterate and update loglikelihood
        best_logl = save_prev!(v, next_logl, best_logl)

        # take one IHT step in positive score direction
        (η, η_step, next_logl) = iht_one_step!(v, next_logl, max_step)

        # perform debiasing if support didn't change
        debias && iter ≥ 5 && v.idx == v.idx0 && debias!(v)

        # track convergence
        # Note: estimated beta in first few iterations can be very small, so scaled_norm is very small
        # Thus we force IHT to iterate at least 5 times
        scaled_norm = check_convergence(v)
        progr = "Iteration $iter: loglikelihood = $next_logl, backtracks = $η_step, tol = $scaled_norm"
        verbose && println(io, progr)
        verbose && io != stdout && println(progr)
        if iter ≥ min_iter && scaled_norm < tol
            best_logl = save_prev!(v, next_logl, best_logl)
            save_best_model!(v)
            tot_time = time() - start_time
            mm_iter  = iter
            break
        end
    end

    return tot_time, best_logl, mm_iter
end

"""
Performs 1 iteration of the IHT algorithm, backtracking a maximum of `nstep` times.
We allow loglikelihood to potentially decrease to avoid bad boundary cases.
"""
function iht_one_step!(
    v::Union{IHTVariable{T, M}, mIHTVariable{T, M}},
    old_logl::T,
    nstep::Int
    ) where {T <: Float, M <: AbstractMatrix}

    # first calculate step size 
    η = iht_stepsize!(v)

    # update b and c by taking gradient step v.b = P_k(β + ηv) where v is the score direction
    _iht_gradstep!(v, η)

    # update the linear predictors `xb`, `μ`, and residuals with the new proposed b
    update_xb!(v)
    update_μ!(v)

    # for multivariate IHT, also update precision matrix Γ = 1/n * (Y-BX)(Y-BX)' 
    if typeof(v) <: mIHTVariable
        solve_Σ!(v)
    end

    # update r (nuisance parameter for negative binomial)
    if typeof(v) <: IHTVariable && v.est_r != :None
        v.d = mle_for_r(v)
    end

    # calculate current loglikelihood with the new computed xb and zc
    new_logl = loglikelihood(v)

    η_step = 0
    while _iht_backtrack_(new_logl, old_logl, η_step, nstep)

        # stephalving
        η /= 2

        # compute new loglikelihood after linesearch
        new_logl = backtrack!(v, η)

        # increment the counter
        η_step += 1
    end

    # compute score with the new mean
    score!(v)

    # check for finiteness before moving to the next iteration
    isnan(new_logl) && throw(error("Loglikelihood function is NaN, aborting..."))
    isinf(new_logl) && throw(error("Loglikelihood function is Inf, aborting..."))

    return η::T, η_step::Int, new_logl::T
end
