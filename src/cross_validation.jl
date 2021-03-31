"""
    cv_iht(y, x, z; kwargs...)

For each model specified in `path`, performs `q`-fold cross validation and 
returns the (averaged) deviance residuals. 

The purpose of this function is to find the best sparsity level `k`, obtained
from selecting the model with the minimum out-of-sample error. Different cross
validation folds are cycled through sequentially different `paths` are fitted
in parallel on different CPUs. Currently there are no routines to cross validate
different group sizes. 

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
- `path`: Different model sizes to be tested in cross validation (default 1:20)
- `q`: Number of cross validation folds. (default 5)
- `d`: Distribution of your phenotype. (default Normal)
- `l`: A link function (default IdentityLink)
- `est_r`: Symbol for whether to estimate nuisance parameters. Only supported distribution is negative binomial and choices include :Newton or :MM.
- `group`: vector storing group membership for each predictor
- `weight`: vector storing vector of weights containing prior knowledge on each predictor
- `folds`: Vector that separates the sample into q disjoint subsets
- `destin`: Directory where intermediate files will be generated. Directory name must end with `/`.
- `use_maf`: Boolean indicating we should scale the projection step by a weight vector 
- `debias`: Boolean indicating whether we should debias at each IHT step
- `verbose`: Whether we want IHT to print meaningful intermediate steps
- `parallel`: Whether we want to run cv_iht using multiple CPUs (highly recommended)
"""
function cv_iht(
    y        :: AbstractVecOrMat{T},
    x        :: AbstractMatrix{T},
    z        :: AbstractVecOrMat{T};
    d        :: Distribution = is_multivariate(y) ? MvNormal(T[]) : Normal(),
    l        :: Link = IdentityLink(),
    path     :: AbstractVector{<:Integer} = 1:20,
    q        :: Int64 = 5,
    est_r    :: Symbol = :None,
    group    :: AbstractVector{Int} = Int[],
    weight   :: AbstractVector{T} = T[],
    folds    :: AbstractVector{Int} = rand(1:q, is_multivariate(y) ? size(x, 2) : size(x, 1)),
    destin   :: String = "./",
    use_maf  :: Bool = false,
    debias   :: Bool = false,
    verbose  :: Bool = true,
    parallel :: Bool = true,
    max_iter :: Int = 100
    ) where T <: Float

    typeof(x) <: AbstractSnpArray && error("x is a SnpArray! Please convert it to a SnpLinAlg first!")
    check_data_dim(y, x, z)

    # preallocate mean squared error matrix
    mses = zeros(length(path), q)

    # for displaying cross validation progress
    pmeter = Progress(q * length(path), "Cross validating...")
    channel = RemoteChannel(()->Channel{Bool}(q * length(path)), 1)    
    @async while take!(channel)
        next!(pmeter)
    end

    for fold in 1:q
        # find entries that are for test sets and train sets
        test_idx  = folds .== fold
        train_idx = .!test_idx

        # test different k
        mses[:, fold] = (parallel ? pmap : map)(path) do k
            # run IHT on training data with current k
            v = initialize(x, z, y, 1, k, d, l, group, weight, est_r, cv_train_idx=train_idx)
            result = fit_iht!(v, debias=debias, verbose=false, max_iter=max_iter)

            # predict on validation data
            v.cv_wts[train_idx] .= zero(T)
            v.cv_wts[test_idx] .= one(T)

            # update progres
            put!(channel, true)

            return predict!(v, result)
        end
    end
    put!(channel, false)

    #weight mses for each fold by their size before averaging
    mse = meanloss(mses, q, folds)

    # find best model size and print cross validation result
    k = path[argmin(mse)] :: Int
    verbose && print_cv_results(mse, path, k)

    return mse
end

function cv_iht(y::AbstractVecOrMat{T}, x::AbstractMatrix; kwargs...) where T
    z = is_multivariate(y) ? ones(T, 1, size(y, 2)) : ones(T, length(y))
    return cv_iht(y, x, z; kwargs...)
end

"""
Runs IHT across many different model sizes specifed in `path` using the full
design matrix. Same as `cv_iht` but **DOES NOT** validate in a holdout set, meaning
that this will definitely induce overfitting as we increase model size.
Use this if you want to quickly estimate a range of feasible model sizes before 
engaging in full cross validation. 
"""
function iht_run_many_models(
    y        :: AbstractVecOrMat{T},
    x        :: AbstractMatrix,
    z        :: AbstractVecOrMat{T};
    d        :: Distribution = Normal(),
    l        :: Link = canonicallink(d),
    path     :: AbstractVector{Int} = 1:20,
    est_r    :: Symbol = :None,
    group    :: AbstractVector{Int} = Int[],
    weight   :: AbstractVector{T} = Float64[],
    use_maf  :: Bool = false,
    debias   :: Bool = false,
    verbose  :: Bool = true,
    parallel :: Bool = false,
    max_iter :: Int = 100
    ) where T <: Float

    # for each k, fit model and store the loglikelihoods
    results = (parallel ? pmap : map)(path) do k
        if typeof(x) == SnpArray 
            xla = SnpLinAlg{T}(x, model=ADDITIVE_MODEL, center=true, scale=true, 
                impute=true)
            return fit_iht(y, xla, z, J=1, k=k, d=d, l=l, est_r=est_r, group=group, 
                weight=weight, use_maf=use_maf, debias=debias, verbose=false,
                max_iter=max_iter)
        else 
            return fit_iht(y, x, z, J=1, k=k, d=d, l=l, est_r=est_r, group=group, 
                weight=weight, use_maf=use_maf, debias=debias, verbose=false,
                max_iter=max_iter)
        end
    end

    loglikelihoods = zeros(size(path, 1))
    for i in 1:length(results)
        loglikelihoods[i] = results[i].logl
    end

    #display result and then return
    verbose && print_a_bunch_of_path_results(loglikelihoods, path)
    return loglikelihoods
end

function iht_run_many_models(y::AbstractVecOrMat{T}, x::AbstractMatrix; kwargs...) where T
    z = is_multivariate(y) ? ones(T, 1, size(y, 2)) : ones(T, length(y))
    return iht_run_many_models(y, x, z; kwargs...)
end

function predict!(v::IHTVariable{T, M}, result::IHTResult) where {T <: Float, M}
    # first update mean μ with estimated (trained) beta (cv weights are handled in deviance)
    mul!(v.xb, v.x, result.beta)
    mul!(v.zc, v.z, result.c)
    update_μ!(v)

    # Compute deviance residual (MSE for Gaussian response)
    return deviance(v)
end

function predict!(v::mIHTVariable{T, M}, result::mIHTResult) where {T <: Float, M}
    # first update mean μ with estimated (trained) beta
    mul!(v.BX, result.beta, v.X)
    mul!(v.CZ, result.c, v.Z)
    update_μ!(v)

    # Compute MSE
    mse = zero(T)
    @inbounds for j in 1:size(v.Y, 2), i in 1:ntraits(v)
        mse += abs2(v.Y[i, j] - v.μ[i, j]) * v.cv_wts[j]
    end
    return mse
end

"""
Scale mean squared errors (deviance residuals) for each fold by its own fold size.
"""
function meanloss(fitloss::AbstractMatrix, q::Int64, folds::AbstractVector{Int})
    ninfold = zeros(Int, q)
    for fold in folds
        ninfold[fold] += 1
    end

    loss = zeros(eltype(fitloss), size(fitloss, 1))
    for j = 1:size(fitloss, 2)
        wfold = convert(eltype(fitloss), ninfold[j]/length(folds))
        for i = 1:size(fitloss, 1)
            loss[i] += fitloss[i, j]*wfold
        end
    end

    return loss
end

meanloss(mses::Vector{Vector{T}}, num_fold, folds) where {T <: Float} = 
    meanloss(hcat(mses...), num_fold, folds)
