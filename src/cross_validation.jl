"""
    cv_iht(y, x, z; path=1:20, q=5, d=Normal(), l=IdentityLink(), ...)

For each model specified in `path`, performs `q`-fold cross validation and 
returns the (averaged) deviance residuals. The purpose of this function is to
find the best sparsity level `k`, obtained from selecting the model with the
minimum out-of-sample error. Note sparsity is enforced on `x` only, unless `zkeep`
keyword is specified. 

To check if multithreading is enabled, check output of `Threads.nthreads()`.

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
- `path`: Different values of `k` that should be tested. One can input a vector of 
    `Int` (e.g. `path=[5, 10, 15, 20]`) or a range (default `path=1:20`).
- `q`: Number of cross validation folds. Larger means more accurate and more computationally
    intensive. Should be larger 2 and smaller than 10. Default `q=5`.
- `d`: Distribution of phenotypes. Specify `Normal()` for quantitative traits,
    `Bernoulli()` for binary traits, `Poisson()` or `NegativeBinomial()` for
    count traits, and `MvNormal()` for multiple quantitative traits. 
- `l`: A link function. The recommended link functions are `l=IdentityLink()` for
    quantitative traits, `l=LogitLink()` for binary traits, `l=LogLink()` for Poisson
    distribution, and `l=Loglink()` for NegativeBinomial distribution. For multivariate
    analysis, the choice of link does not matter. 
- `zkeep`: BitVector determining whether non-genetic covariates in `z` will be subject 
    to sparsity constraint. `zkeep[i] = true` means covariate `i` will NOT be projected.
    Note covariates forced in the model are not subject to sparsity constraints in `path`. 
- `est_r`: Symbol (`:MM`, `:Newton` or `:None`) to estimate nuisance parameters for negative binomial regression
- `group`: vector storing group membership for each predictor
- `weight`: vector storing vector of weights containing prior knowledge on each predictor
- `folds`: Vector that separates the sample into `q` disjoint subsets
- `debias`: Boolean indicating whether we should debias at each IHT step. Defaults `false`
- `verbose`: Boolean indicating whether to print progress and mean squared error for
    each `k` in `path`. Defaults `true`
- `max_iter`: is the maximum IHT iteration for a model to converge. Defaults to 100 
- `min_iter`: is the minimum IHT iteration before checking for convergence. Defaults to 5.
- `init_beta`: Whether to initialize beta values to univariate regression values. 
    Currently only Gaussian traits can be initialized. Default `false`. 
- `memory_efficient`: If `true,` it will cause ~1.1 times slow down but one only
    needs to store the genotype matrix (requiring 2np bits for PLINK binary files
    and `8np` bytes for other formats). If `memory_efficient=false`, one may potentially
    store `t` sparse matrices of dimension `8nk` bytes where `k` are the cross validated
    sparsity levels. 

# Output
- `mse`: A vector of mean-squared error for each `k` specified in `path`. 
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
    zkeep    :: BitVector = trues(size(y, 2) > 1 ? size(z, 1) : size(z, 2)),
    folds    :: AbstractVector{Int} = rand(1:q, is_multivariate(y) ? size(x, 2) : size(x, 1)),
    debias   :: Bool = false,
    verbose  :: Bool = true,
    max_iter :: Int = 100,
    min_iter :: Int = 5,
    init_beta :: Bool = false,
    memory_efficient :: Bool = true
    ) where T <: Float

    typeof(x) <: AbstractSnpArray && throw(ArgumentError("x is a SnpArray! Please convert it to a SnpLinAlg first!"))
    check_data_dim(y, x, z)
    verbose && print_iht_signature()

    # preallocated arrays for efficiency
    test_idx  = [falses(length(folds)) for i in 1:Threads.nthreads()]
    train_idx = [falses(length(folds)) for i in 1:Threads.nthreads()]
    V = [initialize(x, z, y, 1, 1, d, l, group, weight, est_r, false, zkeep,
        memory_efficient=memory_efficient) for i in 1:Threads.nthreads()]

    # for displaying cross validation progress
    pmeter = verbose ? Progress(q * length(path), "Cross validating...") : nothing

    # cross validate. TODO: wrap pmap with batch_size keyword to enable distributed CV
    combinations = allocate_fold_and_k(q, path)
    mses = zeros(length(combinations))
    ThreadPools.@qthreads for i in 1:length(combinations)
        fold, sparsity = combinations[i]

        # assign train/test indices
        id = Threads.threadid()
        v = V[id]
        test_idx[id]  .= folds .== fold
        train_idx[id] .= folds .!= fold

        # run IHT on training data with current (fold, sparsity)
        v.k = sparsity
        init_iht_indices!(v, init_beta, train_idx[id], false)
        fit_iht!(v, debias=debias, verbose=false, max_iter=max_iter, min_iter=min_iter)

        # predict on validation data
        v.cv_wts[train_idx[id]] .= zero(T)
        v.cv_wts[test_idx[id]] .= one(T)
        mses[i] = predict!(v)

        # update progres
        verbose && next!(pmeter)
    end

    # weight mses for each fold by their size before averaging
    mse = meanloss(mses, q, folds)

    # find best model size and print cross validation result
    k = path[argmin(mse)] :: Int
    verbose && print_cv_results(mse, path, k)

    return mse
end

"""
    cmsa_iht(y, x, z; path=1:20, q=5, d=Normal(), l=IdentityLink(), ...)

# Cross-model selection and averaging
First, this procedure separates the training set in K folds (e.g. 10 folds). 
Secondly, in turn, each fold is considered as an inner validation set and the other 
(K-1) folds form an inner training set (blue). A "regularization
path" of models is trained on the inner training set and the corresponding 
predictions (scores) for the inner validation set are computed. The model that 
minimizes the loss on the inner validation set is selected. Finally, the K 
resulting models are averaged; this is different to standard cross-validation 
where the model is refitted on the whole training set using the best-performing 
hyper-parameters.
"""
function cmsa_iht(
    y        :: AbstractVecOrMat{T},
    x        :: AbstractMatrix{T},
    z        :: AbstractVecOrMat{T};
    d        :: Distribution = is_multivariate(y) ? MvNormal(T[]) : Normal(),
    l        :: Link = IdentityLink(),
    dfmax    :: Int = 50000, # The maximum number of predictors in the largest model
    nk       :: Int = 100, # The number of values of k along the path to consider
    nabort   :: Int = 5, # Number of k values for which prediction on the validation set must decrease before stopping
    q        :: Int64 = 5, 
    est_r    :: Symbol = :None,
    group    :: AbstractVector{Int} = Int[],
    weight   :: AbstractVector{T} = T[],
    zkeep    :: BitVector = trues(size(y, 2) > 1 ? size(z, 1) : size(z, 2)),
    folds    :: AbstractVector{Int} = rand(1:q, is_multivariate(y) ? size(x, 2) : size(x, 1)),
    debias   :: Bool = false,
    verbose  :: Bool = true,
    max_iter :: Int = 200,
    min_iter :: Int = 5,
    init_beta :: Bool = false,
    memory_efficient :: Bool = true
    ) where T <: Float

    typeof(x) <: AbstractSnpArray && throw(ArgumentError("x is a SnpArray! Please convert it to a SnpLinAlg first!"))
    check_data_dim(y, x, z)
    verbose && print_iht_signature()

    # preallocated arrays for efficiency
    test_idx  = [falses(length(folds)) for i in 1:Threads.nthreads()]
    train_idx = [falses(length(folds)) for i in 1:Threads.nthreads()]
    V = [initialize(x, z, y, 1, 1, d, l, group, weight, est_r, false, zkeep,
        memory_efficient=memory_efficient) for _ in 1:q]

    # for displaying cross validation progress
    pmeter = verbose ? Progress(nk * q, "Cross validating...") : nothing

    # define initial search path
    pathstep = round(Int, dfmax / nk)
    searchpath = 1:pathstep:dfmax

    # variables for CMSE 
    path = Int[]
    path_loss = T[]
    fold_loss = zeros(q)
    num_k_tested = 0
    early_stop_counter = 0
    betas = Vector{T}[]
    cs = Vector{T}[]

    while num_k_tested < nk
        # run cross validation for current sparsity
        sparsity = pop!(searchpath)
        fill!(fold_loss, typemax(T))
        ThreadPools.@qthreads for fold in 1:q
            # assign train/test indices
            id = Threads.threadid()
            test_idx[id]  .= folds .== fold
            train_idx[id] .= folds .!= fold

            # run IHT on training data with current (fold, sparsity)
            v = V[id]
            v.k = sparsity
            init_iht_indices!(v, init_beta, train_idx[id], false)
            fit_iht!(v, debias=debias, verbose=false, max_iter=max_iter, min_iter=min_iter)

            # predict on validation data
            v.cv_wts[train_idx[id]] .= zero(T)
            v.cv_wts[test_idx[id]] .= one(T)
            fold_loss[fold] = predict!(v)

            # update progres
            verbose && next!(pmeter)
        end
        push!(path, sparsity)
        push!(path_loss, mean(fold_loss))
        num_k_tested += 1
        early_stop_counter += 1

        # average genetic and non-genetic effect sizes among folds
        beta = zeros(size(x, 2))
        c = zeros(size(z, 2))
        for j in 1:q
            beta .+= V[j].beta
            c .+= V[j].c
        end
        push!(betas, beta ./= q)
        push!(cs, c ./= q)

        # if early stop, define a denser grid between best loss and 3rd best loss (to be prudent)
        early_stop = early_stop_counter ≥ nabort && issorted(@view(mses[end-nabort+1:end]))
        if early_stop
            println("cmsa_iht: Reached early abort, refining grid...")
            perm = partialsortperm(tested_k_mses, 1:3)
            best_k, next_k = tested_k[perm[1]], tested_k[perm[3]]
            if best_k < next_k
                next_k, best_k = best_k, next_k
            end
            # now search between (next_k + pathstep, next_k + 2pathstep, ..., best_k-pathstep)
            pathstep = round(Int, (best_k - next_k) / (nk - num_k_tested))
            if pathstep ≤ 0
                println("Successfully reached early stop, exiting")
                break # increments of k are smaller than 1
            end
            searchpath = next_k+pathstep:pathstep:best_k-pathstep
            early_stop_counter = 0 # reset early stop counter
        end
    end

    return CMSA(path, path_loss, betas, cs)
end

# Distributed memory cv_iht: (TODO merge pmap with multithreading)
# function cv_iht(
#     y        :: AbstractVecOrMat{T},
#     x        :: AbstractMatrix{T},
#     z        :: AbstractVecOrMat{T};
#     d        :: Distribution = is_multivariate(y) ? MvNormal(T[]) : Normal(),
#     l        :: Link = IdentityLink(),
#     path     :: AbstractVector{<:Integer} = 1:20,
#     q        :: Int64 = 5,
#     est_r    :: Symbol = :None,
#     group    :: AbstractVector{Int} = Int[],
#     weight   :: AbstractVector{T} = T[],
#     folds    :: AbstractVector{Int} = rand(1:q, is_multivariate(y) ? size(x, 2) : size(x, 1)),
#     debias   :: Bool = false,
#     verbose  :: Bool = true,
#     parallel :: Bool = true,
#     max_iter :: Int = 100,
#     min_iter :: Int = 20,
#     init_beta :: Bool = false
#     ) where T <: Float

#     typeof(x) <: AbstractSnpArray && throw(ArgumentError("x is a SnpArray! Please convert it to a SnpLinAlg first!"))
#     check_data_dim(y, x, z)

#     # preallocated arrays for efficiency
#     test_idx  = [falses(length(folds)) for i in 1:nprocs()]
#     train_idx = [falses(length(folds)) for i in 1:nprocs()]
#     V = [initialize(x, z, y, 1, 1, d, l, group, weight, est_r, init_beta,
#         cv_train_idx=train_idx[i]) for i in 1:nprocs()]

#     # for displaying cross validation progress
#     pmeter = Progress(q * length(path), "Cross validating...")
#     channel = RemoteChannel(()->Channel{Bool}(q * length(path)), 1)    
#     @async while take!(channel)
#         next!(pmeter)
#     end

#     # cross validation all (fold, k) combinations
#     combinations = allocate_fold_and_k(q, path)
#     mses = (parallel ? pmap : map)(combinations) do (fold, k)
#         # assign train/test indices
#         id = myid()
#         test_idx[id]  .= folds .== fold
#         train_idx[id] .= folds .!= fold
#         v = V[id]

#         # run IHT on training data with current (fold, k)
#         v.k = k
#         init_iht_indices!(v, init_beta, train_idx[id])
#         fit_iht!(v, debias=debias, verbose=false, max_iter=max_iter, min_iter=min_iter)

#         # predict on validation data
#         v.cv_wts[train_idx[id]] .= zero(T)
#         v.cv_wts[test_idx[id]] .= one(T)
#         mse = predict!(v)

#         # update progres
#         put!(channel, true)

#         return mse
#     end
#     put!(channel, false)

#     #weight mses for each fold by their size before averaging
#     mse = meanloss(mses, q, folds)

#     # find best model size and print cross validation result
#     k = path[argmin(mse)] :: Int
#     verbose && print_cv_results(mse, path, k)

#     return mse
# end

function cv_iht(y::AbstractVecOrMat{T}, x::AbstractMatrix; kwargs...) where T
    z = is_multivariate(y) ? ones(T, 1, size(y, 2)) : ones(T, length(y))
    return cv_iht(y, x, z; kwargs...)
end

"""
    allocate_fold_and_k(q, path)

Returns all combinations of (qᵢ, kᵢ) where `q` is number of cross validation fold
and path is a vector of different `k` to be tested. 
"""
function allocate_fold_and_k(q::Int, path::AbstractVector{<:Integer})
    combinations = Tuple{Int, Int}[]
    for fold in 1:q, k in path
        push!(combinations, (fold, k))
    end
    return combinations
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

function predict!(v::IHTVariable{T, M}) where {T <: Float, M}
    # first update mean μ with estimated (trained) beta (cv weights are handled in deviance)
    update_xb!(v)
    update_μ!(v)

    # Compute deviance residual (MSE for Gaussian response)
    return deviance(v)
end

function predict!(v::mIHTVariable{T, M}) where {T <: Float, M}
    # first update mean μ with estimated (trained) beta
    update_xb!(v)
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
function meanloss(fitloss::AbstractVector, q::Int64, folds::AbstractVector{Int})
    ninfold = zeros(Int, q)
    for fold in folds
        ninfold[fold] += 1
    end

    pathsize = Int(length(fitloss) / q)
    loss = zeros(eltype(fitloss), pathsize)
    for j = 1:q
        wfold = convert(eltype(fitloss), ninfold[j] / length(folds))
        for i = 1:pathsize
            loss[i] += fitloss[i + (j - 1) * pathsize] * wfold
        end
    end

    return loss
end
