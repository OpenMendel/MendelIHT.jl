"""
    cv_iht(y, x, z; kwargs...)

For each model specified in `path`, performs `q`-fold cross validation and 
returns the (averaged) deviance residuals. 

The purpose of this function is to find the best sparsity level `k`, obtained
from selecting the model with the minimum out-of-sample error. Different cross
validation folds are cycled through sequentially different `paths` are fitted
in parallel on different CPUs. Currently there are no routines to cross validate
different group sizes. 

# Warning
Do not remove files with random file names when you run this function. These are 
memory mapped files that will be deleted automatically once they are no longer needed.

# Arguments
- `y`: Response vector (phenotypes), should be an `Array{T, 1}`.
- `x`: A design matrix (genotypes). Should be a `SnpArray` or an `Array{T, 2}`. 
- `z`: Matrix of non-genetic covariates of type `Array{T, 2}` or `Array{T, 1}`. The first column should be the intercept (i.e. column of 1). 

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
    y        :: AbstractVector{T},
    x        :: AbstractMatrix,
    z        :: AbstractVecOrMat{T};
    d        :: UnivariateDistribution = Normal(),
    l        :: Link = IdentityLink(),
    path     :: AbstractVector{<:Integer} = 1:20,
    q        :: Int64 = 5,
    est_r    :: Symbol = :None,
    group    :: AbstractVector{Int} = Int[],
    weight   :: AbstractVector{T} = T[],
    folds    :: AbstractVector{Int} = rand(1:q, size(x, 1)),
    destin   :: String = "./",
    use_maf  :: Bool = false,
    debias   :: Bool = false,
    verbose  :: Bool = true,
    parallel :: Bool = false
    ) where T <: Float

    # preallocate mean squared error matrix
    nmodels = length(path)
    mses = zeros(nmodels, q)

    for fold in 1:q
        # find entries that are for test sets and train sets
        test_idx  = folds .== fold
        train_idx = .!test_idx

        # validate trained models on test data by computing deviance residuals
        mses[:, fold] .= train_and_validate(train_idx, test_idx, d, l, x, z, y,
            path, est_r, group=group, weight=weight, destin=destin, 
            use_maf=use_maf, debias=debias, verbose=false, parallel=parallel)
    end

    #weight mses for each fold by their size before averaging
    mse = meanloss(mses, q, folds)

    # find best model size and print cross validation result
    k = path[argmin(mse)] :: Int
    verbose && print_cv_results(mse, path, k)

    return mse
end

cv_iht(y::AbstractVector{T}, x::AbstractMatrix; kwargs...) where T = 
    cv_iht(y, x, ones(T, size(x, 1)); kwargs...)

"""
Performs q-fold cross validation for Iterative hard thresholding to 
determine the best model size `k`. The function is the same as `cv_iht` 
except here each `fold` is distributed to a different CPU as opposed 
to each `path` to a different CPU. 

This function has the edge over `cv_iht` because one can fit different 
sparsity levels on different computers. But this is assuming you have 
enough RAM and disk space to store all training data simultaneously.
"""
function cv_iht_distribute_fold(
    y        :: AbstractVector{T},
    x        :: AbstractMatrix,
    z        :: AbstractVecOrMat{T};
    d        :: UnivariateDistribution = Normal(),
    l        :: Link = IdentityLink,
    path     :: AbstractVector{<:Integer} = 1:20,
    q        :: Int64 = 5,
    est_r    :: Symbol = :None,
    group    :: AbstractVector{Int} = Int[],
    weight   :: AbstractVector{T} = T[],
    folds    :: AbstractVector{Int} = rand(1:q, size(x, 1)),
    destin   :: String = "./", 
    use_maf  :: Bool = false,
    debias   :: Bool = false,
    verbose  :: Bool = true,
    parallel :: Bool = false
    ) where T <: Float

    # for each fold, allocate train/test set, train the model, and test the model
    mses = (parallel ? pmap : map)(1:q) do fold
        test_idx  = folds .== fold
        train_idx = .!test_idx
        betas, cs = pfold_train(train_idx, x, z, y, d, l, path, est_r, 
            group=group, weight=weight, destin=destin, 
            use_maf=use_maf, debias=debias, verbose=false)
        return pfold_validate(test_idx, betas, cs, x, z, y, d, l, path,
            group=group, weight=weight, destin=destin, 
            use_maf=use_maf, debias=debias, verbose=false)
    end

    #weight mses for each fold by their size before averaging
    mse = meanloss(mses, q, folds)

    # find best model size and print cross validation result
    k = path[argmin(mse)] :: Int
    verbose && print_cv_results(mse, path, k)

    return mse
end

cv_iht_distribute_fold(y::AbstractVector{T}, x::AbstractMatrix; kwargs...) where T =
    cv_iht_distribute_fold(y, x, ones(T, size(x, 1)); kwargs...)

"""
Runs IHT across many different model sizes specifed in `path` using the full
design matrix. Same as `cv_iht` but **DOES NOT** validate in a holdout set, meaning
that this will definitely induce overfitting as we increase model size.
Use this if you want to quickly estimate a range of feasible model sizes before 
engaging in full cross validation. 
"""
function iht_run_many_models(
    y        :: AbstractVector{T},
    x        :: AbstractMatrix,
    z        :: AbstractVecOrMat{T};
    d        :: UnivariateDistribution,
    l        :: Link,
    path     :: AbstractVector{Int},
    est_r    :: Symbol = :None,
    group    :: AbstractVector{Int} = Int[],
    weight   :: AbstractVector{T} = Float64[],
    use_maf  :: Bool = false,
    debias   :: Bool = false,
    verbose  :: Bool = true,
    parallel :: Bool = false
    ) where T <: Float

    # for each k, fit model and store the loglikelihoods
    results = (parallel ? pmap : map)(path) do k
        if typeof(x) == SnpArray 
            xla = SnpLinAlg{T}(x, model=ADDITIVE_MODEL, center=true, scale=true, 
                impute=true)
            return fit_iht(y, xla, z, J=1, k=k, d=d, l=l, est_r=est_r, group=group, 
                weight=weight, use_maf=use_maf, debias=debias, verbose=false)
        else 
            return fit_iht(y, x, z, J=1, k=k, d=d, l=l, est_r=est_r, group=group, 
                weight=weight, use_maf=use_maf, debias=debias, verbose=false)
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

iht_run_many_models(y::AbstractVector{T}, x::AbstractMatrix; kwargs...) where T =
    iht_run_many_models(y, x, ones(T, size(x, 1)); kwargs...)

"""
This function trains a bunch of models, where each model has a different sparsity 
parameter, k, which is specified in the variable `path`. Then each trained model is used to
compute the deviance residuals (i.e. mean squared error for normal response) on the test set.
This deviance residuals vector is returned
"""
function train_and_validate(train_idx::BitArray, test_idx::BitArray,
    d::UnivariateDistribution, l::Link, x::SnpArray, z::AbstractVecOrMat{T},
    y::AbstractVector{T}, path::AbstractVector{Int}, est_r::Symbol;
    group::AbstractVector=Int[], weight::AbstractVector{T}=T[],
    destin::String = "./", use_maf::Bool=false,
    debias::Bool=false, verbose::Bool=true, parallel::Bool=false
    ) where {T <: Float}

    # create directory for memory mapping
    train_file = destin * randstring(100) * ".bed"
    test_file  = destin * randstring(100) * ".bed"

    # first allocate arrays needed for computing deviance residuals
    p, q = size(x, 2), size(z, 2)
    nmodels = length(path)
    test_size = sum(test_idx)
    xb = zeros(T, test_size, nprocs())
    zc = zeros(T, test_size, nprocs())
    μ  = zeros(T, test_size, nprocs())

    # allocate train model
    x_train = SnpArray(train_file, sum(train_idx), p)
    copyto!(x_train, @view(x[train_idx, :]))
    x_trainla = SnpLinAlg{T}(x_train, model=ADDITIVE_MODEL, center=true,
        scale=true, impute=true)
    z_train = z[train_idx, :]
    y_train = y[train_idx]

    # allocate test model
    x_test = SnpArray(test_file, test_size, p)
    copyto!(x_test, @view(x[test_idx, :]))
    x_testla = SnpLinAlg{T}(x_test, model=ADDITIVE_MODEL, center=true,
        scale=true, impute=true)
    z_test = z[test_idx, :]
    y_test = y[test_idx]

    # allocate group and weight vectors if supplied
    group_train = (group == Int[] ? Int[] : group[train_idx])
    weight_train = (weight == T[] ? T[] : weight[train_idx])
    
    # for each k in path, fit and compute mse
    mses = try # this try statement enables deletion of intermediate files if `fit` fails
        mses = (parallel ? pmap : map)(path) do k

            #run IHT on training model with given k
            result = fit_iht(y_train, x_trainla, z_train, J=1, k=k, d=d, l=l,
                est_r=est_r, group=group_train, weight=weight_train, 
                use_maf=use_maf, debias=debias, verbose=verbose)

            # compute estimated response Xb: [xb zc] = [x_test z_test] * [b; c] and update mean μ = g^{-1}(xb)
            id = myid()
            A_mul_B!(@view(xb[:, id]), @view(zc[:, id]), x_testla, z_test,
                result.beta, result.c) 
            update_μ!(@view(μ[:, id]), @view(xb[:, id]) .+ @view(zc[:, id]), l)

            # compute sum of squared deviance residuals. For normal, this is equivalent to out-of-sample error
            return deviance(d, y_test, @view(μ[:, id]))
        end

        return mses
    finally 
        #clean up 
        try 
            rm(train_file, force=true) 
            rm(test_file, force=true)
        catch
            println("Can't remove intermediate file! Windows users need to delete intermediate files manually.")
        end
    end

    return mses
end

# for general matrix x 
function train_and_validate(train_idx::BitArray, test_idx::BitArray, d::UnivariateDistribution, 
    l::Link, x::AbstractMatrix{T}, z::AbstractVecOrMat{T}, y::AbstractVector{T}, 
    path::AbstractVector{Int}, est_r::Symbol;
    group::AbstractVector{Int}=Int[], weight::AbstractVector{T}=T[],
    destin::String = "./", use_maf::Bool=false,
    debias::Bool=false, verbose::Bool=true, parallel::Bool=false
    ) where {T <: Float}

    # first allocate arrays needed for computing deviance residuals
    p, q = size(x, 2), size(z, 2)
    nmodels = length(path)
    test_size = sum(test_idx)
    xb = zeros(T, test_size)
    zc = zeros(T, test_size)
    μ  = zeros(T, test_size)

    # allocate train model and test models
    x_train = x[train_idx, :]
    z_train = z[train_idx, :]
    y_train = y[train_idx]
    x_test = x[test_idx, :]
    z_test = z[test_idx, :]
    y_test = y[test_idx]

    # allocate group and weight vectors if supplied
    group_train = (group == Int[] ? Int[] : group[train_idx])
    weight_train = (weight == T[] ? T[] : weight[train_idx])
    
    # for each k in path, fit and compute mse
    mses = (parallel ? pmap : map)(path) do k

        #run IHT on training model with given k
        result = fit_iht(y_train, x_train, z_train, J=1, k=k, d=d, l=l, est_r=est_r,
            group=group_train, weight=weight_train, use_maf=use_maf,
            debias=debias, verbose=verbose)

        # compute estimated response Xb: [xb zc] = [x_test z_test] * [b; c] and update mean μ = g^{-1}(xb)
        A_mul_B!(xb, zc, x_test, z_test, result.beta, result.c) 
        update_μ!(μ, xb .+ zc, l)

        # compute sum of squared deviance residuals. For normal, this is equivalent to out-of-sample error
        return deviance(d, y_test, μ)
    end

    return mses
end

"""
Creates training model of `x` based on `train_idx` and returns `betas` and `cs` that stores the 
estimated coefficients for genetic and non-genetic predictors. 

This function initialize the training model as a memory-mapped file at a `destin`, which will
be removed upon completion. 
"""
function pfold_train(train_idx::BitArray, x::SnpArray, z::AbstractVecOrMat{T},
    y::AbstractVector{T}, d::UnivariateDistribution, l::Link, 
    path::AbstractVector{Int}, est_r::Symbol;
    group::AbstractVector{Int}=Int[], weight::AbstractVector{T}=T[],
    destin::String = "./", use_maf::Bool =false,
    max_iter::Int = 100, max_step::Int = 3, debias::Bool = false,
    verbose::Bool = false
    ) where {T <: Float}

    # create directory for memory mapping
    train_file = destin * randstring(100) * ".bed"

    #preallocate arrays
    p, q = size(x, 2), size(z, 2)
    nmodels = length(path)
    betas = zeros(T, p, nmodels)
    cs = zeros(T, q, nmodels)

    # allocate training data
    x_train = SnpArray(train_file, sum(train_idx), p)
    copyto!(x_train, view(x, train_idx, :))
    x_trainla = SnpLinAlg{T}(x_train, model=ADDITIVE_MODEL, center=true,
        scale=true, impute=true)
    y_train = y[train_idx]
    z_train = z[train_idx, :]

    try 
        for i in 1:length(path)
            k = path[i]
            result = fit_iht(y_train, x_trainla, z_train, J=1, k=k, d=d, l=l, 
                est_r=est_r, group=group, weight=weight, 
                use_maf=use_maf, debias=debias, verbose=false)
            betas[:, i] .= result.beta
            cs[:, i] .= result.c
        end
    finally
        try 
            rm(train_file, force=true) 
        catch
            println("Can't remove intermediate file! Windows users need to delete intermediate files manually.")
        end    
    end

    return betas, cs
end

function pfold_train(train_idx::BitArray, x::AbstractMatrix{T}, z::AbstractVecOrMat{T},
    y::AbstractVector{T}, d::UnivariateDistribution, l::Link, 
    path::AbstractVector{Int}, est_r::Union{Symbol, Nothing};
    group::AbstractVector{Int}=Int[], weight::AbstractVector{T}=T[],
    destin::String = "./", use_maf::Bool =false,
    max_iter::Int = 100, max_step::Int = 3, debias::Bool = false,
    verbose::Bool = false
    ) where {T <: Float}

    #preallocate arrays
    p, q = size(x, 2), size(z, 2)
    nmodels = length(path)
    betas = zeros(T, p, nmodels)
    cs = zeros(T, q, nmodels)

    # allocate training model
    x_train = x[train_idx, :]
    y_train = y[train_idx]
    z_train = z[train_idx, :]

    # fit training model on various sparsity levels
    for i in 1:length(path)
        k = path[i]
        result = fit_iht(y_train, x_train, z_train, J=1, k=k, d=d, l=l, group=group,
            weight=weight, use_maf=use_maf, debias=debias,
            est_r=est_r, verbose=false) 
        betas[:, i] .= result.beta
        cs[:, i] .= result.c
    end

    return betas, cs
end

"""
This function takes a trained model, and returns the mean squared error (mse) of that model 
on the test set. A vector of mse is returned, where each entry corresponds to the training
set on each fold with different sparsity parameter. 
"""
function pfold_validate(test_idx::BitArray, betas::AbstractMatrix{T}, 
    cs::AbstractMatrix{T}, x::SnpArray, z::AbstractVecOrMat{T}, y::AbstractVector{T},
    d::UnivariateDistribution, l::Link, path::AbstractVector{Int};
    group::AbstractVector{Int}=Int[], weight::AbstractVector{T}=T[],
    destin::String = "./", use_maf::Bool = false, 
    max_iter::Int = 100, max_step::Int = 3, debias::Bool = false,
    verbose::Bool = false) where {T <: Float}
    
    # create directory for memory mapping
    test_file = destin * randstring(100) * ".bed"

    # preallocate arrays
    p, q = size(x, 2), size(z, 2)
    test_size = sum(test_idx)
    mse = zeros(T, length(path))
    xb = zeros(T, test_size)
    zc = zeros(T, test_size)
    μ  = zeros(T, test_size)

    # allocate test model
    x_test = SnpArray(test_file, sum(test_idx), p)
    y_test = y[test_idx]
    copyto!(x_test, @view(x[test_idx, :]))
    x_testbm = SnpBitMatrix{T}(x_test, model=ADDITIVE_MODEL, center=true, scale=true); 
    z_test = z[test_idx, :]

    # for each computed model stored in betas, compute the deviance residuals (i.e. generalized mean squared error) on test set
    try
        for i = 1:size(betas, 2)
            # compute estimated response Xb: [xb zc] = [x_test z_test] * [b; c] and update mean μ = g^{-1}(xb)
            A_mul_B!(xb, zc, x_testbm, z_test, @view(betas[:, i]), @view(cs[:, i])) 
            update_μ!(μ, xb .+ zc, l)

            # compute sum of squared deviance residuals. For normal, this is equivalent to out-of-sample error
            mse[i] = deviance(d, y_test, μ)
        end
    finally
        try 
            rm(test_file, force=true)
        catch
            println("Can't remove intermediate file! Windows users need to delete intermediate files manually.")
        end    
    end

    return mse
end

function pfold_validate(test_idx::BitArray, betas::AbstractMatrix{T},
    cs::AbstractMatrix{T}, x::AbstractMatrix{T}, z::AbstractVecOrMat{T},
    y::AbstractVector{T}, d::UnivariateDistribution, l::Link,
    path::AbstractVector{Int}; group::AbstractVector{Int}=Int[], 
    weight::AbstractVector{T}=T[], destin::String = "./", 
    use_maf::Bool = false, max_iter::Int = 100, max_step::Int = 3,
    debias::Bool = false, verbose::Bool = false
    ) where {T <: Float}

    # preallocate arrays
    p, q = size(x, 2), size(z, 2)
    test_size = sum(test_idx)
    mse = zeros(T, length(path))
    xb = zeros(T, test_size)
    zc = zeros(T, test_size)
    μ  = zeros(T, test_size)

    # allocate test model
    x_test = x[test_idx, :]
    y_test = y[test_idx]
    z_test = z[test_idx, :]

    # Compute the deviance residuals on test set for each computed model stored in betas, c
    for i = 1:size(betas, 2)
        # compute estimated response Xb: [xb zc] = [x_test z_test] * [b; c] and update mean μ = g^{-1}(xb)
        A_mul_B!(xb, zc, x_test, z_test, @view(betas[:, i]), @view(cs[:, i])) 
        update_μ!(μ, xb .+ zc, l)

        # compute sum of squared deviance residuals. For normal, this is equivalent to out-of-sample error
        mse[i] = deviance(d, y_test, μ)
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
