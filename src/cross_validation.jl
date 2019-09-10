"""
    cv_iht(d, l, x, z, y, J, path, q)

For each model specified in `path`, performs `q`-fold cross validation and 
returns the (averaged) deviance residuals. 

The purpose of this function is to find the best sparsity level `k`, judiciously obtained
from selecting the model with the minimum out-of-sample error. Automatically finds the 
correct version of `L0_reg` to use depending on the type of `x`. By default, each CPU runs 
a different model for a given fold. To use this function, start julia using 4 processors 
(the more the better) by:

    julia> using Distributed
    julia> addprocs(4)

# Warning
Do not remove files with random file names when you run this function. There are 
memory mapped files that will be deleted automatically once they are no longer needed.

# Arguments
- `d`: A distribution (e.g. Normal, Bernoulli)
- `l`: A link function (e.g. Loglink, ProbitLink)
- `x`: A SnpArray, which can be memory mapped to a file. Does not engage in any linear algebra
- `z`: Matrix of non-genetic covariates. The first column usually denotes the intercept. 
- `y`: Response vector
- `J`: The number of maximum groups
- `path`: Vector storing different model sizes
- `q`: Number of cross validation folds. For large data do not set this to be greater than 5

# Optional Arguments: 
- `group` vector storing group membership
- `weight` vector storing vector of weights containing prior knowledge on each SNP
- `folds`: Vector that separates the sample into q disjoint subsets
- `init`: Boolean indicating whether we should initialize IHT algorithm at a good starting guess
- `use_maf`: Boolean indicating we should scale the projection step by a weight vector 
- `debias`: Boolean indicating whether we should debias at each IHT step
- `showinfo`: Whether we want IHT to print meaningful intermediate steps
- `parallel`: Whether we want to run cv_iht using multiple CPUs (highly recommended)
"""
function cv_iht(
    d        :: UnivariateDistribution,
    l        :: Link,
    x        :: Union{AbstractMatrix{T}, SnpArray},
    z        :: AbstractMatrix{T},
    y        :: AbstractVector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    q        :: Int64;
    group    :: AbstractVector{Int} = Int[],
    weight   :: AbstractVector{T} = T[],
    folds    :: DenseVector{Int} = rand(1:q, size(x, 1)),
    init     :: Bool = false,
    use_maf  :: Bool = false,
    debias   :: Bool = false,
    showinfo :: Bool = true,
    parallel :: Bool = false
) where {T <: Float}

    # preallocate mean squared error matrix
    nmodels = length(path)
    mses = zeros(nmodels, q)

    for fold in 1:q
        # find entries that are for test sets and train sets
        test_idx  = folds .== fold
        train_idx = .!test_idx

        # validate trained models on test data by computing deviance residuals
        mses[:, fold] .= train_and_validate(train_idx, test_idx, d, l, x, z, y, J, path, fold, group=group, weight=weight, init=init, use_maf=use_maf, debias=debias, showinfo=false, parallel=parallel)
    end

    #weight mses for each fold by their size before averaging
    mse = meanloss(mses, q, folds)

    # find best model size and print cross validation result
    k = path[argmin(mse)] :: Int
    showinfo && print_cv_results(mse, path, k)

    return mse
end

"""
Performs q-fold cross validation for Iterative hard thresholding to 
determine the best model size `k`. The function is the same as `cv_iht` 
except here each `fold` is distributed to a different CPU as opposed 
to each `path` to a different CPU. 
"""
function cv_iht_distribute_fold(
    d        :: UnivariateDistribution,
    l        :: Link,
    x        :: Union{AbstractMatrix{T}, SnpArray},
    z        :: AbstractMatrix{T},
    y        :: AbstractVector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    q        :: Int64;
    group    :: AbstractVector{Int} = Int[],
    weight   :: AbstractVector{T} = T[],
    folds    :: DenseVector{Int} = rand(1:q, size(x, 1)),
    init     :: Bool = false,
    use_maf  :: Bool = false,
    debias   :: Bool = false,
    showinfo :: Bool = true,
    parallel :: Bool = false
) where {T <: Float}

    # for each fold, allocate train/test set, train the model, and test the model
    mses = (parallel ? pmap : map)(1:q) do fold
        test_idx  = folds .== fold
        train_idx = .!test_idx
        betas, cs = pfold_train(train_idx, x, z, y, J, d, l, path, fold, group=group, weight=weight, init=init, use_maf=use_maf, debias=debias, showinfo=false)
        return pfold_validate(test_idx, betas, cs, x, z, y, J, d, l, path, fold, group=group, weight=weight, init=init, use_maf=use_maf, debias=debias, showinfo=false)
    end

    #weight mses for each fold by their size before averaging
    mse = meanloss(mses, q, folds)

    # find best model size and print cross validation result
    k = path[argmin(mse)] :: Int
    showinfo && print_cv_results(mse, path, k)

    return mse
end

"""
Runs IHT across many different model sizes specifed in `path`. 

This is basically the same as `cv_iht` except we **DO NOT** validate each model 
in a holdout set, meaning that this will definitely induce overfitting as we increase
model size. Use this to perform a quick estimate a range of feasible model sizes before 
engaging in full cross validation. 
"""
function iht_run_many_models(
    d        :: UnivariateDistribution,
    l        :: Link,
    x        :: Union{AbstractMatrix{T}, SnpArray},
    z        :: AbstractMatrix{T},
    y        :: AbstractVector{T},
    J        :: Int64,
    path     :: DenseVector{Int};
    group    :: AbstractVector{Int} = Int[],
    weight   :: AbstractVector{T} = T[],
    init     :: Bool = false,
    use_maf  :: Bool = false,
    debias   :: Bool = false,
    showinfo :: Bool = true,
    parallel :: Bool = false
) where {T <: Float}

    # for each k, run L0_reg and store the loglikelihoods
    typeof(x) == SnpArray && (xbm = SnpBitMatrix{T}(x, model=ADDITIVE_MODEL, center=true, scale=true);)
    results = (parallel ? pmap : map)(path) do k
        if typeof(x) == SnpArray 
            return L0_reg(x, xbm, z, y, 1, k, d, l, group=group, weight=weight, init=init, use_maf=use_maf, debias=debias, show_info=false)
        else 
            return L0_reg(x, z, y, 1, k, d, l, group=group, weight=weight, init=init, use_maf=use_maf, debias=debias, show_info=false)
        end
    end

    loglikelihoods = zeros(size(path, 1))
    for i in 1:length(results)
        loglikelihoods[i] = results[i].logl
    end

    #display result and then return
    showinfo && print_a_bunch_of_path_results(loglikelihoods, path)
    return loglikelihoods
end

"""
This function trains a bunch of models, where each model has a different sparsity 
parameter, k, which is specified in the variable `path`. Then each trained model is used to
compute the deviance residuals (i.e. mean squared error for normal response) on the test set.
on the test set. This deviance residuals vector is returned
"""
function train_and_validate(train_idx::BitArray, test_idx::BitArray, d::UnivariateDistribution, 
                    l::Link, x::SnpArray, z::AbstractMatrix{T}, y::AbstractVector{T}, J::Int64, 
                    path::DenseVector{Int}, fold::Int; group::AbstractVector{Int}=Int[],
                    weight::AbstractVector{T}=T[], init::Bool=false, use_maf::Bool=false, 
                    debias::Bool=false, showinfo::Bool=true, parallel::Bool=false) where {T <: Float}

    # create directory for memory mapping
    train_file = randstring(100) * ".bed"
    test_file = randstring(100) * ".bed"

    # first allocate arrays needed for computing deviance residuals
    p, q = size(x, 2), size(z, 2)
    nmodels = length(path)
    test_size = sum(test_idx)
    xb = zeros(T, test_size)
    zc = zeros(T, test_size)
    μ  = zeros(T, test_size)

    # allocate train model
    x_train = SnpArray(train_file, sum(train_idx), p)
    copyto!(x_train, @view(x[train_idx, :]))
    x_trainbm = SnpBitMatrix{T}(x_train, model=ADDITIVE_MODEL, center=true, scale=true); 

    # allocate test model
    x_test = SnpArray(test_file, test_size, p)
    copyto!(x_test, @view(x[test_idx, :]))
    x_testbm = SnpBitMatrix{T}(x_test, model=ADDITIVE_MODEL, center=true, scale=true); 

    # allocate group and weight vectors if supplied
    group_train = (group == Int[] ? Int[] : group[train_idx])
    weight_train = (weight == T[] ? T[] : weight[train_idx])
    
    # for each k in path, run L0_reg and compute mse
    mses = (parallel ? pmap : map)(path) do k

        #run IHT on training model with given k
        result = L0_reg(x_train, x_trainbm, z[train_idx, :], y[train_idx], 1, k, d, l, group=group_train, weight=weight_train, init=init, use_maf=use_maf, debias=debias, show_info=showinfo)

        # compute estimated response Xb: [xb zc] = [x_test z_test] * [b; c] and update mean μ = g^{-1}(xb)
        A_mul_B!(xb, zc, x_testbm, z[test_idx, :], result.beta, result.c) 
        update_μ!(μ, xb .+ zc, l)

        # compute sum of squared deviance residuals. For normal, this is equivalent to out-of-sample error
        return deviance(d, y[test_idx], μ)
    end

    #clean up 
    rm(train_file, force=true)
    rm(test_file, force=true)

    return mses
end

function train_and_validate(train_idx::BitArray, test_idx::BitArray, d::UnivariateDistribution, 
                    l::Link, x::AbstractMatrix{T}, z::AbstractMatrix{T}, y::AbstractVector{T}, J::Int64, 
                    path::DenseVector{Int}, fold::Int; group::AbstractVector{Int}=Int[],
                    weight::AbstractVector{T}=T[], init::Bool=false, use_maf::Bool=false, 
                    debias::Bool=false, showinfo::Bool=true, parallel::Bool=false) where {T <: Float}

    # first allocate arrays needed for computing deviance residuals
    p, q = size(x, 2), size(z, 2)
    nmodels = length(path)
    test_size = sum(test_idx)
    xb = zeros(T, test_size)
    zc = zeros(T, test_size)
    μ  = zeros(T, test_size)

    # allocate train model
    x_train = x[train_idx, :]
    # copyto!(x_train, @view(x[train_idx, :]))
    # x_trainbm = SnpBitMatrix{Float64}(x_train, model=ADDITIVE_MODEL, center=true, scale=true); 

    # allocate test model
    x_test = x[test_idx, :]
    # copyto!(x_test, @view(x[test_idx, :]))
    # x_testbm = SnpBitMatrix{Float64}(x_test, model=ADDITIVE_MODEL, center=true, scale=true); 

    # allocate group and weight vectors if supplied
    group_train = (group == Int[] ? Int[] : group[train_idx])
    weight_train = (weight == T[] ? T[] : weight[train_idx])
    
    # for each k in path, run L0_reg and compute mse
    mses = (parallel ? pmap : map)(path) do k

        #run IHT on training model with given k
        result = L0_reg(x_train, z[train_idx, :], y[train_idx], 1, k, d, l, group=group_train, weight=weight_train, init=init, use_maf=use_maf, debias=debias, show_info=showinfo)

        # compute estimated response Xb: [xb zc] = [x_test z_test] * [b; c] and update mean μ = g^{-1}(xb)
        A_mul_B!(xb, zc, x_test, z[test_idx, :], result.beta, result.c) 
        update_μ!(μ, xb .+ zc, l)

        # compute sum of squared deviance residuals. For normal, this is equivalent to out-of-sample error
        return deviance(d, y[test_idx], μ)
    end

    return mses
end

function pfold_train(train_idx::BitArray, x::SnpArray, z::AbstractMatrix{T},
                    y::AbstractVector{T}, J::Int64, d::UnivariateDistribution,
                    l::Link, path::DenseVector{Int}, fold::Int64; 
                    group::AbstractVector{Int}=Int[], weight::AbstractVector{T}=T[],
                    init::Bool=false, use_maf::Bool =false, max_iter::Int = 100,
                    max_step::Int = 3, debias::Bool = false, showinfo::Bool = false
                    ) where {T <: Float}

    # create directory for memory mapping
    train_file = randstring(100) * ".bed"

    #preallocate arrays
    p, q = size(x, 2), size(z, 2)
    nmodels = length(path)
    betas = zeros(T, p, nmodels)
    cs = zeros(T, q, nmodels)

    # allocate training data
    x_train = SnpArray(train_file, sum(train_idx), p)
    copyto!(x_train, view(x, train_idx, :))
    x_trainbm = SnpBitMatrix{T}(x_train, model=ADDITIVE_MODEL, center=true, scale=true); 

    for i in 1:length(path)
        k = path[i]
        result = L0_reg(x_train, x_trainbm, z[train_idx, :], y[train_idx], 1, k, d, l, group=group, weight=weight, init=init, use_maf=use_maf, debias=debias, show_info=false)
        betas[:, i] .= result.beta
        cs[:, i] .= result.c
    end

    #clean up
    rm(train_file, force=true)   

    return betas, cs
end

"""
This function takes a trained model, and returns the mean squared error (mse) of that model 
on the test set. A vector of mse is returned, where each entry corresponds to the training
set on each fold with different sparsity parameter. 
"""
function pfold_validate(test_idx::BitArray, betas::AbstractMatrix{T}, cs::AbstractMatrix{T},
                        x::SnpArray, z::AbstractMatrix{T}, y::AbstractVector{T}, J::Int64,
                        d::UnivariateDistribution, l::Link, path::DenseVector{Int}, fold::Int64; 
                        group::AbstractVector{Int}=Int[], weight::AbstractVector{T}=T[], 
                        init::Bool=false, use_maf::Bool = false, max_iter::Int = 100, 
                        max_step::Int = 3, debias::Bool = false, showinfo::Bool = false
                        ) where {T <: Float}
    
    # create directory for memory mapping
    test_file = randstring(100) * ".bed"

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

    # for each computed model stored in betas, compute the deviance residuals (i.e. generalized mean squared error) on test set
    for i = 1:size(betas, 2)
        # compute estimated response Xb: [xb zc] = [x_test z_test] * [b; c] and update mean μ = g^{-1}(xb)
        A_mul_B!(xb, zc, x_testbm, z[test_idx, :], @view(betas[:, i]), @view(cs[:, i])) 
        update_μ!(μ, xb .+ zc, l)

        # compute sum of squared deviance residuals. For normal, this is equivalent to out-of-sample error
        mse[i] = deviance(d, y_test, μ)
    end

    #clean up
    rm(test_file, force=true)

    return mse
end

"""
This function scale mean squared errors (deviance residuals) for each fold by its own fold size.
"""
function meanloss(fitloss::Matrix{T}, q::Int64, 
                  folds::DenseVector{Int}) where {T <: Float}
    ninfold = zeros(Int, q)
    for fold in folds
        ninfold[fold] += 1
    end

    loss = zeros(T, size(fitloss, 1))
    for j = 1:size(fitloss, 2)
        wfold = convert(T, ninfold[j]/length(folds))
        for i = 1:size(fitloss, 1)
            loss[i] += fitloss[i, j]*wfold
        end
    end

    return loss
end

function meanloss(mses::Vector{Vector{T}}, num_fold::Int64, 
                  folds::DenseVector{Int}) where {T <: Float}

    fitloss = hcat(mses...) :: Matrix{T}
    ninfold = zeros(Int, num_fold)
    for fold in folds
        ninfold[fold] += 1
    end

    loss = zeros(size(fitloss, 1))
    for j = 1:size(fitloss, 2)
        wfold = convert(T, ninfold[j]/length(folds))
        for i = 1:size(fitloss, 1)
            loss[i] += fitloss[i, j]*wfold
        end
    end

    return loss
end
