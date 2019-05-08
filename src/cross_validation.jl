"""
    cv_iht(d, l, x, z, y, J, path, num_fold)

Performs q-fold cross validation and returns the deviance residuals for each model. 

Each CPU runs a different model for a given fold. To use this function, start julia 
using 4 (the more the better) processors by:

    julia> using Distributed
    julia> addprocs(4)

# Arguments
- `d`: A distribution (e.g. Normal, Bernoulli)
- `l`: A link function (e.g. Loglink, ProbitLink)
- `x`: A SnpArray, which can be memory mapped to a file. Does not engage in any linear algebra
- `z`: Matrix of non-genetic covariates. The first column usually denotes the intercept. 
- `y`: Response vector
- `J`: The number of maximum groups
- `path`: Vector storing different model sizes
- `num_fold`: Number of cross validation folds. For large data do not set this to be greater than 5

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
    x        :: SnpArray,
    z        :: AbstractMatrix{T},
    y        :: AbstractVector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    num_fold :: Int64;
    group    :: AbstractVector{Int} = Int[],
    weight   :: AbstractVector{T} = T[],
    folds    :: DenseVector{Int} = rand(1:num_fold, size(x, 1)),
    init     :: Bool = false,
    use_maf  :: Bool = false,
    debias   :: Bool = false,
    showinfo :: Bool = true,
    parallel :: Bool = false
) where {T <: Float}

    # preallocate mean squared error matrix
    nmodels = length(path)
    mses = zeros(nmodels, num_fold)

    for fold in 1:num_fold
        # find entries that are for test sets and train sets
        test_idx  = folds .== fold
        train_idx = .!test_idx

        # validate trained models on test data by computing deviance residuals
        mses[:, fold] .= train_and_validate(train_idx, test_idx, d, l, x, z, y, J, path, fold, group=group, weight=weight, init=init, use_maf=use_maf, debias=debias, showinfo=false, parallel=parallel)
    end

    #weight mses for each fold by their size before averaging
    mse = meanloss(mses, num_fold, folds)

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
    x        :: SnpArray,
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
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    results = (parallel ? pmap : map)(path) do k
        return L0_reg(x, xbm, z, y, 1, k, d, l, group=group, weight=weight, init=init, use_maf=use_maf, debias=debias, show_info=false)
    end

    loglikelihoods = zeros(size(path, 1))
    for i in 1:length(results)
        loglikelihoods[i] = results[i].logl
    end

    #display result and then return
    showinfo && print_a_bunch_of_path_results(loglikelihoods, path)
    return [path loglikelihoods]
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
    train_file = "train_tmp$fold.bed"
    test_file = "test_tmp$fold.bed"

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
    x_trainbm = SnpBitMatrix{Float64}(x_train, model=ADDITIVE_MODEL, center=true, scale=true); 

    # allocate test model
    x_test = SnpArray(test_file, test_size, p)
    copyto!(x_test, @view(x[test_idx, :]))
    x_testbm = SnpBitMatrix{Float64}(x_test, model=ADDITIVE_MODEL, center=true, scale=true); 

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

"""
This function scale mean squared errors (deviance residuals) for each fold by its own fold size.
"""
function meanloss(fitloss::Matrix{Float64}, num_fold::Int64, folds::DenseVector{Int})
    ninfold = zeros(Int, num_fold)
    for fold in folds
        ninfold[fold] += 1
    end

    loss = zeros(size(fitloss, 1))
    for j = 1:size(fitloss, 2)
        wfold = ninfold[j]/length(folds)
        for i = 1:size(fitloss, 1)
            loss[i] += fitloss[i, j]*wfold
        end
    end

    return loss
end
