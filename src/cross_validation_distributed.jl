"""
This function performs q-fold cross validation for Iterative hard thresholding to determine 
the best model size k. Each fold is distributed to a different core (CPU). 

To use this function, start julia using 2 (or more) processors by

    julia -p 2

Some description of variables: 
`path`: Vector storing different model sizes
`folds`: Vector that separates the sample into q disjoint subsets.
"""
function cv_iht_distributed(
    d        :: UnivariateDistribution,
    l        :: Link,
    x        :: SnpArray,
    z        :: AbstractMatrix{T},
    y        :: AbstractVector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    num_fold :: Int64;
    use_maf  :: Bool = false,
    debias   :: Bool = false,
    showinfo :: Bool = true,
    parallel :: Bool = false
) where {T <: Float}

    # for each fold, allocate train/test set, train the model, and test the model
    mses = (parallel ? pmap : map)(1:num_fold) do fold
        test_idx  = folds .== fold
        train_idx = .!test_idx
        betas, cs = pfold_train(train_idx, x, z, y, J, d, l, path, fold, use_maf=use_maf, debias=debias, showinfo=false)
        return pfold_validate(test_idx, betas, cs, x, z, y, J, d, l, path, fold, use_maf=use_maf, debias=debias, showinfo=false)
    end

    #weight mses for each fold by their size before averaging
    mse = meanloss(mses, num_fold, folds)

    # find best model size and print cross validation result
    k = path[argmin(mse)] :: Int
    showinfo && print_cv_results(mse, path, k)

    return mse
end

function cv_iht_distributed2(
    d        :: UnivariateDistribution,
    l        :: Link,
    x        :: SnpArray,
    z        :: AbstractMatrix{T},
    y        :: AbstractVector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    num_fold :: Int64;
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

        # train IHT on different paths of train model
        betas, cs = pfold_train2(train_idx, d, l, x, z, y, 1, path, parallel=parallel, use_maf=use_maf, debias=debias, showinfo=false);

        # validate trained models on test data by computing deviance residuals
        mses[:, fold] .= pfold_validate2(test_idx, betas, cs, x, z, y, J, d, l, path, fold, use_maf=use_maf, debias=debias, showinfo=false)
    end

    #weight mses for each fold by their size before averaging
    mse = meanloss(mses, num_fold, folds)

    # find best model size and print cross validation result
    k = path[argmin(mse)] :: Int
    showinfo && print_cv_results(mse, path, k)

    return mse
end

function iht_run_many_models(
    d        :: UnivariateDistribution,
    l        :: Link,
    x        :: SnpArray,
    z        :: AbstractMatrix{T},
    y        :: AbstractVector{T},
    J        :: Int64,
    path     :: DenseVector{Int};
    use_maf  :: Bool = false,
    debias   :: Bool = false,
    showinfo :: Bool = true,
    parallel :: Bool = false
) where {T <: Float}

    # for each k, run L0_reg and store the loglikelihoods
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    results = (parallel ? pmap : map)(path) do k
        return L0_reg(x, xbm, z, y, 1, k, d, l, debias=debias, init=false, show_info=false)
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
This function trains a bunch of models using the training set, where each model has a 
different sparsity parameter, k, which is specified in the variable `path`. The trained model 
is stored in each column of betas (coefficients for each SNP) and cs (coefficients for 
intercept and non genetic covariates). 
"""
function pfold_train(train_idx::BitArray, x::SnpArray, z::AbstractMatrix{T},
                    y::AbstractVector{T}, J::Int64, d::UnivariateDistribution,
                    l::Link, path::DenseVector{Int}, fold::Int64; 
                    use_maf::Bool =false, max_iter::Int = 100,max_step::Int = 3, 
                    debias::Bool = false, showinfo::Bool = false) where {T <: Float}

    #preallocate arrays
    p, q = size(x, 2), size(z, 2)
    nmodels = length(path)
    betas = zeros(p, nmodels)
    cs = zeros(q, nmodels)

    # allocate training datas
    # x_train = SnpArray("x_train_fold$fold.bed", sum(train_idx), p)
    x_train = SnpArray(undef, sum(train_idx), p)
    y_train = y[train_idx]
    z_train = z[train_idx, :]
    copyto!(x_train, @view x[train_idx, :])
    x_trainbm = SnpBitMatrix{Float64}(x_train, model=ADDITIVE_MODEL, center=true, scale=true); 

    for i in 1:length(path)
        k = path[i]
        result = L0_reg(x_train, x_trainbm, z_train, y_train, 1, k, d, l, debias=debias, init=false, show_info=showinfo)
        betas[:, i] .= result.beta
        cs[:, i] .= result.c
    end

    #clean up
    # rm("x_train_fold$fold.bed", force=true)   

    return betas, cs
end

function pfold_train2(train_idx::BitArray, d::UnivariateDistribution, l::Link, x::SnpArray,
                    z::AbstractMatrix{T}, y::AbstractVector{T}, J::Int64, path::DenseVector{Int};
                    use_maf::Bool=false, debias::Bool=false, showinfo::Bool=true,
                    parallel::Bool=false) where {T <: Float}

    # first allocate return arrays
    p, q = size(x, 2), size(z, 2)
    nmodels = length(path)
    betas = zeros(p, nmodels)
    cs = zeros(q, nmodels)

    # next allocate train model
    x_train = SnpArray(undef, sum(train_idx), p)
    y_train = y[train_idx]
    z_train = z[train_idx, :]
    copyto!(x_train, @view(x[train_idx, :]))
    x_trainbm = SnpBitMatrix{Float64}(x_train, model=ADDITIVE_MODEL, center=true, scale=true); 

    # for each k, run L0_reg 
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    trained_models = (parallel ? pmap : map)(path) do k
        return L0_reg(x_train, x_trainbm, z_train, y_train, 1, k, d, l, debias=debias, init=false, show_info=false)
    end

    #copy result into beta's and c's
    for i in 1:length(path)
        betas[:, i] .= trained_models[i].beta
        cs[:, i] .= trained_models[i].c
    end

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
                        use_maf::Bool = false, max_iter::Int = 100, max_step::Int = 3,
                        debias::Bool = false, showinfo::Bool = false) where {T <: Float}
    
    # preallocate arrays
    p, q = size(x, 2), size(z, 2)
    test_size = sum(test_idx)
    mse = zeros(T, length(path))
    xb = zeros(T, test_size)
    zc = zeros(T, test_size)
    μ  = zeros(T, test_size)

    # allocate test model
    # x_test = SnpArray("x_test_fold$fold.bed", sum(test_idx), p)
    x_test = SnpArray(undef, sum(test_idx), p)
    y_test = @view(y[test_idx])
    z_test = @view(z[test_idx, :])
    copyto!(x_test, @view(x[test_idx, :]))
    x_testbm = SnpBitMatrix{Float64}(x_test, model=ADDITIVE_MODEL, center=true, scale=true); 

    # for each computed model stored in betas, compute the deviance residuals (i.e. generalized mean squared error) on test set
    for i = 1:size(betas, 2)
        # compute estimated response Xb: [xb zc] = [x_test z_test] * [b; c] and update mean μ = g^{-1}(xb)
        A_mul_B!(xb, zc, x_testbm, z_test, @view(betas[:, i]), @view(cs[:, i])) 
        update_μ!(μ, xb .+ zc, l)

        # compute sum of squared deviance residuals. For normal, this is equivalent to out-of-sample error
        mse[i] = deviance(d, y_test, μ)
    end

    #clean up
    # rm("x_test_fold$fold.bed", force=true)

    return mse
end

function pfold_validate2(test_idx::BitArray, betas::AbstractMatrix{T}, cs::AbstractMatrix{T},
                        x::SnpArray, z::AbstractMatrix{T}, y::AbstractVector{T}, J::Int64,
                        d::UnivariateDistribution, l::Link, path::DenseVector{Int}, fold::Int64; 
                        use_maf::Bool = false, max_iter::Int = 100, max_step::Int = 3,
                        debias::Bool = false, showinfo::Bool = false) where {T <: Float}
    
    # preallocate arrays
    p, q = size(x, 2), size(z, 2)
    test_size = sum(test_idx)
    mse = zeros(T, length(path))
    xb = zeros(T, test_size)
    zc = zeros(T, test_size)
    μ  = zeros(T, test_size)

    # allocate test model
    x_test = SnpArray(undef, sum(test_idx), p)
    y_test = @view(y[test_idx])
    z_test = @view(z[test_idx, :])
    copyto!(x_test, @view(x[test_idx, :]))
    x_testbm = SnpBitMatrix{Float64}(x_test, model=ADDITIVE_MODEL, center=true, scale=true); 

    # for each computed model stored in betas, compute the deviance residuals (i.e. generalized mean squared error) on test set
    for i = 1:size(betas, 2)
        # compute estimated response Xb: [xb zc] = [x_test z_test] * [b; c] and update mean μ = g^{-1}(xb)
        A_mul_B!(xb, zc, x_testbm, z_test, @view(betas[:, i]), @view(cs[:, i])) 
        update_μ!(μ, xb .+ zc, l)

        # compute sum of squared deviance residuals. For normal, this is equivalent to out-of-sample error
        mse[i] = deviance(d, y_test, μ)
    end

    return mse
end

"""
This function scale mean squared errors (deviance residuals) for each fold by its own fold size.
"""
function meanloss(mses::Vector{Vector{Float64}}, num_fold::Int64, folds::DenseVector{Int})
    fitloss = hcat(mses...) :: Matrix{Float64}

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
