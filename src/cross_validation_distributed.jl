"""
This function performs q-fold cross validation for Iterative hard thresholding to determine 
the best model size k. To use this function, start julia using 4 (or more) processors by

IMPORTANT: The user must manually ensure each processor has sufficient memory.

`path`: Vector storing different model sizes
`folds`: Vector that separates the sample into q disjoint subsets.
"""
function cv_iht_distributed(
    x        :: SnpArray,
    z        :: AbstractMatrix{T},
    y        :: AbstractVector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    num_fold :: Int64;
    use_maf  :: Bool = false,
    glm      :: String = "normal",
    debias   :: Bool = false,
    showinfo :: Bool = false,
    parallel :: Bool = false
) where {T <: Float}

    # for each fold, allocate train/test set, train the model, and test the model
    mses = (parallel ? pmap : map)(1:num_fold) do fold
        test_idx  = folds .== fold
        train_idx = .!test_idx
        betas, cs = pfold_train(train_idx, x, z, y, J, path, fold, glm, use_maf=use_maf, debias=debias, showinfo=showinfo)
        return pfold_validate(test_idx, betas, cs, x, z, y, J, path, fold, glm, use_maf=use_maf, debias=debias, showinfo=showinfo)
    end

    #weight mses for each fold by their size before averaging
    mse = meanloss(mses, num_fold, folds)

    # find best model size and print cross validation result
    k = path[argmin(mse)] :: Int
    print_cv_results(mse, path, k)

    return mse
end

function pfold_train(
    train_idx :: BitArray,
    x         :: SnpArray,
    z         :: AbstractMatrix{T},
    y         :: AbstractVector{T},
    J         :: Int64,
    path      :: DenseVector{Int},
    fold      :: Int64,
    glm       :: String;
    use_maf   :: Bool = false,
    max_iter  :: Int  = 100,
    max_step  :: Int  = 3,
    debias    :: Bool = false,
    showinfo  :: Bool = false,
) where {T <: Float}

    #preallocate arrays
    p, q = size(x, 2), size(z, 2)
    nmodels = length(path)
    betas = zeros(p, nmodels)
    cs = zeros(q, nmodels)

    # allocate training datas
    x_train = SnpArray("x_train_fold$fold.bed", sum(train_idx), p)
    # x_train = SnpArray(undef, sum(train_idx), p)
    copyto!(x_train, @view x[train_idx, :])
    y_train = y[train_idx]
    z_train = z[train_idx, :]
    x_trainbm = SnpBitMatrix{Float64}(x_train, model=ADDITIVE_MODEL, center=true, scale=true); 

    for i in 1:length(path)
        k = path[i]
        if glm == "normal"
            output = L0_normal_reg(x_train, x_trainbm, z_train, y_train, J, k, use_maf=use_maf, debias=debias, show_info=showinfo)
        elseif glm == "logistic"
            output = L0_logistic_reg(x_train, x_trainbm, z_train, y_train, J, k, glm="logistic", show_info=showinfo, debias=debias)
        elseif glm == "poisson"
            output = L0_poisson_reg(x_train, x_trainbm, z_train, y_train, J, k, glm="poisson", show_info=showinfo, debias=debias)
        end
        betas[:, i] .= output.beta
        cs[:, i] .= output.c
    end

    #clean up
    rm("x_train_fold$fold.bed", force=true)   

    return betas, cs
end

function pfold_validate(
    test_idx :: BitArray,
    betas    :: AbstractMatrix{T},
    cs       :: AbstractMatrix{T},
    x        :: SnpArray,
    z        :: AbstractMatrix{T},
    y        :: AbstractVector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    fold     :: Int64,
    glm      :: String;
    use_maf  :: Bool = false,
    max_iter :: Int  = 100,
    max_step :: Int  = 3,
    debias   :: Bool = false,
    showinfo :: Bool = false, 
) where {T <: Float}
    
    # preallocate arrays
    p, q = size(x, 2), size(z, 2)
    test_size = sum(test_idx)
    mse = zeros(T, length(path))
    xb = zeros(T, test_size,)
    zc = zeros(T, test_size,)
    r  = zeros(T, test_size,)
    b  = zeros(T, p,)
    c  = zeros(T, q,)

    # allocate test model
    x_test = SnpArray("x_test_fold$fold.bed", sum(test_idx), p)
    # x_test = SnpArray(undef, sum(test_idx), p)
    copyto!(x_test, @view(x[test_idx, :]))
    y_test = @view(y[test_idx])
    z_test = @view(z[test_idx, :])
    x_testbm = SnpBitMatrix{Float64}(x_test, model=ADDITIVE_MODEL, center=true, scale=true); 

    # for each computed model stored in betas, compute the mean out-of-sample error for the test set
    for i = 1:size(betas,2)

        # pull ith model in dense vector format
        b .= betas[:, i]
        c .= cs[:, i] 

        # compute estimated response Xb: [xb zc] = [x_test z_test] * [b; c]
        A_mul_B!(xb, zc, x_testbm, z_test, b, c) 

        # compute residuals. For glm, apply the inverse link
        if glm == "normal"
            r .= y_test .- xb .- zc
        elseif glm == "logistic"
            r .= y_test .- logistic.(xb .+ zc)
        elseif glm == "poisson"
            r .= y_test .- exp.(xb .+ zc)
        else
            error("unsupported glm method")
        end

        # reduction step. Return out-of-sample error as squared residual averaged over size of test set
        mse[i] = sum(abs2, r) / test_size / 2
    end

    #clean up
    rm("x_test_fold$fold.bed", force=true)

    return mse
end


"""
This function scale mses for each fold by its own fold size.
"""
function meanloss(
    mses     :: Vector{Vector{Float64}},
    num_fold :: Int64,
    folds    :: DenseVector{Int},
)
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