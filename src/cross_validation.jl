"""
This function computes and stores different models in each column of the matrix `betas` and 
matrix `cs`. 

The additional optional arguments are:
- `mask_n`, a `Bool` vector used as a bitmask for crossvalidation purposes. Defaults to a vector of trues.
"""
function iht_path(
    x         :: SnpArray,
    z         :: AbstractMatrix{T},
    y         :: AbstractVector{T},
    J         :: Int64,
    path      :: DenseVector{Int},
    train_idx :: BitArray;
    use_maf   :: Bool   = false,
    glm       :: String = "normal",
    tol       :: T      = convert(T, 1e-4),
    max_iter  :: Int    = 100,
    max_step  :: Int    = 50,
    debias    :: Bool   = false
) where {T <: Float}

    # size of problem?
    n, p = size(x)
    q = size(z, 2) #number of non-genetic covariates

    # how many models will we compute?
    nmodels = length(path)

    # preallocate matrix to store betas and errors
    betas  = spzeros(T, p, nmodels) # a sparse matrix to store calculated models
    cs     = zeros(T, q, nmodels)   # matrix of models of non-genetic covariates

    # Construct the training datas
    x_train = SnpArray(undef, sum(train_idx), p)
    copyto!(x_train, @view x[train_idx, :])
    y_train = y[train_idx]
    z_train = z[train_idx, :]

    # compute the specified paths
    @inbounds for i = 1:nmodels

        # current model size?
        k = path[i]

        # now compute current model
        if glm == "normal"
            output = L0_normal_reg(x_train, z_train, y_train, J, k, use_maf=use_maf,debias=debias)
        elseif glm == "logistic"
            output = L0_logistic_reg(x_train, z_train, y_train, J, k, glm="logistic",debias=debias)
        elseif glm == "poisson"
            output = L0_poisson_reg(x_train, z_train, y_train, J, k, glm="poisson", show_info=false,debias=debias)
        end

        # put model into sparse matrix of betas
        betas[:, i] .= sparsevec(output.beta)
        cs[:, i] .= output.c
    end

    # return a sparsified copy of the models
    return betas, cs
end

"""
Multi-threaded version of `iht_path`. Each thread writes to a different matrix of betas
and cs, and the reduction step is to sum all these matrices. The increase in memory usage 
increases linearly with the number of paths, which is negligible as long as the number of 
paths is reasonable (e.g. less than 100). 
"""
function iht_path_threaded(
    x         :: SnpArray,
    z         :: AbstractMatrix{T},
    y         :: AbstractVector{T},
    J         :: Int64,
    path      :: DenseVector{Int},
    train_idx :: BitArray;
    use_maf   :: Bool   = false,
    glm       :: String = "normal",    
    tol       :: T      = convert(T, 1e-4),
    max_iter  :: Int    = 100,
    max_step  :: Int    = 3,
    debias    :: Bool   = false
) where {T <: Float}
    
    # number of threads available?
    num_threads = Threads.nthreads()

    # size of problem?
    n, p = size(x)
    q    = size(z, 2) #number of non-genetic covariates

    # how many models will we compute?
    nmodels = length(path)

    # Initialize vector of matrices to hold a separate matrix for each thread to access. This makes everything thread-safe
    betas = [zeros(p, nmodels) for i in 1:num_threads]
    cs    = [zeros(q, nmodels) for i in 1:num_threads]

    # compute the specified paths
    Threads.@threads for i = 1:nmodels

        # current thread?
        cur_thread = Threads.threadid()

        # current model size?
        k = path[i]

        # Construct the training datas (it appears I must make the training data sets inside this for loop. Not sure why. Perhaps to avoid thread access issues)
        x_train = SnpArray(undef, sum(train_idx), p)
        copyto!(x_train, @view x[train_idx, :])
        y_train = y[train_idx]
        z_train = z[train_idx, :]

        # now compute current model
        if glm == "normal"
            output = L0_normal_reg(x_train, z_train, y_train, J, k, use_maf=use_maf, debias=debias)
        elseif glm == "logistic"
            output = L0_logistic_reg(x_train, z_train, y_train, J, k, glm="logistic", show_info=false, debias=debias)
        elseif glm == "poisson"
            output = L0_poisson_reg(x_train, z_train, y_train, J, k, glm="poisson", show_info=false, debias=debias)
        end

        # put model into sparse matrix of betas in the corresponding thread
        betas[cur_thread][:, i] = output.beta
        cs[cur_thread][:, i]    = output.c
    end

    # reduce the vector of matrix into a single matrix, where each column stores a different model 
    return sum(betas), sum(cs)
end

"""
In cross validation we separate samples into `q` disjoint subsets. This function fits a model on 
q-1 of those sets (indexed by train_idx), and then tests the model's performance on the 
qth set (indexed by test_idx). We loop over all sparsity level specified in `path` and 
returns the out-of-sample errors in a vector. 

- `path` , a vector of various model sizes
- `folds`, a vector indicating which of the q fold each sample belongs to. 
- `fold` , the current fold that is being used as test set. 
"""
function one_fold(
    x        :: SnpArray,
    z        :: AbstractMatrix{T},
    y        :: AbstractVector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int}, 
    fold     :: Int;
    use_maf  :: Bool = false,
    glm      :: String = "normal",
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 3,
    debias   :: Bool = false
) where {T <: Float}
    # dimensions of problem
    n, p = size(x)
    q    = size(z, 2)

    # find entries that are for test sets and train sets
    test_idx  = folds .== fold
    train_idx = .!test_idx
    test_size = sum(test_idx)

    # allocate test model
    x_test = SnpArray(undef, sum(test_idx), p)
    copyto!(x_test, @view(x[test_idx, :]))
    z_test = @view(z[test_idx, :])
    x_testbm = SnpBitMatrix{Float64}(x_test, model=ADDITIVE_MODEL, center=true, scale=true); 

    # compute the regularization path on the training set
    betas, cs = iht_path_threaded(x, z, y, J, path, train_idx, use_maf=use_maf, glm=glm, max_iter=max_iter, max_step=max_step, tol=tol, debias=debias)
    # betas, cs = iht_path(x, z, y, J, path, train_idx, use_maf=use_maf, glm = glm, max_iter=max_iter, max_step=max_step, tol=tol, debias=debias)

    # preallocate vector for output
    myerrors = zeros(T, length(path))

    # allocate the arrays for the test set
    xb = zeros(T, test_size,)
    zc = zeros(T, test_size,)
    r  = zeros(T, test_size,)
    b  = zeros(T, p,)
    c  = zeros(T, q,)

    # for each computed model in regularization path, compute the mean out-of-sample error for the TEST set
    for i = 1:size(betas,2)

        # pull ith model in dense vector format
        b .= betas[:, i]
        c .= cs[:, i] 

        # compute estimated response Xb: [xb zc] = [x_test z_test] * [b; c] with $(path[i]) nonzeroes
        A_mul_B!(xb, zc, x_testbm, z_test, b, c) 

        # compute residuals. For glm, recall E(Y) = g^-1(Xβ) where g^-1 is the inverse link
        if glm == "normal"
            r .= view(y, test_idx) .- xb .- zc
        elseif glm == "logistic"
            r .= view(y, test_idx) .- logistic.(xb .+ zc)
        elseif glm == "poisson"
            r .= view(y, test_idx) .- exp.(xb .+ zc)
        else
            error("unsupported glm method")
        end

        # reduction step. Return out-of-sample error as squared residual averaged over size of test set
        myerrors[i] = sum(abs2, r) / test_size / 2
    end

    return myerrors :: Vector{T}
end

# function one_fold_advanced(
#     x        :: SnpArray,
#     z        :: AbstractMatrix{T},
#     y        :: AbstractVector{T},
#     J        :: Int64,
#     path     :: DenseVector{Int},
#     folds    :: DenseVector{Int}, 
#     fold     :: Int;
#     use_maf  :: Bool = false,
#     glm      :: String = "normal",
#     tol      :: T    = convert(T, 1e-4),
#     max_iter :: Int  = 100,
#     max_step :: Int  = 3,
#     debias   :: Bool = false,
#     parallel :: Bool = false
# ) where {T <: Float}
#     # dimensions of problem
#     n, p = size(x)
#     q    = size(z, 2)

#     # find entries that are for test sets and train sets
#     test_idx  = folds .== fold
#     train_idx = .!test_idx
#     test_size = sum(test_idx)

#     fits = (parallel ? pmap : map)(1:fold) do i
#         f = folds .== i
#         holdoutidx = findall(f)
#         modelidx = findall(!, f)
#         g = L0_normal_reg!(X[modelidx, :], y[modelidx], z[modelidx, :];
#                     weights=weights[modelidx], lambda=path.lambda, kw...)
#         loss(g, X[holdoutidx, :], isa(y, AbstractVector) ? y[holdoutidx] : y[holdoutidx, :],
#              weights[holdoutidx])
#     end
# end

"""
Wrapper function for one_fold. Returns the averaged MSE for each fold of cross validation.
mse[i, j] stores the ith model size for fold j. Thus to obtain the mean mse for each fold, 
we take average along the rows and find the minimum.  
"""
function pfold_naive(
    x        :: SnpArray,
    z        :: AbstractMatrix{T},
    y        :: AbstractVector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    num_fold :: Int64;
    use_maf  :: Bool = false,
    glm      :: String = "normal",
    max_iter :: Int  = 100,
    max_step :: Int  = 3,
    debias   :: Bool = false,
) where {T <: Float}

    @assert num_fold >= 1 "number of folds must be positive integer"

    mses = zeros(T, length(path), num_fold)
    for fold in 1:num_fold
        mses[:, fold] = one_fold(x, z, y, J, path, folds, fold, use_maf=use_maf, glm=glm,debias=debias)
    end
    return vec(sum(mses, dims=2) ./ num_fold)
end

"""
This function runs q-fold cross-validation across a specified regularization path in a 
maximum of 8 parallel threads. 

Important arguments and defaults include:
- `x` is the genotype matrix
- `z` is the matrix of non-genetic covariates (which includes the intercept as the first column)
- `y` is the response (phenotype) vector
- `J` is the maximum allowed active groups. 
- `path` is vector of model sizes k1, k2...etc. IHT will compute all model sizes on each fold.
- `folds` a vector of integers, with the same length as the number predictor, indicating a partitioning of the samples into q disjoin subsets
- `num_folds` indicates how many disjoint subsets the samples are partitioned into. 
- `use_maf` whether IHT wants to scale predictors using their minor allele frequency. This is experimental feature
"""
function cv_iht(
    x        :: SnpArray,
    z        :: AbstractMatrix{T},
    y        :: AbstractVector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    num_fold :: Int64;
    use_maf  :: Bool = false,
    glm      :: String = "normal",
    debias   :: Bool = false
) where {T <: Float}

    # how many elements are in the path?
    nmodels = length(path)

    # compute folds
    mses = pfold_naive(x, z, y, J, path, folds, num_fold, use_maf=use_maf, glm=glm, debias=debias)

    # find best model size and print cross validation result
    k = path[argmin(mses)] :: Int
    print_cv_results(mses, path, k)

    return mses
end
