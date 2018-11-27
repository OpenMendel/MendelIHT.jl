"""
This function computes and stores different models in each column of the matrix `betas` and 
matrix `cs`. 

The additional optional arguments are:
- `mask_n`, a `Bool` vector used as a bitmask for crossvalidation purposes. Defaults to a vector of trues.
"""
function iht_path(
    x        :: SnpLike{2},
    z        :: Matrix{T},
    y        :: Vector{T},
    J        :: Int64,
    path     :: DenseVector{Int};
    use_maf  :: Bool = false,
    #pids     :: Vector{Int} = procs(x),
    mask_n   :: BitArray = trues(size(y)),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    #quiet    :: Bool = true
) where {T <: Float}

    # size of problem?
    n, p = size(x)
    q = size(z, 2) #number of non-genetic covariates

    # how many models will we compute?
    nmodels = length(path)

    # also preallocate matrix to store betas and errors
    betas  = spzeros(T, p, nmodels) # a sparse matrix to store calculated models
    cs     = zeros(T, q, nmodels)   # matrix of models of non-genetic covariates

    # compute the specified paths
    @inbounds for i = 1:nmodels

        # current model size?
        k = path[i]

        #define the IHTVariable used for cleaner code
        v = IHTVariables(x, z, y, J, k)

        # now compute current model
        output = L0_reg(v, x, z, y, J, k, use_maf=use_maf, mask_n=mask_n)

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
    x        :: SnpLike{2},
    z        :: Matrix{T},
    y        :: Vector{T},
    J        :: Int64,
    path     :: DenseVector{Int};
    use_maf  :: Bool = false,
    #pids     :: Vector{Int} = procs(x),
    mask_n   :: BitArray = trues(size(y)),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    #quiet    :: Bool = true

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
    @inbounds Threads.@threads for i = 1:nmodels

        # current thread?
        cur_thread = Threads.threadid()

        # current model size?
        k = path[i]

        #define the IHTVariable used for cleaner code #TODO: should declare this only 1 time for max efficiency. 
        v = IHTVariables(x, z, y, J, k)

        # now compute current model
        output = L0_reg(v, x, z, y, J, k, use_maf=use_maf, mask_n=mask_n)

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
- `pids` , a vector of process IDs. Defaults to `procs(x)`.
"""
function one_fold(
    x        :: SnpLike{2},
    z        :: Matrix{T},
    y        :: Vector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int}, 
    fold     :: Int;
    use_maf  :: Bool = false,
    #pids     :: Vector{Int} = procs(x),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    # quiet    :: Bool = true
) where {T <: Float}
    # dimensions of problem
    n, p = size(x)
    q    = size(z, 2)

    # find entries that are for test sets and train sets
    test_idx  = folds .== fold
    train_idx = .!test_idx
    test_size = sum(test_idx)

    # compute the regularization path on the training set
    betas, cs = iht_path_threaded(x, z, y, J, path, use_maf=use_maf, mask_n=train_idx, max_iter=max_iter, max_step=max_step, tol=tol)
    # betas, cs = iht_path(x, z, y, J, path, use_maf=use_maf, mask_n=train_idx, max_iter=max_iter, max_step=max_step, tol=tol)

    # preallocate vector for output
    myerrors = zeros(T, length(path))

    # allocate the arrays for the test set
    xb = zeros(test_size,)
    zc = zeros(test_size,)
    r  = zeros(test_size,)
    b  = zeros(p,)
    c  = zeros(q,)

    # compute the mean out-of-sample error for the TEST set
    # do this for every computed model in regularization path
    for i = 1:size(betas,2)

        # pull ith model in dense vector format
        b .= betas[:, i]
        c .= cs[:, i] 

        # allocate test model, this can be avoided with view(x, test_idx, :), but SnpArray code needs to gets fixed first 
        x_test = x[test_idx, :]
        z_test = z[test_idx, :]

        # compute some statistics needed to standardize the snpmatrix
        mean_vec, minor_allele, = summarize(x_test)
        people, snps = size(x)
        update_mean!(mean_vec, minor_allele, snps)
        std_vec = std_reciprocal(x, mean_vec)

        # compute estimated response Xb with $(path[i]) nonzeroes
        SnpArrays.A_mul_B!(xb, x[test_idx, :], b, mean_vec, std_vec) 
        BLAS.A_mul_B!(zc, z[test_idx, :], c)

        # compute residuals
        r .= view(y, test_idx) .- xb .- zc

        # reduction step. Return out-of-sample error as squared residual averaged over size of test set
        myerrors[i] = sum(abs2, r) / test_size / 2
    end

    return myerrors :: Vector{T}
end

"""
Wrapper function for one_fold. Returns the averaged MSE for each fold of cross validation.
mse[i, j] stores the ith model size for fold j. Thus to obtain the mean mse for each fold, 
we take average along the rows and find the minimum.  
"""
function pfold_naive(
    x        :: SnpLike{2},
    z        :: Matrix{T},
    y        :: Vector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    num_fold :: Int64;
    use_maf  :: Bool = false,
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
) where {T <: Float}

    @assert num_fold >= 1 "number of folds must be positive integer"

    mses = zeros(length(path), num_fold)
    for fold in 1:num_fold
        mses[:, fold] = one_fold(x, z, y, J, path, folds, fold, use_maf=use_maf)
    end
    return vec(sum(mses, 2) ./ num_fold)
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
    x        :: SnpLike{2},
    z        :: Matrix{T},
    y        :: Vector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    num_fold :: Int64;
    use_maf  :: Bool = false,
    # pids     :: Vector{Int} = procs(),
    # tol      :: Float = convert(T, 1e-4),
    # max_iter :: Int   = 100,
    # max_step :: Int   = 50,
    # quiet    :: Bool  = true,
    # header   :: Bool  = false
) where {T <: Float}

    # how many elements are in the path?
    nmodels = length(path)

    # compute folds
    mses = pfold_naive(x, z, y, J, path, folds, num_fold, use_maf=use_maf)

    # find best model size and print cross validation result
    k = path[indmin(mses)] :: Int
    print_cv_results(mses, path, k)

    return nothing
end
