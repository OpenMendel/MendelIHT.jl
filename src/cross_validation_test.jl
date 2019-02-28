function iht_path_test(
    x_train   :: SnpArray,
    z_train   :: AbstractMatrix{T},
    y_train   :: AbstractVector{T},
    J         :: Int64,
    path      :: DenseVector{Int};
    use_maf   :: Bool   = false,
    glm       :: String = "normal",    
    tol       :: T      = convert(T, 1e-4),
    max_iter  :: Int    = 100,
    max_step  :: Int    = 3,
    debias    :: Bool   = false,
    showinfo  :: Bool   = false,
) where {T <: Float}
    
    # number of threads available?
    num_threads = Threads.nthreads()

    # size of problem?
    n, p = size(x_train)
    q    = size(z_train, 2) #number of non-genetic covariates

    # how many models will we compute?
    nmodels = length(path)

    # Initialize vector of matrices to hold a separate matrix for each thread to access. This makes everything thread-safe
    betas = [zeros(p, nmodels) for i in 1:num_threads]
    cs    = [zeros(q, nmodels) for i in 1:num_threads]

    # compute the specified paths
    for i = 1:nmodels

        # current thread?
        cur_thread = Threads.threadid()

        # current model size?
        k = path[i]

        # now compute current model
        if glm == "normal"
            x_train_copy = deepcopy(x_train)
            output = L0_normal_reg(x_train, z_train, y_train, J, k, use_maf=use_maf, debias=debias, show_info=showinfo)
            println(all(x_train_copy.columncounts .== x_train.columncounts))
            println(all(x_train_copy.data .== x_train.data))
            println(all(x_train_copy.rowcounts .== x_train.rowcounts))
            println(x_train_copy.m == x_train.m)
        elseif glm == "logistic"
            output = L0_logistic_reg(x_train, z_train, y_train, J, k, glm="logistic", show_info=showinfo, debias=debias)
        elseif glm == "poisson"
            output = L0_poisson_reg(x_train, z_train, y_train, J, k, glm="poisson", show_info=showinfo, debias=debias)
        end

        # if any(isnan.(output.beta))
        #     nz_entry = findall(!iszero, output.beta)
        #     println(output.beta[nz_entry])
        #     println("wtf some b is nan")
        # end

        # put model into sparse matrix of betas in the corresponding thread
        betas[cur_thread][:, i] .= output.beta
        cs[cur_thread][:, i]    .= output.c
    end

    # for i in 1:num_threads
    #     if any(isnan.(betas[i]))
    #         println("some b in thread $i is nan.... how???")
    #     end
    # end

    # reduce the vector of matrix into a single matrix, where each column stores a different model 
    return sum(betas), sum(cs)
end

function iht_path_threaded_test(
    x_train   :: SnpArray,
    z_train   :: AbstractMatrix{T},
    y_train   :: AbstractVector{T},
    J         :: Int64,
    path      :: DenseVector{Int};
    use_maf   :: Bool   = false,
    glm       :: String = "normal",    
    tol       :: T      = convert(T, 1e-4),
    max_iter  :: Int    = 100,
    max_step  :: Int    = 3,
    debias    :: Bool   = false,
    showinfo  :: Bool   = false,
) where {T <: Float}

    # number of threads available?
    num_threads = Threads.nthreads()

    # size of problem?
    n, p = size(x_train)
    q    = size(z_train, 2) #number of non-genetic covariates

    # how many models will we compute?
    nmodels = length(path)

    # Initialize vector of matrices to hold a separate matrix for each thread to access. This makes everything thread-safe
    betas = [zeros(p, nmodels) for i in 1:num_threads]
    cs    = [zeros(q, nmodels) for i in 1:num_threads]

    mutex = Threads.Mutex()

    # compute the specified paths
    Threads.@threads for i = 1:nmodels

        # current thread?
        cur_thread = Threads.threadid()

        # current model size?
        k = path[i]

        # now compute current model
        if glm == "normal"
            output = L0_normal_reg(x_train, z_train, y_train, J, k, use_maf=use_maf, debias=debias, show_info=showinfo)
        elseif glm == "logistic"
            output = L0_logistic_reg(x_train, z_train, y_train, J, k, glm="logistic", show_info=showinfo, debias=debias)
        elseif glm == "poisson"
            output = L0_poisson_reg(x_train, z_train, y_train, J, k, glm="poisson", show_info=showinfo, debias=debias)
        end

        # if any(isnan.(output.beta))
        #     nz_entry = findall(!iszero, output.beta)
        #     println(output.beta[nz_entry])
        #     println("wtf some b is nan")
        # end

        # put model into sparse matrix of betas in the corresponding thread
        lock(mutex)
        betas[cur_thread][:, i] .= output.beta
        cs[cur_thread][:, i]    .= output.c
        unlock(mutex)
    end

    for i in 1:num_threads
        if any(isnan.(betas[i]))
            println("some b in thread $i is nan.... how???")
        end
    end

    # reduce the vector of matrix into a single matrix, where each column stores a different model 
    return sum(betas), sum(cs)
end

function one_fold_test(
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
    debias   :: Bool = false,
    showinfo :: Bool = false,
) where {T <: Float}
    # dimensions of problem
    n, p = size(x)
    q    = size(z, 2)

    # find entries that are for test sets and train sets
    test_idx  = folds .== fold
    train_idx = .!test_idx
    test_size = sum(test_idx)

    # allocate test model
    x_test = SnpArray("x_test_fold$fold.bed", sum(test_idx), p)
    copyto!(x_test, @view(x[test_idx, :]))
    z_test = @view(z[test_idx, :])
    x_testbm = SnpBitMatrix{Float64}(x_test, model=ADDITIVE_MODEL, center=true, scale=true); 

    # allocate training datas
    x_train = SnpArray("x_train_fold$fold.bed", sum(train_idx), p)
    copyto!(x_train, @view x[train_idx, :])
    y_train = y[train_idx]
    z_train = z[train_idx, :]

    # compute the regularization path on the training set
    # betas, cs = iht_path_threaded_test(x_train, z_train, y_train, J, path, use_maf=use_maf, glm=glm, max_iter=max_iter, max_step=max_step, tol=tol, debias=debias,showinfo=showinfo)
    betas, cs = iht_path_test(x_train, z_train, y_train, J, path, use_maf=use_maf, glm=glm, max_iter=max_iter, max_step=max_step, tol=tol, debias=debias,showinfo=showinfo)

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

    #clean up
    rm("x_test_fold$fold.bed", force=true)
    rm("x_train_fold$fold.bed", force=true)    

    return myerrors :: Vector{T}
end

function pfold_naive_test(
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
    showinfo :: Bool = false,
) where {T <: Float}

    @assert num_fold >= 1 "number of folds must be positive integer"

    mses = zeros(T, length(path), num_fold)
    for fold in 1:num_fold
        mses[:, fold] = one_fold_test(x, z, y, J, path, folds, fold, use_maf=use_maf, glm=glm,debias=debias,showinfo=showinfo)
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
function cv_iht_test(
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
    showinfo :: Bool = false
) where {T <: Float}

    # how many elements are in the path?
    nmodels = length(path)

    # compute folds
    # mses = pfold_naive_test(x, z, y, J, path, folds, num_fold, use_maf=use_maf, glm=glm, debias=debias, showinfo=showinfo)
    mses = pfold_test2(x, z, y, J, path, folds, num_fold, use_maf=use_maf, glm=glm, debias=debias, showinfo=showinfo)

    # find best model size and print cross validation result
    k = path[argmin(mses)] :: Int
    print_cv_results(mses, path, k)

    return mses
end
