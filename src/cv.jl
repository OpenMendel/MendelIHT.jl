"""
    one_fold(x,y,path,folds,fold) -> Vector

For a regularization path given by the `Int` vector `path`,
this function performs penalized linear regression on `x` and `y` and computes an out-of-sample error based on the indices given in `folds`.

Arguments:

- `x` is the `n` x `p` design matrix.
- `y` is the `n`-vector of responses.
- `path` is an `Int` vector that specifies which model sizes to include in the path, e.g. `path = collect(k0:increment:k_end)`.
- `folds` is an `Int` vector indicating which component of `y` goes to which fold, e.g. `folds = IHT.cv_get_folds(n,nfolds)`
- `fold` is the current fold to compute.

Optional Arguments:

- `tol` is the convergence tolerance to pass to the path computations. Defaults to `1e-4`.
- `max_iter` caps the number of permissible iterations in the IHT algorithm. Defaults to `1000`.
- `max_step` caps the number of permissible backtracking steps. Defaults to `50`.
- `quiet` is a Boolean to activate output. Defaults to `true` (no output).

Output:

- `errors` is a vector of out-of-sample errors (MSEs) for the current fold.
"""
function one_fold{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    fold     :: Int;
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true,
)

    # make vector of indices for folds
    test_idx  = folds .== fold
    test_size = sum(test_idx)

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # allocate the arrays for the training set
    x_train = x[train_idx,:]
    y_train = y[train_idx]

    # compute the regularization path on the training set
    betas = iht_path(x_train, y_train, path, tol=tol, max_iter=max_iter, quiet=quiet, max_step=max_step)

    # compute the mean out-of-sample error for the TEST set
    Xb = view(x, test_idx, :) * betas
#    r  = broadcast(-, view(y, test_idx, 1), Xb)
    r  = broadcast(-, y[test_idx], Xb)
    er = vec(sumabs2(r, 1)) ./ (2*test_size)

    return er :: Vector{T}
end

"""
    pfold(x,y,path,folds,q [, pids=procs()]) -> Vector

This function is the parallel execution kernel in `cv_iht()`. It is not meant to be called outside of `cv_iht()`.
For floating point data `x` and `y` and an integer vector `path`, `pfold` will distribute `q` crossvalidation folds across the processes supplied by the optional argument `pids`.
It calls `one_fold()` for each fold, then collects the vectors of MSEs each process, applys a reduction, and finally returns the average MSE vector across all folds.
"""
function pfold{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    q        :: Int;
    pids     :: Vector{Int} = procs(),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true,
)
    # how many CPU processes can pfold use?
    np = length(pids)

    # report on CPU processes
    quiet || println("pfold: np = ", np)
    quiet || println("pids = ", pids)

    # set up function to share state (indices of folds)
    i = 1
    nextidx() = (idx=i; i+=1; idx)

<<<<<<< HEAD
    # preallocate cell array for results
#    results = zeros(T, length(path), q)
    results = SharedArray(T, (length(path),q), pids=pids)
=======
    # preallocate array for results
    results = SharedArray(T, (length(path),q), pids=pids) :: SharedMatrix{T}
>>>>>>> 962306819f141138c4221a03ce6a6b012896323b

    # master process will distribute tasks to workers
    # master synchronizes results at end before returning
    @sync begin

        # loop over all workers
        for worker in pids

            # exclude process that launched pfold, unless only one process is available
            if worker != myid() || np == 1

                # asynchronously distribute tasks
                @async begin
                    while true

                        # grab next fold
                        current_fold = nextidx()

                        # if current fold exceeds total number of folds then exit loop
                        current_fold > q && break

                        # report distribution of fold to worker and device
                        quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker.\n\n")

                        # launch job on worker
                        # worker loads data from file paths and then computes the errors in one fold
#                        results[:, current_fold] = remotecall_fetch(worker) do
                        r = remotecall_fetch(worker) do
                                one_fold(x, y, path, folds, current_fold, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet)
                        end # end remotecall_fetch()
                        setindex!(results, r, :, current_fold)
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return (vec(sum(results, 2) ./ q)) :: Vector{T}
end


"""
    cv_iht(x,y) -> Vector

This function will perform `q`-fold cross validation for the ideal model size in IHT least squares regression using the `n` x `p` design matrix `x` and the response vector `y`.
Each path is asynchronously spawned using any available processor.
For each path, one fold is held out of the analysis for testing, while the rest of the data are used for training.
The function to compute each path, `one_fold()`, will return a vector of out-of-sample errors (MSEs).

Arguments:

- `x` is the `n` x `p` design matrix.
- `y` is the `n`-vector of responses.

Optional Arguments:

- `q` is the number of folds to compute. Defaults to `max(3, min(CPU_CORES, 5))`, where `CPU_CORES`is the Julia variable to query the number of available CPU cores.
- `path` is an `Int` vector that specifies which model sizes to include in the path. Defaults to `path = collect(1:min(p,20))`.
- `folds` is the partition of the data. Defaults to `IHT.cv_get_folds(n,q)`.
- `pids`, a vector of process IDs. Defaults to `procs()`, which recruits all available processes.
- `tol` is the convergence tolerance to pass to the path computations. Defaults to `1e-4`.
- `max_iter` caps the number of permissible iterations in the IHT algorithm. Defaults to `100`.
- `max_step` caps the number of permissible backtracking steps. Defaults to `50`.
- `quiet` is a Boolean to activate output. Defaults to `true` (no output).
   *NOTA BENE*: each processor outputs feed to the console without regard to the others,
   so setting `quiet = false` can yield very messy output!

Output:

An `IHTCrossvalidationResults` object with the following fields:

- `mses` is the averaged MSE over all folds.
- `k` is the best crossvalidated model size.
- `path` is the regularization path used in the crossvalidation.
- `b`, a vector of `k_star` floats
- `bidx`, a vector of `k_star` indices indicating the support of the best model.
"""
function cv_iht{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T};
    q        :: Int   = cv_get_num_folds(3, 5),
    path     :: DenseVector{Int} = collect(1:min(size(x,2),20)),
    folds    :: DenseVector{Int} = cv_get_folds(sdata(y),q),
    pids     :: Vector{Int} = procs(),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true,
)
    # do not allow crossvalidation with fewer than 3 folds
    q > 2 || throw(ArgumentError("Number of folds q = $q must be at least 3."))

    # problem dimensions?
    n,p = size(x)

    # check for conformable arrays
    n == length(y) || throw(DimensionMismatch("Row dimension of x ($n) must match number of rows in y ($(length(y)))"))

    # how many elements are in the path?
    nmodels = length(path)

    # want to compute a path for each fold
    # the folds are computed asynchronously over processes enumerated by pids
    # master process then reduces errors across folds and returns MSEs
    mses = pfold(x, y, path, folds, q, pids=pids, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet)

    # what is the best model size?
    # if there are multiple model sizes of EXACTLY the same MSE,
    # then this chooses the smaller of the two
    k = path[indmin(errors)] :: Int

    # print results
    !quiet && print_cv_results(mses, path, k)

    # refit the best model
    b, bidx = refit_iht(x, y, k, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet)

<<<<<<< HEAD
    # which components of beta are nonzero?
    bidx = find(output.beta) :: Vector{Int}

    # allocate the submatrix of x corresponding to the inferred model
    x_inferred = x[:,bidx]

    # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
    xty = BLAS.gemv('T', one(T), x_inferred, y) :: Vector{T}
    xtx = BLAS.gemm('T', 'N', one(T), x_inferred, x_inferred) :: Matrix{T}
    b   = zeros(T, length(bidx))
    try
        b = (xtx \ xty) :: Vector{T}
    catch e
        warn("in refit, caught error: ", e, "\nSetting returned values of b to -Inf")
        fill!(b, -Inf)
    end

=======
>>>>>>> 962306819f141138c4221a03ce6a6b012896323b
    return IHTCrossvalidationResults(mses, sdata(path), b, bidx, k)
end
