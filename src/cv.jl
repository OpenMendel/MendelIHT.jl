"""
    one_fold(x,y,path,folds,fold) -> Vector{Float}

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
    tol      :: Float = convert(T, 1e-4),
    max_iter :: Int   = 1000,
    max_step :: Int   = 50,
    quiet    :: Bool  = true,
)

    # make vector of indices for folds
    test_idx = folds .== fold
    test_size = sum(test_idx)

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # allocate the arrays for the training set
    x_train   = x[train_idx,:]
    y_train   = y[train_idx]

    # compute the regularization path on the training set
    betas    = iht_path(x_train,y_train,path, tol=tol, max_iter=max_iter, quiet=quiet, max_step=max_step)

    # compute the mean out-of-sample error for the TEST set
    errors = vec(sumabs2(broadcast(-, y[test_idx], x[test_idx,:] * betas), 1)) ./ (2*test_size)

    return errors
end

"""
    pfold(xfile, xtfile, x2file,yfile, meanfile, invstdfile,path,kernfile,folds,q [, pids=procs(), devindices=ones(Int,q])

This function is the parallel execution kernel in `cv_iht()`. It is not meant to be called outside of `cv_iht()`.
It will distribute `q` crossvalidation folds across the processes supplied by the optional argument `pids` and call `one_fold()` for each fold.
Each fold will use the GPU device indexed by its corresponding component of the optional argument `devindices` to compute a regularization path given by `path`.
`pfold()` collects the vectors of MSEs returned by calling `one_fold()` for each process, reduces them, and returns their average across all folds.
"""
function pfold{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    q        :: Int;
    pids     :: DenseVector{Int} = procs(),
    tol      :: Float = convert(T, 1e-4),
    max_iter :: Int   = 100,
    max_step :: Int   = 50,
    quiet    :: Bool  = true,
)
    # how many CPU processes can pfold use?
    np = length(pids)

    # report on CPU processes
    quiet || println("pfold: np = ", np)
    quiet || println("pids = ", pids)

    # set up function to share state (indices of folds)
    i = 1
    nextidx() = (idx=i; i+=1; idx)

    # preallocate cell array for results
    results = cell(q)

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
                        results[current_fold] = remotecall_fetch(worker) do
                                one_fold(x, y, path, folds, i, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet)
                        end # end remotecall_fetch()
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return reduce(+, results[1], results) ./ q
end


"""
    cv_iht(x,y,path,q) -> Vector{Float}

This function will perform `q`-fold cross validation for the ideal model size in IHT least squares regression.
It computes several paths as specified in the `path` argument using the design matrix `x` and the response vector `y`.
Each path is asynchronously spawned using any available processor.
For each path, one fold is held out of the analysis for testing, while the rest of the data are used for training.
The function to compute each path, `one_fold()`, will return a vector of out-of-sample errors (MSEs).

Arguments:

- `x` is the `n` x `p` design matrix.
- `y` is the `n`-vector of responses.
- `path` is an `Int` vector that specifies which model sizes to include in the path, e.g. `path = collect(k0:increment:k_end)`.
- `q` is the number of folds to compute.

Optional Arguments:

- `n` is the number of samples. Defaults to `length(y)`.
- `p` is the number of predictors. Defaults to `size(x,2)`.
- `folds` is the partition of the data. Defaults to `IHT.cv_get_folds(n,q)`.
- `tol` is the convergence tolerance to pass to the path computations. Defaults to `1e-4`.
- `max_iter` caps the number of permissible iterations in the IHT algorithm. Defaults to `100`.
- `max_step` caps the number of permissible backtracking steps. Defaults to `50`.
- `quiet` is a Boolean to activate output. Defaults to `true` (no output).
   *NOTA BENE*: each processor outputs feed to the console without regard to the others,
   so setting `quiet = false` can yield very messy output!
- `refit` is a `Bool` to indicate whether or not to recompute the best model. Defaults to `true` (recompute).

Output:

- `errors` is the averaged MSE over all folds.

If called with `refit = true`, then the output also includes, for best model size `k_star`:

- `b`, a vector of `k_star` floats
- `bidx`, a vector of `k_star` indices indicating the support of the best model.
"""
function cv_iht{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    path     :: DenseVector{Int},
    q        :: Int;
    folds    :: DenseVector{Int} = cv_get_folds(sdata(y),q),
    pids     :: DenseVector{Int} = procs(),
    tol      :: Float            = convert(T, 1e-4),
    n        :: Int              = length(y),
    p        :: Int              = size(x,2),
    max_iter :: Int              = 100,
    max_step :: Int              = 50,
    quiet    :: Bool             = true,
    refit    :: Bool             = true
)
    # how many elements are in the path?
    num_models = length(path)

    # want to compute a path for each fold
    # the folds are computed asynchronously over processes enumerated by pids
    # master process then reduces errors across folds and returns MSEs
    errors = pfold(x, y, path, folds, q, pids=pids, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet) 

    # what is the best model size?
    k = convert(Int, floor(mean(path[errors .== minimum(errors)])))

    # print results
    !quiet && print_cv_results(errors, path, k)

    # recompute ideal model
    if refit

        # first use L0_reg to extract model
        output = L0_reg(x,y,k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol)

        # which components of beta are nonzero?
        bidx = find(output.beta)

        # allocate the submatrix of x corresponding to the inferred model
        x_inferred = x[:,bidx]

        # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
        Xty = BLAS.gemv('T', one(T), x_inferred, y)
        xtx = BLAS.gemm('T', 'N', one(T), x_inferred, x_inferred)
        b  = xtx \ Xty

        return errors, b, bidx
    end

    return errors
end
