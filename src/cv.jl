"""
COMPUTE ONE FOLD IN A CROSSVALIDATION SCHEME FOR A PENALIZED LINEAR REGRESSION REGULARIZATION PATH

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
PARALLEL CROSSVALIDATION ROUTINE FOR IHT ALGORITHM FOR PENALIZED LEAST SQUARES REGRESSION

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

    # preallocate vectors used in xval
    errors  = zeros(T, num_models)    # vector to save mean squared errors

    # want to compute a path for each fold
    # the folds are computed asynchronously
    # the @sync macro ensures that we wait for all of them to finish before proceeding
    @sync for i = 1:q

        quiet || print_with_color(:blue, "spawning fold $i\n")

        # one_fold returns a vector of out-of-sample errors (MSE for linear regression)
        # @fetch(one_fold(...)) sends calculation to any available processor and returns out-of-sample error
        errors[i] = @fetch(one_fold(x, y, path, folds, i, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet))
    end

    # average the mses
    errors ./= q

    # what is the best model size?
    k = convert(Int, floor(mean(path[errors .== minimum(errors)])))

    # print results
    if !quiet
        println("\n\nCrossvalidation Results:")
        println("k\tMSE")
        @inbounds for i = 1:num_models
            println(path[i], "\t", errors[i])
        end
        println("\nThe lowest MSE is achieved at k = ", k)
    end

    # recompute ideal model
    if refit

        # initialize parameter vector
        b = zeros(T, p)

        # first use L0_reg to extract model
        output = L0_reg(x,y,k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol)
        copy!(b, output["beta"])

        # which components of beta are nonzero?
        bidx = find( x -> x .!= zero(T), b)

        # allocate the submatrix of x corresponding to the inferred model
        x_inferred = x[:,bidx]

        # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
        Xty = BLAS.gemv('T', one(T), x_inferred, y)
        xtx = BLAS.gemm('T', 'N', one(T), x_inferred, x_inferred)
        b2  = xtx \ Xty

        return errors, b2, bidx
    end

    return errors
end
