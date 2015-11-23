"""
COMPUTE ONE FOLD IN A CROSSVALIDATION SCHEME FOR A REGULARIZATION PATH

    one_fold(x,y,path,folds,fold) -> Vector{Float}

For a regularization path given by the vector `path`,
this function computes an out-of-sample error based on the indices given in `folds`.

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
- `logreg` is a switch to activate logistic regression. Defaults to `false` (perform linear regression).

Output:

- `errors` is a vector of out-of-sample errors (MSEs) for the current fold.
"""
function one_fold(
    x        :: DenseArray{Float64,2},
    y        :: DenseArray{Float64,1},
    path     :: DenseArray{Int,1},
    folds    :: DenseArray{Int,1},
    fold     :: Int;
    tol      :: Float64 = 1e-4,
    max_iter :: Int     = 1000,
    max_step :: Int     = 50,
    quiet    :: Bool    = true,
    logreg   :: Bool    = false
)

    # make vector of indices for folds
    test_idx = folds .== fold
    test_size = sum(test_idx)

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # allocate the arrays for the training set
    x_train   = x[train_idx,:]
    y_train   = y[train_idx]

#    if logreg
#        # compute the regularization path on the training set
#        betas    = iht_path_log(x_train,y_train,path, max_iter=max_iter, quiet=quiet, max_step=max_step)
#
#        # compute the mean out-of-sample error for the TEST set
#        myerrors = vec(sumabs2(broadcast(-, round(y[test_idx]), round(logistic(x[test_idx,:] * betas))), 1)) ./ length(test_idx)
#    else

        # compute the regularization path on the training set
        betas    = iht_path(x_train,y_train,path, tol=tol, max_iter=max_iter, quiet=quiet, max_step=max_step)

        # compute the mean out-of-sample error for the TEST set
        errors = vec(0.5*sumabs2(broadcast(-, y[test_idx], x[test_idx,:] * betas), 1)) ./ test_size
#    end

    return errors
end


"""
PARALLEL CROSSVALIDATION ROUTINE FOR IHT

    cv_iht(x,y,path,nfolds) -> Vector{Float}

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
- `max_iter` caps the number of permissible iterations in the IHT algorithm. Defaults to `1000`.
- `max_step` caps the number of permissible backtracking steps. Defaults to `50`.
- `quiet` is a Boolean to activate output. Defaults to `true` (no output).
   *NOTA BENE*: each processor outputs feed to the console without regard to the others,
   so setting `quiet = true` can yield very messy output!
- `logreg` is a `Bool` to indicate whether or not to perform logistic regression. Defaults to `false` (do linear regression).
- `compute_model` is a `Bool` to indicate whether or not to recompute the best model. Defaults to `true` (recompute).

Output:

- `errors` is the averaged MSE over all folds.

If called with `compute_model = true`, then the output also includes, for best model size `k_star`

- `b`, a vector of `k_star` floats
- `bidx`, a vector of `k_star` indices indicating the support of the best model.
"""
function cv_iht(
    x             :: DenseArray{Float64,2},
    y             :: DenseArray{Float64,1},
    path          :: DenseArray{Int,1},
    q             :: Int;
    folds         :: DenseArray{Int,1} = cv_get_folds(sdata(y),q),
    tol           :: Float64           = 1e-4,
    n             :: Int               = length(y),
    p             :: Int               = size(x,2),
    max_iter      :: Int               = 1000,
    max_step      :: Int               = 50,
    quiet         :: Bool              = true,
    logreg        :: Bool              = false,
    compute_model :: Bool              = true
)

    # how many elements are in the path?
    num_models = length(path)

    # preallocate vectors used in xval
    errors  = zeros(Float64, num_models)    # vector to save mean squared errors

    # want to compute a path for each fold
    # the folds are computed asynchronously
    # the @sync macro ensures that we wait for all of them to finish before proceeding
    @sync for i = 1:q

        quiet || print_with_color(:blue, "spawning fold $i")
        # one_fold returns a vector of out-of-sample errors (MSE for linear regression, MCE for logistic regression)
        # @spawn(one_fold(...)) returns a RemoteRef to the result
        # store that RemoteRef so that we can query the result later
        errors[i] = @fetch(one_fold(x, y, path, folds, i, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, logreg=logreg))
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
    if compute_model

        # initialize parameter vector
        b = zeros(Float64, p)

#        if logreg
#
#            # can preallocate some of the temporary arrays for use in both model selection and fitting
#            # notice that they all depend on n, which is fixed,
#            # as opposed to p, which changes depending on the number of nonzeroes in b
#            xb   = zeros(Float64, n)      # xb = x*b
#            lxb  = zeros(Float64, n)      # logistic(xb), which we call pi
#            l2xb = zeros(Float64, n)      # logistic(xb) [ 1 - logistic(xb) ], or pi(1 - pi)
#
#            # first use L0_reg to extract model
#            output = L0_log(x,y,k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, Xb=xb, Xb0=lxb, r=l2xb)
#            copy!(b, output["beta"])
#
#            # which components of beta are nonzero?
#            bidx = find( x -> x .!= zero(Float64), b)
#
#            # allocate the submatrix of x corresponding to the inferred model
#            x_inferred = x[:,bidx]
#
#            # compute logistic fit
#            b2 = fit_logistic(x_inferred, y, xb=xb, lxb=lxb, l2xb=l2xb)
#        else

            # first use L0_reg to extract model
            output = L0_reg(x,y,k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol)
            copy!(b, output["beta"])

            # which components of beta are nonzero?
            bidx = find( x -> x .!= zero(Float64), b)

            # allocate the submatrix of x corresponding to the inferred model
            x_inferred = x[:,bidx]

            # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
            xty = BLAS.gemv('T', one(Float64), x_inferred, y)
            xtx = BLAS.gemm('T', 'N', one(Float64), x_inferred, x_inferred)
            b2  = xtx \ xty
#        end
        return errors, b2, bidx
    end
    return errors
end
