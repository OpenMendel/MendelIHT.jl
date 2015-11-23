function one_fold(
    x        :: DenseArray{Float32,2},
    y        :: DenseArray{Float32,1},
    path     :: DenseArray{Int,1},
    folds    :: DenseArray{Int,1},
    fold     :: Int;
    tol      :: Float32 = 1e-4,
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
        errors = vec(0.5f0*sumabs2(broadcast(-, y[test_idx], x[test_idx,:] * betas), 1)) ./ test_size
#    end

    return errors
end


function cv_iht(
    x             :: DenseArray{Float32,2},
    y             :: DenseArray{Float32,1},
    path          :: DenseArray{Int,1},
    q             :: Int;
    folds         :: DenseArray{Int,1} = cv_get_folds(sdata(y),q),
    tol           :: Float32           = 1e-4,
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
    errors  = zeros(Float32, num_models)    # vector to save mean squared errors

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
        b = zeros(Float32, p)

#        if logreg
#
#            # can preallocate some of the temporary arrays for use in both model selection and fitting
#            # notice that they all depend on n, which is fixed,
#            # as opposed to p, which changes depending on the number of nonzeroes in b
#            xb   = zeros(Float32, n)      # xb = x*b
#            lxb  = zeros(Float32, n)      # logistic(xb), which we call pi
#            l2xb = zeros(Float32, n)      # logistic(xb) [ 1 - logistic(xb) ], or pi(1 - pi)
#
#            # first use L0_reg to extract model
#            output = L0_log(x,y,k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, Xb=xb, Xb0=lxb, r=l2xb)
#            copy!(b, output["beta"])
#
#            # which components of beta are nonzero?
#            bidx = find( x -> x .!= zero(Float32), b)
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
            bidx = find( x -> x .!= zero(Float32), b)

            # allocate the submatrix of x corresponding to the inferred model
            x_inferred = x[:,bidx]

            # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
            xty = BLAS.gemv('T', one(Float32), x_inferred, y)
            xtx = BLAS.gemm('T', 'N', one(Float32), x_inferred, x_inferred)
            b2  = xtx \ xty
#        end
        return errors, b2, bidx
    end
    return errors
end
