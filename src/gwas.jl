function update_r_grad!{T}(
    v    :: IHTVariables{T},
    x    :: BEDFile{T},
    y    :: DenseVector{T};
    pids :: DenseVector{Int} = procs()
)
    difference!(v.r, y, v.xb)
    PLINK.At_mul_B!(v.df, x, v.r, pids=pids)
    return nothing
end

"""
    iht!(b, x::BEDFile, y, k, g)

If used with a `BEDFile` object `x`, then the temporary arrays `b0`, `Xb`, `Xb0`, and `sortidx` are all initialized as `SharedArray`s of the proper dimensions.
The additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs()`.
"""
function iht!{T <: Float}(
    v        :: IHTVariables{T}, 
    x        :: BEDFile{T},
    y        :: DenseVector{T},
    k        :: Int;
    pids     :: DenseVector{Int} = procs(x),
    iter     :: Int = 1,
    max_step :: Int = 50,
)
    # compute indices of nonzeroes in beta
    _iht_indices(v, k)

    # if support has not changed between iterations,
    # then xk and gk are the same as well
    # avoid extracting and computing them if they have not changed
    # one exception: we should always extract columns on first iteration
    if !isequal(v.idx, v.idx0) || iter < 2
        decompress_genotypes!(v.xk, x, v.idx) 
    end

    # store relevant components of gradient
    fill_perm!(v.gk, v.df, v.idx)  # gk = g[v.idx]

    # now compute subset of x*g
    BLAS.gemv!('N', one(T), v.xk, v.gk, zero(T), v.xgk)

    # warn if xgk only contains zeros
    all(v.xgk .== zero(T)) && warn("Entire active set has values equal to 0")

    # compute step size
    mu = sumabs2(v.gk) / sumabs2(v.xgk)

    # notify problems with step size
    isfinite(mu) || throw(error("Step size is not finite, is active set all zero?"))
    mu <= eps(typeof(mu))  && warn("Step size $(mu) is below machine precision, algorithm may not converge correctly")

    # compute gradient step
    _iht_gradstep(v, mu, k)

    # update xb
    PLINK.A_mul_B!(v.xb, x, v.b, v.idx, k, pids=pids)

    # calculate omega
#    omega_top = sqeuclidean(sdata(v.b), v.b0)
#    omega_bot = sqeuclidean(sdata(v.xb), v.xb0)
    omega_top, omega_bot = _iht_omega(v)

    # backtrack until mu sits below omega and support stabilizes
    mu_step = 0
    while mu*omega_bot > 0.99*omega_top && sum(v.idx) != 0 && sum(v.idx $ v.idx0) != 0 && mu_step < max_step

        # stephalving
        mu /= 2

        # warn if mu falls below machine epsilon
        mu <= eps(typeof(mu)) && warn("Step size equals zero, algorithm may not converge correctly")

        # recompute gradient step
        copy!(v.b, v.b0)
#        _iht_gradstep(v, mu, k)
        BLAS.axpy!(mu, sdata(v.df), sdata(v.b))

        # recompute projection onto top k components of b
        project_k!(v.b, k)

        # which indices of new beta are nonzero?
        update_indices!(v.idx, v.b)

        # must correct for equal entries at kth pivot of b
        # this is a total hack! but matching magnitudes are very rare
        # should not drastically affect performance, esp. with big data
        # hack randomly permutes indices of duplicates and retains one 
        if sum(v.idx) > k 
            a = select(v.b, k, by=abs, rev=true)          # compute kth pivot
            duples = find(x -> abs(x) .== abs(a), v.b)    # find duplicates
            c = randperm(length(duples))                # shuffle 
            d = duples[c[2:end]]                        # permute, clipping top 
            v.b[d] = zero(T)                             # zero out duplicates
            v.idx[d] = false                              # set corresponding indices to false
        end 

        # recompute xb
        PLINK.A_mul_B!(v.xb, x, v.b, v.idx, k, pids=pids)

        # calculate omega
#        omega_top = sqeuclidean(sdata(v.b),sdata(v.b0))
#        omega_bot = sqeuclidean(sdata(v.xb),sdata(v.xb0))
        omega_top, omega_bot = _iht_omega(v)

        # increment the counter
        mu_step += 1
    end

    return mu, mu_step
end

"""
    L0_reg(x::BEDFile, y, k)

If used with a `BEDFile` object `x`, then the temporary floating point arrays are all initialized as `SharedArray`s of the proper dimensions.
The additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs()`.
"""
function L0_reg{T <: Float}(
    x        :: BEDFile{T},
    y        :: DenseVector{T},
    k        :: Int;
    pids     :: DenseVector{Int} = procs(),
    temp     :: IHTVariables{T}  = IHTVariables(x, y, k),
    tol      :: Float            = convert(T, 1e-4),
    max_iter :: Int              = 100,
    max_step :: Int              = 50,
    quiet    :: Bool             = true
)

    # start timer
    tic()

    # first handle errors
    k        >= 0      || throw(ArgumentError("Value of k must be nonnegative!\n"))
    max_iter >= 0      || throw(ArgumentError("Value of max_iter must be nonnegative!\n"))
    max_step >= 0      || throw(ArgumentError("Value of max_step must be nonnegative!\n"))
    tol      >  eps(T) || throw(ArgumentError("Value of global tol must exceed machine precision!\n"))

    # initialize return values
    mm_iter   = 0                 # number of iterations of L0_reg
    mm_time   = zero(T)           # compute time *within* L0_reg
    next_loss = oftype(tol,Inf)   # loss function value

    # initialize floats
    loss = oftype(tol,Inf) # tracks previous objective function value
    the_norm    = zero(T)         # norm(b - b0)
    scaled_norm = zero(T)         # the_norm / (norm(b0) + 1)
    mu          = zero(T)         # Landweber step size, 0 < tau < 2/rho_max^2

    # initialize integers
    i       = 0                   # used for iterations in loops
    mu_step = 0                   # counts number of backtracking steps for mu

    # initialize booleans
    converged = false             # scaled_norm < tol?

    # update Xb, r, and gradient
    if sum(temp.idx) == 0
        fill!(temp.xb,zero(T))
        copy!(temp.r,sdata(y))
        At_mul_B!(temp.df, x, temp.r, pids=pids)
    else
        A_mul_B!(temp.xb,x,temp.b,temp.idx,k, pids=pids)
        update_r_grad!(temp, x, y, pids=pids)
    end

    # formatted output to monitor algorithm progress
    !quiet && print_header()

    # main loop
    for mm_iter = 1:max_iter

        # notify and break if maximum iterations are reached.
        if mm_iter >= max_iter

            # alert about hitting maximum iterations
            !quiet && print_maxiter(max_iter, loss)

            # send elements below tol to zero
            threshold!(temp.b, tol)

            # stop timer
            mm_time = toq()

            # these are output variables for function
            # wrap them into a Dict and return
            return IHTResults(mm_time, next_loss, mm_iter, copy(temp.b))
        end

        # save values from previous iterate
        copy!(temp.b0, temp.b)             # b0 = b
        copy!(temp.xb0, temp.xb)           # Xb0 = Xb
        loss = next_loss

        # now perform IHT step
        (mu, mu_step) = iht!(temp, x, y, k, max_step=max_step, iter=mm_iter, pids=pids)

        # the IHT kernel gives us an updated x*b
        # use it to recompute residuals and gradient
        update_r_grad!(temp, x, y, pids=pids)

        # update loss, objective, and gradient
        next_loss = sumabs2(sdata(temp.r)) / 2

        # guard against numerical instabilities
        # ensure that objective is finite
        # if not, throw error
#        isnan(next_loss) && throw(error("Objective function is NaN, aborting..."))
#        isinf(next_loss) && throw(error("Objective function is Inf32, aborting..."))
        check_finiteness(next_loss)

        # track convergence
        the_norm    = chebyshev(temp.b, temp.b0)
        scaled_norm = the_norm / ( norm(temp.b0,Inf) + 1)
        converged   = scaled_norm < tol

        # output algorithm progress
        quiet || @printf("%d\t%d\t%3.7f\t%3.7f\t%3.7f\n", mm_iter, mu_step, mu, the_norm, next_loss)

        # check for convergence
        # if converged and in feasible set, then algorithm converged before maximum iteration
        # perform final computations and output return variables
        if converged

            # send elements below tol to zero
            threshold!(temp.b, tol)

            # stop time
            mm_time = toq()

            # announce convergence 
            !quiet && print_convergence(mm_iter, next_loss, mm_time)

            # these are output variables for function
            return IHTResults(mm_time, next_loss, mm_iter, copy(temp.b))
        end

        # algorithm is unconverged at this point.
        # if algorithm is in feasible set, then rho should not be changing
        # check descent property in that case
        # if rho is not changing but objective increases, then abort
        if next_loss > loss + tol
            !quiet && print_descent_error(mm_iter, loss, next_loss)
            throw(ErrorException("Descent failure!"))
        end
    end # end main loop
end # end function



"""
    iht_path(x::BEDFile, y, path)

If used with a `BEDFile` object `x`, then the temporary arrays are all initialized as `SharedArray`s of the proper dimensions.
The additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs()`.
- `means`, a vector of SNP means. Defaults to `mean(T, x, shared=true, pids=procs()`.
- `invstds`, a vector of SNP precisions. Defaults to `invstd(x, means, shared=true, pids=procs()`.
"""
function iht_path{T <: Float}(
    x        :: BEDFile{T},
    y        :: DenseVector{T},
    path     :: DenseVector{Int};
    pids     :: DenseVector{Int} = procs(x),
    tol      :: Float            = convert(T, 1e-4),
    max_iter :: Int              = 100,
    max_step :: Int              = 50,
    quiet    :: Bool             = true
)

    # size of problem?
    n = length(y)
    p = size(x,2)
#    b = zeros(T, p)

    # how many models will we compute?
    num_models = length(path)

    # also preallocate matrix to store betas
    betas = spzeros(T,p,num_models)  # a matrix to store calculated models

    # preallocate temporary arrays
    temp = IHTVariables(x, y, 1)

    # compute the path
    @inbounds for i = 1:num_models

        # model size?
        q = path[i]

        # monitor progress
        quiet || print_with_color(:blue, "Computing model size $q.\n\n")

        # these arrays change in size from iteration to iteration
        # we must allocate them for every new model size
        update_variables!(temp, x, q)

        # store projection of beta onto largest k nonzeroes in magnitude
        project_k!(temp.b, q)

        # now compute current model
        output = L0_reg(x, y, q, temp=temp, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids)

        # extract and save model
#        copy!(b, output.beta)

        # ensure that we correctly index the nonzeroes in b
#        update_indices!(temp.idx, b)
        update_indices!(temp.idx, output.beta)
        fill!(temp.idx0, false)

        # put model into sparse matrix of betas
#        betas[:,i] = sparsevec(b)
        betas[:,i] = sparsevec(output.beta)
    end

    # return a sparsified copy of the models
    return betas
end


"""
    one_fold(x::BEDFile, y, path, folds, fold)

If used with a `BEDFile` object `x`, then the additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs()`.
- `means`, a vector of SNP means. Defaults to `mean(T, x, shared=true, pids=procs()`.
- `invstds`, a vector of SNP precisions. Defaults to `invstd(x, means, shared=true, pids=procs()`.
"""
function one_fold{T <: Float}(
    x        :: BEDFile,
    y        :: DenseVector{T},
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    fold     :: Int;
    pids     :: DenseVector{Int} = procs(),
    means    :: DenseVector{T}   = mean(T,x, shared=true, pids=pids),
    invstds  :: DenseVector{T}   = invstd(x,means, shared=true, pids=pids),
    max_iter :: Int              = 100,
    max_step :: Int              = 50,
    quiet    :: Bool             = true
)

    # make vector of indices for folds
    test_idx = folds .== fold
    test_size = sum(test_idx)

    # preallocate vector for output
    myerrors = zeros(T, test_size)

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # allocate the arrays for the training set
    x_train = x[train_idx,:]
    y_train = y[train_idx]
    Xb      = SharedArray(T, test_size, init = S -> S[localindexes(S)] = zero(T), pids=pids)
    b       = SharedArray(T, size(x,2), init = S -> S[localindexes(S)] = zero(T), pids=pids)
    r       = SharedArray(T, test_size, init = S -> S[localindexes(S)] = zero(T), pids=pids)

    # compute the regularization path on the training set
    betas = iht_path(x_train,y_train,path, max_iter=max_iter, quiet=quiet, max_step=max_step, means=means, invstds=invstds, pids=pids)

    # compute the mean out-of-sample error for the TEST set
    # do this for every computed model in regularization path
    for i = 1:size(betas,2)

        # pull ith model in dense vector format
        b2 = full(vec(betas[:,i]))

        # copy it into SharedArray b
        copy!(b,b2)

        # indices stores Boolean indexes of nonzeroes in b
        update_indices!(indices, b, p=p)

        # compute estimated response Xb with $(path[i]) nonzeroes
        xb!(Xb,x_test,b,indices,path[i],test_idx, means=means, invstds=invstds, pids=pids)

        # compute residuals
        difference!(r,y,Xb)

        # compute out-of-sample error as squared residual averaged over size of test set
        myerrors[i] = sumabs2(r) / test_size / 2
    end

    return myerrors
end



"""
    cv_iht(x::BEDFile, y, path, q)

If used with a `BEDFile` object `x`, then the additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs()`.
- `means`, a vector of SNP means. Defaults to `mean(T, x, shared=true, pids=procs()`.
- `invstds`, a vector of SNP precisions. Defaults to `invstd(x, means, shared=true, pids=procs()`.
"""
function cv_iht{T <: Float}(
    x             :: BEDFile,
    y             :: DenseVector{T},
    path          :: DenseVector{Int},
    q             :: Int;
    folds         :: DenseVector{Int} = cv_get_folds(sdata(y),q),
    pids          :: DenseVector{Int} = procs(),
    means         :: DenseVector{T}   = mean(T,x, shared=true, pids=pids),
    invstds       :: DenseVector{T}   = invstd(x,means, shared=true, pids=pids),
    tol           :: Float            = convert(T, 1e-4),
    n             :: Int              = length(y),
    p             :: Int              = size(x,2),
    max_iter      :: Int              = 100,
    max_step      :: Int              = 50,
    quiet         :: Bool             = true,
    compute_model :: Bool             = false
)

    # how many elements are in the path?
    num_models = length(path)

    # preallocate vectors used in xval
    errors  = zeros(T, num_models)    # vector to save mean squared errors

    # want to compute a path for each fold
    # the folds are computed asynchronously
    # the @sync macro ensures that we wait for all of them to finish before proceeding
    @sync for i = 1:q
        # one_fold returns a vector of out-of-sample errors (MSE for linear regression)
        # @fetch(one_fold(...)) spawns one_fold() on a CPU and returns the MSE
        errors[i] = @fetch(one_fold(x, y, path, folds, i, max_iter=max_iter, max_step=max_step, quiet=quiet, means=means, invstds=invstds, pids=pids))
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

        # initialize model
        b = SharedArray(T, p, init = S -> S[localindexes(S)] = zero(T), pids=pids)

        # first use L0_reg to extract model
        output = L0_reg(x,y,k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, b=b, pids=pids)

        # which components of beta are nonzero?
        inferred_model = output["beta"] .!= zero(T)
        bidx = find( x -> x .!= zero(T), b)

        # allocate the submatrix of x corresponding to the inferred model
        x_inferred = zeros(T,n,sum(inferred_model))
        decompress_genotypes!(x_inferred,x)

        # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
        Xty = BLAS.gemv('T', one(T), x_inferred, y)
        xtx = BLAS.gemm('T', 'N', one(T), x_inferred, x_inferred)
        b   = xtx \ Xty
        return errors, b, bidx
    end
    return errors
end
