default_lambda{T <: Float}(x::DenseMatrix{T}, y::DenseVector{T}) = sqrt( log(size(x,2)) / length(y)) :: T
default_lambda{T <: Float}(x::DenseMatrix{T}, y::DenseVector{T}, q::Int) = (ones(T, q) * sqrt( log(size(x,2)) / length(y))) :: Vector{T}


"""
    fit_logistic(x, y, λ)

Refit a regularized logistic model using Newton's method.

Arguments:

- `x` is the `n` x `p` statistical model.
- `y` is the `n`-vector of binary responses.
- `λ` is the regularization parameter

Optional Arguments:

- `tol` is the convergence tolerance. Defaults to `1e-8`.
- `max_iter` sets the maximum number of backtracking steps. Defaults to `50`.
- `quiet` controls output. Defaults to `true` (no output).

Output:

- a vector of refit coefficients for `β`
- the number of backtracking steps taken.
"""
function fit_logistic!{T <: Float, V <: DenseVector}(
    v        :: IHTLogVariables{T, V},
    y        :: V, 
    λ        :: T;
    tol      :: T    = convert(T, 1e-8),
    max_iter :: Int  = 50,
    quiet    :: Bool = true
)
    # number of cases?
    n = length(y)

    # if b is not warm-started, then ensure that it is not entirely zero
    if all(v.bk .== 0) 
        v.bk[1] = logit(abs(mean(y)))
    end

    # initialize intermediate arrays for calculations
    #BLAS.gemv!('N', one(T), x, b, zero(T), xβ)
    A_mul_B!(v.xb, v.xk, v.bk)
    log2xb!(v.lxb, v.l2xb, v.xb)
    copy!(v.bk0, v.bk)
    fill!(v.db, zero(T))

    # track objective
    old_obj = convert(T, Inf)
    # new_obj = (sum(log(1 + exp(v.xb))) - dot(y, v.xb)) / n + λ*sumabs2(v.bk) / 2
    new_obj = logistic_loglik(v.xb, y, v.bk, λ)

    # output progress to console
    quiet || println("Iter\tHalves\tObjective")

    i = 0
    bktrk = 0
    # enter loop for Newton's method
    for i = 1:max_iter

        # db = (x'*(lxβ - y)) / n + λ*b
        BLAS.axpy!(-one(T), sdata(y), sdata(v.lxb))
        At_mul_B!(v.db, v.xk, v.lxb)
        scale!(v.db, 1/n)
        BLAS.axpy!(λ, v.bk, v.db)

        # d2b = (x'*diagm(l2xb)*x)/n + λ*I
        # note that log2xb!() already performs division by n on l2xb
        copy!(v.xk2, v.xk)
        scale!(v.l2xb, v.xk2)
        At_mul_B!(v.d2b, v.xk, v.xk2)
        v.d2b += λ*I

        # b = b0 - ntb = b0 - inv(d2b)*db
        #   = b0 - inv[ x' diagm(pi) diagm(1 - pi) x + λ*I] [x' (pi - y) + λ*b]
        try
            v.ntb = v.d2b \ v.db
        catch e
            warn("in fit_logistic, aborting after caught error: ", e)
            return i, div(bktrk, max_iter)
        end
        copy!(v.bk, v.bk0)
        BLAS.axpy!(-one(T), v.ntb, v.bk)

        # compute objective
        A_mul_B!(v.xb, v.xk, v.bk)
        # new_obj = (sum(log(1 + exp(v.xb))) - dot(y, v.xb)) / n + λ*sumabs2(v.bk) / 2
        new_obj = logistic_loglik(v.xb, y, v.bk, λ)

        # control against nonfinite objective
        isfinite(new_obj) || throw(error("in fit_logistic, objective is no longer finite"))

        # backtrack
        j = 0
        while (new_obj > old_obj + tol) && (j < 50)

            # increment iterator
            j += 1

            # b = b0 - 0.5*ntb
            copy!(v.bk, v.bk0)
            #BLAS.axpy!(p,-one(T) / (2^j),ntb,1,b,1)
            #BLAS.axpy!(-one(T) / (2^j),ntb,v.b)
            BLAS.axpy!(-1/(2^j), v.ntb, v.bk)

            # recalculate objective
            A_mul_B!(v.xb, v.xk, v.bk)
            new_obj = logistic_loglik(v.xb, y, v.bk, λ)

            # ensure a finite objective
            isfinite(new_obj) || throw(error("in fit_logistic, objective is no longer finite"))
        end

        # accumulate total backtracking steps
        bktrk += j

        # track distance between iterates
        dist = euclidean(v.bk, v.bk0) / (norm(v.bk0,2) + one(T))

        # track progress
        quiet || println(i, "\t", j, "\t", dist)

        # check for convergence
        # if converged, then return b
        dist < tol && return i, div(bktrk, i)

        # unconverged at this point, so update intermediate arrays
        #BLAS.gemv!('N', one(T), sdata(x), sdata(b), zero(T), sdata(xβ))
        A_mul_B!(v.xb, v.xk, v.bk)
        log2xb!(v.lxb, v.l2xb, v.xb)

        # save previous β 
        copy!(v.bk0, v.bk)
        old_obj = new_obj
    end

    warn("fit_logistic failed to converge in $(max_iter) iterations, exiting...")
    return i, div(bktrk, max_iter)
end



"""
L0 PENALIZED LOGISTIC REGRESSION

    L0_log(x,y,k) -> IHTLogResults 

This routine minimizes the loss function given by the negative logistic loglikelihood

    L(β) = sum(log(1 + exp(x*β))) - y'*x*β

subject to `β` lying in the set S_k = { x in R^p : || x ||_0 <= k }.
To ensure a stable model selection process, the optimization is performed over a Tikhonov-regularized copy of `L(β)`; the actual optimized objective is

    g(β) = L(β) + 0.5*lambda*sumabs2(β)

where `lambda` controls the strength of the L2 penalty.
This function extends the [MATLAB source code](http://users.ece.gatech.edu/sbahmani7/GraSP.html) for Sohail Bahmani's nonlinear hard thresholding pursuit framework [GraSP](http://jmlr.csail.mit.edu/papers/v14/bahmani13a.html).

Arguments:

- `x` is the `n` x `p` data matrix.
- `y` is the `n`-dimensional continuous response vector.
- `k` is the desired model size (support).

Optional Arguments:

- `lambda` is the strength of the regularization parameter. Defaults to `sqrt(log(p)/n)`.
- `mu` is the step size used in gradient descent. Defaults to `1.0`.
- `max_iter` is the maximum number of iterations for the algorithm. Defaults to `100`.
- `max_step` is the maximum number of backtracking steps for the step size calculation. Defaults to `100`.
- `tol` is the global tolerance. Defaults to `1e-6`.
- `tolG` is the tolerance for the gradient. Convergence occurs if the norm of the largest `3*k` elements of the gradient falls below `tolG`. Defaults to `1e-6`.
- `tolrefit` is the tolerance for Newton's method in the refitting routine. The refitting algorithm converges when the loss function calculated on the active set (that is, the refit coefficients) falls below `tolrefit`. Defaults to `1e-6`.
- `refit` is a `Bool` that controls refitting at every iterations. It is wise to refit the nonzero coefficients of `b` at every iteration as this may improve convergence behavior and estimation. Defaults to `true` (refit at every iteration).
- `quiet` is a `Bool` that controls algorithm output. Defaults to `true` (no output).

Outputs are wrapped into an `IHTLogResults` object with the following fields:

- 'time' is the compute time for the algorithm. Note that this does not account for time spent initializing optional argument defaults.
- 'iter' is the number of iterations that the algorithm took before converging.
- 'loss' is the optimal loss (half of residual sum of squares) at convergence.
- 'beta' is the final estimate of `b`.
- `active` is the final active set. The size of this set ranges from `k` to `2k`.
"""
function L0_log{T <: Float, V <: DenseVector}(
    x        :: DenseMatrix{T},
    y        :: V,
    k        :: Int;
    v        :: IHTLogVariables{T, V} = IHTLogVariables(x, y, k),
    lambda   :: T    = default_lambda(x, y), 
    mu       :: T    = one(T),
    tol      :: T    = convert(T, 1e-6),
    tolG     :: T    = convert(T, 1e-3),
    tolrefit :: T    = convert(T, 1e-6),
    max_iter :: Int  = 100,
    max_step :: Int  = 100,
    refit    :: Bool = true,
    quiet    :: Bool = true
)

    # start timer
    tic()

    # problem dimensions?
    n,p = size(x)

    # check arguments
    length(y) == n      || throw(ArgumentError("Length n = $n of response vector y does not match number of rows = $(size(x,1)) in x"))
    k        >  p       && throw(ArgumentError("Value of argument k = $k exceeds number of predictors p = $p"))
    lambda   <  zero(T) && throw(ArgumentError("Value of argument lambda = $lambda must be nonnegative"))
    mu       >  zero(T) || throw(ArgumentError("Value of argument mu = $mu must be positive"))
    tol      >  eps(T)  || throw(ArgumentError("Value of argument tol must exceed machine precision"))
    tolG     >  eps(T)  || throw(ArgumentError("Value of argument tolG must exceed machine precision"))
    tolrefit >  eps(T)  || throw(ArgumentError("Value of argument tolrefit must exceed machine precision"))
    max_iter >= 0       || throw(ArgumentError("Value of max_iter must be nonnegative\n"))
    max_step >= 0       || throw(ArgumentError("Value of max_step must be nonnegative\n"))

    # initialize return values
    iter      = 0                 # number of iterations of L0_reg
    exec_time = zero(T)           # compute time *within* L0_reg
    loss      = convert(T, Inf)   # loss function value

    # initialize algorithm parameters
    num_df    = min(p, 3*k)       # largest 3*k active components of gradient
    short_df  = min(p, 2*k)       # largest 2*k active components of gradient
    converged = false             # is algorithm converged?
    stuck     = false             # is algorithm stuck in a cycle?
    normdf    = convert(T, Inf)   # norm of active portion of gradient
    loss0     = convert(T, Inf)   # penultimate loss function value
    loss00    = convert(T, Inf)   # antepenultimate loss function value
    bktrk     = 0                 # number of Newton backtracking steps
    nt_iter   = 0                 # number of Newton iterations
    lt        = length(v.active)  # size of current active set?


    # formatted output to monitor algorithm progress
    quiet || print_header_log()

    # main GraSP iterations
    for iter = 1:max_iter

        # notify and break if maximum iterations are reached
        # also break if algorithm is cycling
        if iter >= max_iter || stuck

            # warn about hitting maximum iterations
            iter >= max_iter && print_maxiter(max_iter, loss) 

            # warn about cycling
            stuck && warn("L0_log appears to be cycling after $iter iterations, aborting early...\nCurrent loss: $loss\n")

            # send elements below tol to zero
            threshold!(v.b, tol)

            # if requested, apply final refit without regularization
            if refit
                copy!(v.idxs0, v.idxs)
                update_indices!(v.idxs, v.b)
                !isequal(v.idxs, v.idxs0) && copy!(v.xk, view(x, :, v.idxs))
                nt_iter, bktrk = fit_logistic!(v, y, zero(T), tol=tolrefit, max_iter=max_step, quiet=true)
                v.b[v.idxs] = v.bk
            end

            # stop timer
            exec_time = toq()

            # these are output variables for function
            # wrap them into a Dict and return
            return IHTLogResults(exec_time, iter, loss, copy(v.b), copy(v.active))
        end

        # save previous loss, iterate
        loss00 = loss0
        loss0  = loss
        copy!(v.b0, v.b)

        # size of current active set?
        if iter > 1
            v.active = union(v.dfidxs[1:short_df], v.bidxs[1:k])
        end
        lt = length(v.active)

        # update x*b
        # no need to compute anything if b = 0
        if all(v.b .== 0)
            fill!(v.xb, zero(T))
        else
#            A_mul_B!(v.xb, x, v.b) 
            update_xb!(v.xb, x, v.b, v.active, lt)
        end

        # recompute active loss = (-dot(y,Xb) + sum(log(1.0 + exp(Xb)))) / n + 0.5*lambda*sumabs2(b[active])
        # special case: b = 0 --> Xb = 0 --> loss = n*log(1 + exp(0))/n + 0.5*lambda*norm(0)
        if iter < 2
            loss = 1 / eps(T)
        elseif all(v.xb .== 0)
            loss = log(2)
        else
            loss = logistic_loglik(v.xb, y, v.b, v.active, lambda, k)
        end

        # guard against numerical instabilities in loss function
        check_finiteness(loss)

        # recompute active gradient df[active] = (x[:,active]'*(logistic(Xb) - y)) / n + lambda*b[active]
        # arrange calculations differently if active set is entire support 1, 2, ..., p
        if lt == p
            logistic_grad!(v.df, v.lxb, x, y, v.b, v.xb, lambda)
        else
            logistic_grad!(v.df, v.lxb, x, y, v.b, v.xb, v.active, lt, lambda)
        end
#        logistic!(v.lxb, v.xb)
#        v.df = (x' * (v.lxb - y)) / n + lambda*v.b

        # identify 2*k dominant directions in gradient
        selectperm!(v.dfidxs, v.df, 1:num_df, by=abs, rev=true, initialized=true)

        # clean b and fill, b[active] = b0[active] - μ*df[active]
        # note that sparsity level is size(active) which is one of [k, k+1, ..., 3*k]
        #update_x!(v.b, v.b0, v.df, v.active, mu)
        fill!(v.b, zero(T))
        v.b[v.active] = v.b0[v.active] .- mu.*v.df[v.active]

        # now apply hard threshold on model to original desired sparsity k
        project_k!(v.b, k)
        selectperm!(v.bidxs, v.b, 1:k, by=abs, rev=true, initialized=true)
        v.bk = v.b[view(v.bidxs, 1:k)]

        # refit nonzeroes in b?
        if refit

            # update boolean vector of nonzeroes
            copy!(v.idxs0, v.idxs)
            update_indices!(v.idxs, v.b)

            # update active set of x, if necessary
#            (iter == 1 || !isequal(idxs,idxs0)) && update_xk!(xk, x, idxs, k=k, n=n, p=p)
            copy!(v.xk, view(x, :, v.idxs))
            copy!(v.xk2, v.xk)

            # attempt refit but guard against possible instabilities or singularities
            # these are more frequent if lambda is small
            # if refitting destabilizes, then leave b alone
            try
                nt_iter, bktrk = fit_logistic!(v, y, lambda, tol=tolrefit, max_iter=max_step, quiet=true)
                v.b[v.idxs] = v.bk
            catch e
#                warn("in refitting, caught error: ", e)
#                warn("skipping refit")
                nt_iter = 0
                bktrk   = 0
            end # end try-catch for refit
        end # end refit

        # need norm of top 3*k components of gradient
        normdf = df_norm(v.df, v.dfidxs, 1, num_df)

        # guard against numerical instabilities in gradient
        check_finiteness(normdf)

        # check for convergence
        converged_obj  = abs(loss - loss0) < tol
        converged_grad = normdf < tolG
        converged      = converged_obj || converged_grad
        stuck          = abs(loss - loss00) < tol
#        stuck          = abs(loss - loss00) < tol && converged_grad

        # output algorithm progress
        quiet || @printf("%d\t%d\t%d\t%3.7f\t%3.7f\n",iter,nt_iter,bktrk,loss,normdf)

        # check for convergence
        # if converged and in feasible set, then algorithm converged before maximum iteration
        # perform final computations and output return variables
        if converged

            # send elements below tol to zero
            threshold!(v.b, tol)

            # if requested, apply final refit without regularization
            if refit
                copy!(v.idxs0, v.idxs)
                update_indices!(v.idxs, v.b)
#                !isequal(idxs,idxs0) && update_xk!(xk, x, idxs, k=k, n=n, p=p)
                copy!(v.xk, view(x, :, v.idxs))
                try
                    nt_iter, bktrk = fit_logistic!(v, y, zero(T), tol=tolrefit, max_iter=max_step, quiet=true)
                    v.b[v.idxs] = v.bk
                catch e
#                    warn("in final refitting, caught error: ", e)
#                    warn("skipping final refit")
                    nt_iter = 0
                    bktrk   = 0
                end # end try-catch for refit
            end

            # stop time
            exec_time = toq()

            # announce convergence
            !quiet && print_log_convergence(iter, loss, exec_time, normdf)

            # return output 
            return IHTLogResults(exec_time, iter, loss, copy(v.b), copy(v.active))
        end # end convergence check
    end # end main GraSP iterations

    # return null result
    return IHTLogResults(zero(T), 0, zero(T), zeros(T, 1), zeros(Int, 1)) 
end # end L0_log





"""
COMPUTE AN IHT REGULARIZATION PATH FOR LOGISTIC REGRESSION

    iht_path_log(x,y,path) -> SparseCSCMatrix

This subroutine computes best logistic models for matrix `x` and response `y` by calling `L0_log` for each model over a regularization path denoted by `path`.

Arguments:

- `x` is the `n` x `p` design matrix.
- `y` is the `n`-vector of responses.
- `path` is an `Int` vector that contains the model sizes to test.

Optional Arguments:

- `tol` is the global convergence tolerance for `L0_log`. Defaults to `1e-6`.
- `tolG` is the tolerance for the gradient. Convergence occurs if the norm of the largest `3*k` elements of the gradient falls below `tolG`. Defaults to `1e-6`.
- `tolrefit` is the tolerance for Newton's method in the refitting routine. The refitting algorithm converges when the loss function calculated on the active set (that is, the refit coefficients) falls below `tolrefit`. Defaults to `1e-6`.
- `max_iter` caps the number of iterations for the algorithm. Defaults to `100`.
- `max_step` caps the number of backtracking steps in the IHT kernel. Defaults to `100`.
- `refit` is a `Bool` that controls refitting at every iterations. It is wise to refit the nonzero coefficients of `b` at every iteration as this may improve convergence behavior and estimation. Defaults to `true` (refit at every iteration).
- `quiet` is a Boolean that controls the output. Defaults to `true` (no output).

Output:

- a sparse `p` x `length(path)` matrix where each column contains the computed model for each component of `path`.
"""
function iht_path_log{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    path     :: DenseVector{Int};
    lambdas  :: Vector{T} = default_lambda(x, y, length(path)),
    mu       :: T    = one(T),
    tol      :: T    = convert(T, 1e-6),
    tolG     :: T    = convert(T, 1e-3),
    tolrefit :: T    = convert(T, 1e-6),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    refit    :: Bool = true,
    quiet    :: Bool = true,
)

    # dimensions of problems?
    n,p = size(x)

    # how many models will we compute?
    num_models = length(path)

    # preallocate temporary arrays
    v = IHTLogVariables(x, y, 1)

    # betas will be a sparse matrix with models
    betas = spzeros(T, p, num_models)

    # compute the path
    for i = 1:num_models

        # model size?
        q = path[i]
        quiet || println("Current model size: $q")

        # update temporary variables with new threshold size q
        update_variables!(v, x, q)

        # store projection of beta onto largest k nonzeroes in magnitude
        project_k!(v.b, q)

        # current regularization parameter?
        λ = lambdas[i]

        # these arrays change in size from iteration to iteration
        # we must allocate them for every new model size

        # now compute current model
        output = L0_log(x, y, q, v=v, lambda=λ, mu=mu, tol=tol, tolG=tolG, tolrefit=tolrefit, max_iter=max_iter, max_step=max_step, refit=refit, quiet=quiet)

        # extract and save model
        betas[:,i] = sparsevec(output.beta)

        # run garbage collector to clear temporary arrays for next step of path
        #gc()
    end

    # return a sparsified copy of the models
    return betas
end



"""
COMPUTE ONE FOLD IN A CROSSVALIDATION SCHEME FOR A PENALIZED LOGISTIC REGRESSION REGULARIZATION PATH

    one_fold_log(x,y,path,folds,fold) -> Vector{Float}

For a regularization path given by the `Int` vector `path`,
this function performs penalized logistic regression on `x` and `y` and computes an out-of-sample error based on the indices given in `folds`.

Arguments:

- `x` is the `n` x `p` design matrix.
- `y` is the `n`-vector of responses.
- `path` is an `Int` vector that specifies which model sizes to include in the path, e.g. `path = collect(k0:increment:k_end)`.
- `folds` is an `Int` vector indicating which component of `y` goes to which fold, e.g. `folds = IHT.cv_get_folds(n,nfolds)`
- `fold` is the current fold to compute.

Optional Arguments:

- `tol` is the convergence tolerance to pass to the path computations. Defaults to `1e-4`.
- `max_iter` caps the number of permissible iterations in the IHT algorithm. Defaults to `100`.
- `max_step` caps the number of permissible backtracking steps. Defaults to `100`.
- `quiet` is a Boolean to activate output. Defaults to `true` (no output).
- `deviance` is an `String` that specifies the crossvalidation criterion, either `"deviance"` for the logistic loglikelihood or `"class"` for misclassification error. Defaults to `"deviance"`.

Output:

- `errors` is a vector of out-of-sample errors (MSEs) for the current fold.
"""
function one_fold_log{T <: Float}(
    x         :: DenseMatrix{T},
    y         :: DenseVector{T},
    path      :: DenseVector{Int},
    folds     :: DenseVector{Int},
    fold      :: Int;
    lambdas   :: Vector{T} = default_lambda(x, y, length(path)),
    criterion :: String = "deviance",
    tol       :: T      = convert(T, 1e-6),
    tolG      :: T      = convert(T, 1e-3),
    tolrefit  :: T      = convert(T, 1e-6),
    max_iter  :: Int    = 100,
    max_step  :: Int    = 100,
    refit     :: Bool   = true,
    quiet     :: Bool   = true,
)
    # dimension of problem?
    n,p = size(x)

    # ensure that crossvalidation criterion is valid
    criterion in ["deviance", "class"] || throw(ArgumentError("Argument criterion must be 'deviance' or 'class'"))

    # for deviance, will need an extra vector of "falses"
    # this is used when computing deviances with logistic_loglik()
    if criterion == "deviance"
        notrues = falses(p)
    end

    # make vector of indices for folds
    # train_idx is the vector that indexes the TRAINING set
    train_idx = folds .!= fold 

    # convert test_idx to numeric
    #test_idx = convert(Vector{Int}, test_idx)

    # allocate the arrays for the training set
    x_train = x[train_idx,:]
    y_train = y[train_idx]

    # compute the regularization path on the training set
    betas = iht_path_log(x_train, y_train, path, tol=tol, tolG=tolG, tolrefit=tolrefit, max_iter=max_iter, max_step=max_step, refit=refit, quiet=quiet, lambdas=lambdas)

    # (masked) estimated responses
    # mask based on test_idx
    xβ  = x*betas
    fill!(view(xβ, train_idx, :), zero(T))
    lxb = zeros(T, size(xβ))
    logistic!(lxb, xβ)
    fill!(view(lxb, train_idx, :), zero(T)) 

    # compute the mean out-of-sample error for the TEST set
    if criterion == "class"
        errors = vec(sumabs(lxb .- y, 1))
    else # else -> criterion == "deviance"
        errors = - one(T) / (2*n) .* vec(sum(xβ .* y, 1) .- sum(log(one(T) .+ exp(xβ)), 1))
    end
    return errors :: Vector{T}
end




#"""
#    pfold_log(xfile, xtfile, x2file, yfile, meanfile, invstdfile, path, folds, numfolds[, pids=procs()])
#
#This function is the parallel execution kernel in `cv_log()`. It is not meant to be called outside of `cv_iht()`.
#It will distribute `numfolds` crossvalidation folds across the processes supplied by the optional argument `pids` and call `one_fold_log()` for each fold.
#Each fold will compute a regularization path given by `path`.
#`pfold()` collects the vectors of MSEs returned by calling `one_fold_log()` for each process, reduces them, and returns their average across all folds.
#"""
function pfold_log{T <: Float}(
    x         :: DenseMatrix{T},
    y         :: DenseVector{T},
    path      :: DenseVector{Int},
    folds     :: DenseVector{Int},
    q         :: Int;
    pids      :: Vector{Int} = procs(),
    lambdas   :: DenseVector{T} = default_lambda(x, y, length(path)),
    criterion :: String = "deviance",
    tol       :: T      = convert(T, 1e-6),
    tolG      :: T      = convert(T, 1e-3),
    tolrefit  :: T      = convert(T, 1e-6),
    max_iter  :: Int    = 100,
    max_step  :: Int    = 100,
    quiet     :: Bool   = true,
    refit     :: Bool   = true,
    header    :: Bool   = false
)

    # problem size?
    n,p = size(x)

    # ensure that crossvalidation criterion is valid
    criterion in ["deviance", "class"] || throw(ArgumentError("Argument criterion must be 'deviance' or 'class'"))

    # how many CPU processes can pfold use?
    np = length(pids)

    # report on CPU processes
    quiet || println("pfold: np = ", np)
    quiet || println("pids = ", pids)

    # set up function to share state (indices of folds)
    i = 1
    nextidx() = (idx=i; i+=1; idx)

    # preallocate cell array for results
    results = SharedArray(T, (length(path),q), pids=pids)

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
                        r = remotecall_fetch(worker) do
                                one_fold_log(x, y, path, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, tolG=tolG, tolrefit=tolrefit, refit=refit, criterion=criterion, lambdas=lambdas)
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
PARALLEL CROSSVALIDATION ROUTINE FOR GRASP ALGORITHM FOR PENALIZED LOGISTIC REGRESSION

    cv_iht_log(x,y,path,q) -> Vector{Float}

This function will perform `q`-fold cross validation for the ideal model size in GraSP logistic regression.
It computes several paths as specified in the `path` argument using the design matrix `x` and the response vector `y`.
Each path is asynchronously spawned using any available processor.
For each path, one fold is held out of the analysis for testing, while the rest of the data are used for training.
The function to compute each path, `one_fold_log()`, will return a vector of out-of-sample errors (MCEs).

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
- `max_step` caps the number of permissible backtracking steps. Defaults to `100`.
- `quiet` is a Boolean to activate output. Defaults to `true` (no output).
   *NOTA BENE*: each processor outputs feed to the console without regard to the others,
   so setting `quiet = false` can yield very messy output!
- `refit` is a `Bool` to indicate whether or not to recompute the best model. Note that this argument affects both the continuous refitting in the GraSP algorithm in addition to a final refit after crossvalidation. Defaults to `true` (recompute).
- `deviance` is an `String` that specifies the crossvalidation criterion, either `"deviance"` for the logistic loglikelihood or `"class"` for misclassification error. Defaults to `"deviance"`.

Output:

- `errors` is the averaged MSE over all folds.

If called with `refit = true`, then the output also includes, for best model size `k_star`:

- `b`, a vector of `k_star` floats
- `bidx`, a vector of `k_star` indices indicating the support of the best model.
"""
function cv_log{T <: Float}(
    x         :: DenseMatrix{T},
    y         :: DenseVector{T};
    q         :: Int              = cv_get_num_folds(3, 5),
    path      :: DenseVector{Int} = collect(1:min(size(x,2),20)),
    folds     :: DenseVector{Int} = cv_get_folds(sdata(y),q),
    pids      :: Vector{Int}      = procs(),
    lambdas   :: Vector{T}        = default_lambda(x, y, length(path)),
    criterion :: String = "deviance",
    tol       :: T      = convert(T, 1e-6),
    tolG      :: T      = convert(T, 1e-3),
    tolrefit  :: T      = convert(T, 1e-6),
    max_iter  :: Int    = 100,
    max_step  :: Int    = 100,
    quiet     :: Bool   = true,
    refit     :: Bool   = true
)

    # problem size?
    n,p = size(x)
    @assert n == length(y)

    # ensure that crossvalidation criterion is valid
    criterion in ["deviance", "class"] || throw(ArgumentError("Argument criterion must be 'deviance' or 'class'"))

    # how many elements are in the path?
    num_models = length(path)

    # compute crossvalidation deviances
    errors = pfold_log(x, y, path, folds, q, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids, tolG=tolG, tolrefit=tolrefit, refit=refit, criterion=criterion, lambdas=lambdas)

    # what is the best model size?
    k = path[indmin(errors)] :: Int

    # print results
    !quiet && print_cv_results(errors, path, k)

    # recompute ideal model
    # get λ value for best model
    λ = lambdas[path .== k][1]

    # use L0_log to extract model
    # with refit = true, L0_log will continuously refit predictors
    # no final refitting code necessary
    output = L0_log(x, y, k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, tolG=tolG, tolrefit=tolrefit, refit=true, lambda=zero(T))#lambda=λ)
    bidx   = find(output.beta)

    return IHTCrossvalidationResults(errors, sdata(path), output.beta[bidx], bidx, k)
end
