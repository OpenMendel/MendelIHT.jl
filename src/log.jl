"""
L0 PENALIZED LOGISTIC REGRESSION

    L0_log{T <: Union{Float32, Float64}}(x,y,k) -> Dict{ASCIIString,Any}

This routine minimizes the loss function given by the negative logistic loglikelihood

    L(b) = sum(log(1 + exp(x*b))) - y'*x*b

subject to `b` lying in the set S_k = { x in R^p : || x ||_0 <= k }.
To ensure a stable model selection process, the optimization is performed over a Tikhonov-regularized copy of `L(b)`; the actual optimized objective is

    g(b) = L(b) + 0.5*lambda*sumabs2(b)

where `lambda` controls the strength of the L2 penalty.
This function extends the [MATLAB source code](http://users.ece.gatech.edu/sbahmani7/GraSP.html) for Sohail Bahmani's nonlinear hard thresholding pursuit framework [GraSP](http://jmlr.csail.mit.edu/papers/v14/bahmani13a.html).

Arguments:

- `x` is the `n` x `p` data matrix.
- `y` is the `n`-dimensional continuous response vector.
- `k` is the desired model size (support).

Optional Arguments:

- `n` is the number of samples. Defaults to `length(y)`.
- `p` is the number of predictors. Defaults to `size(x,2)`.
- `b` is the statistical model. Warm starts should use this argument. Defaults `zeros(p)`, the null model.
- `lambda` is the strength of the regularization parameter. Defaults to `sqrt(log(p)/n)`.
- `mu` is the step size used in gradient descent. Defaults to `1.0`.
- `max_iter` is the maximum number of iterations for the algorithm. Defaults to `100`.
- `max_step` is the maximum number of backtracking steps for the step size calculation. Defaults to `100`.
- `tol` is the global tolerance. Defaults to `1e-6`.
- `tolG` is the tolerance for the gradient. Convergence occurs if the norm of the largest `3*k` elements of the gradient falls below `tolG`. Defaults to `1e-6`.
- `tolrefit` is the tolerance for Newton's method in the refitting routine. The refitting algorithm converges when the loss function calculated on the active set (that is, the refit coefficients) falls below `tolrefit`. Defaults to `1e-6`.
- `refit` is a `Bool` that controls refitting at every iterations. It is wise to refit the nonzero coefficients of `b` at every iteration as this may improve convergence behavior and estimation. Defaults to `true` (refit at every iteration).
- `quiet` is a `Bool` that controls algorithm output. Defaults to `true` (no output).
- several temporary arrays for intermediate steps of algorithm calculations:

    xk     = zeros(T,n,k)  # store k columns of x for refitting
    xk2    = zeros(T,n,k)  # copy of xk also used in refitting
    d2b    = zeros(T,k,k)  # Hessian of k active components of b
    b0     = zeros(T,p)    # previous iterate beta0
    df     = zeros(T,p)    # (negative) gradient
    Xb     = zeros(T,n)    # x*b
    lxb    = zeros(T,n)    # logistic(x*b) = 1 ./ (1 + exp(-x*b))
    l2xb   = zeros(T,n)    # lxb * (1 - lxb)
    bk     = zeros(T,k)    # temporary array of the k active predictors in b
    bk0    = zeros(T,k)    # copy of bk used in refitting
    ntb    = zeros(T,k)    # Newton step for bk used in refitting
    db     = zeros(T,k)    # gradient of bk used in refitting
    dfk    = zeros(T,k)    # size k subset of df used in refitting
    bidxs  = collect(1:p)        # indices that sort b
    dfidxs = collect(1:p)        # indices that sort df
    T      = collect(1:p)        # union of active subsets of b and df
    idxs   = falses(p)           # nonzero components of b
    idxs0  = falses(p)           # store previous nonzero indicators for b

Outputs are wrapped into a `Dict{ASCIIString,Any}` with the following fields:

- 'time' is the compute time for the algorithm. Note that this does not account for time spent initializing optional argument defaults.
- 'iter' is the number of iterations that the algorithm took before converging.
- 'loss' is the optimal loss (half of residual sum of squares) at convergence.
- 'beta' is the final estimate of `b`.
"""
function L0_log{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    k        :: Int;
    n        :: Int              = length(y),
    p        :: Int              = size(x,2),
    xk       :: DenseMatrix{T}   = zeros(T, n,k),
    xk2      :: DenseMatrix{T}   = zeros(T, n,k),
    d2b      :: DenseMatrix{T}   = zeros(T, k,k),
    b        :: DenseVector{T}   = zeros(T, p),
    b0       :: DenseVector{T}   = zeros(T, p),
    df       :: DenseVector{T}   = zeros(T, p),
    Xb       :: DenseVector{T}   = zeros(T, n),
    lxb      :: DenseVector{T}   = zeros(T, n),
    l2xb     :: DenseVector{T}   = zeros(T, n),
    bk       :: DenseVector{T}   = zeros(T, k),
    bk2      :: DenseVector{T}   = zeros(T, k),
    bk0      :: DenseVector{T}   = zeros(T, k),
    ntb      :: DenseVector{T}   = zeros(T, k),
    db       :: DenseVector{T}   = zeros(T, k),
    dfk      :: DenseVector{T}   = zeros(T, k),
    active   :: DenseVector{Int} = collect(1:p),
    bidxs    :: DenseVector{Int} = collect(1:p),
    dfidxs   :: DenseVector{Int} = collect(1:p),
    idxs     :: BitArray{1}      = falses(p),
    idxs0    :: BitArray{1}      = falses(p),
    lambda   :: Float            = convert(T, sqrt(log(p)/n)),
    mu       :: Float            = one(T),
    tol      :: Float            = convert(T, 1e-6),
    tolG     :: Float            = convert(T, 1e-3),
    tolrefit :: Float            = convert(T, 1e-6),
    max_iter :: Int              = 100,
    max_step :: Int              = 100,
    refit    :: Bool             = true,
    quiet    :: Bool             = true,
)

    # start timer
    tic()

    # check arguments
    n        == size(x,1) || throw(ArgumentError("Length n = $n of response vector y does not match number of rows = $(size(x,1)) in x"))
    k        >  p         && throw(ArgumentError("Value of argument k = $k exceeds number of predictors p = $p"))
    lambda   <  zero(T)   && throw(ArgumentError("Value of argument lambda = $lambda must be nonnegative"))
    mu       >  zero(T)   || throw(ArgumentError("Value of argument mu must be positive"))
    tol      >  eps(T)    || throw(ArgumentError("Value of argument tol must exceed machine precision"))
    tolG     >  eps(T)    || throw(ArgumentError("Value of argument tolG must exceed machine precision"))
    tolrefit >  eps(T)    || throw(ArgumentError("Value of argument tolrefit must exceed machine precision"))
    max_iter >= 0         || throw(ArgumentError("Value of max_iter must be nonnegative\n"))
    max_step >= 0         || throw(ArgumentError("Value of max_step must be nonnegative\n"))

    # initialize return values
    mm_iter   = 0               # number of iterations of L0_reg
    mm_time   = zero(T)         # compute time *within* L0_reg
    loss      = oftype(mu, Inf) # loss function value

    # initialize algorithm parameters
    num_df    = min(p,3*k)      # largest 3*k active components of gradient
    short_df  = min(p,2*k)      # largest 2*k active components of gradient
    converged = false           # is algorithm converged?
    stuck     = false           # is algorithm stuck in a cycle?
    normdf    = oftype(mu, Inf) # norm of active portion of gradient
    loss0     = oftype(mu, Inf) # penultimate loss function value
    loss00    = oftype(mu, Inf) # antepenultimate loss function value
    bktrk     = 0               # number of Newton backtracking steps
    nt_iter   = 0               # number of Newton iterations
    lt        = length(active)  # size of current active set?


    # formatted output to monitor algorithm progress
    if !quiet
         println("Iter\tSteps\tHalves\tLoss\t\tGrad Norm")
         println("0\t0\t0\tInf\t\tInf")
    end

    # main GraSP iterations
    for mm_iter = 1:max_iter

        # notify and break if maximum iterations are reached
        # also break if algorithm is cycling
        if mm_iter >= max_iter || stuck

            # warn about hitting maximum iterations
            mm_iter >= max_iter && warn("L0_log has hit maximum iterations $(max_iter)!\nCurrent loss: $(loss)\n")

            # warn about cylcing
            stuck && warn("L0_log appears to be cycling after $mm_iter iterations, aborting early...\nCurrent loss: $loss\n")

            # send elements below tol to zero
            threshold!(b, tol, n=p)

            # if requested, apply final refit without regularization
            if refit
                copy!(idxs0,idxs)
                update_indices!(idxs, b, p=p)
                !isequal(idxs,idxs0) && update_xk!(xk, x, idxs, k=k, n=n, p=p)
                bk2, nt_iter, bktrk = fit_logistic(xk, y, zero(T), n=n, p=k, d2b=d2b, x2=xk2, b=bk, b0=bk0, ntb=ntb, db=db, Xb=Xb, lxb=lxb, l2xb=l2xb, tol=tolrefit, max_iter=max_step, quiet=true)
                b[idxs] = bk2
            end

            # stop timer
            mm_time = toq()

            # these are output variables for function
            # wrap them into a Dict and return
            output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => loss, "iter" => mm_iter, "beta" => copy(b), "active" => copy(active))

            return output
        end

        # save previous loss, iterate
        loss00 = loss0
        loss0  = loss
        copy!(b0,b)

        # size of current active set?
        if mm_iter > 1
            active = union(dfidxs[1:short_df],bidxs[1:k])
        end
        lt = length(active)

        # update x*b
        # no need to compute anything if b = 0
        # force fitted probabilities close to 0,1 to be equal to 0,1
        if all(b .== zero(T))
            fill!(Xb,zero(T))
        else
            update_xb!(Xb,x,b,active,lt,p=p,n=n)
#            update_xb!(Xb,x,b,bidxs,k,p=p,n=n)
            threshold!(Xb,tol,n=n)
        end

        # recompute active loss = (-dot(y,Xb) + sum(log(1.0 + exp(Xb)))) / n + 0.5*lambda*sumabs2(b[active])
        # special case: b = 0 --> Xb = 0 --> loss = n*log(1 + exp(0))/n + 0.5*lambda*norm(0)
        if all(Xb .== zero(T))
            loss = log(one(T) + one(T))
        else
            loss = logistic_loglik(Xb,y,b,active,lambda,k, n=n)
        end

        # guard against numerical instabilities in loss function
        isnan(loss) && throw(error("Loss function is NaN, something went wrong..."))
        isinf(loss) && throw(error("Loss function is Inf, something went wrong..."))

        # recompute active gradient df[active] = (x[:,active]'*(logistic(Xb) - y)) / n + lambda*b[active]
        # arrange calculations differently if active set is entire support 1, 2, ..., p
        if lt == p
            logistic_grad!(df, lxb, x, y, b, Xb, lambda, n=n, p=p)
        else
            logistic_grad!(df, lxb, x, y, b, Xb, active, lt, lambda, n=n)
        end

        # identify 3*k dominant directions in gradient
        selectperm!(dfidxs, df, 1:num_df, by=abs, rev=true, initialized=true)

        # clean b and fill, b[active] = b0[active] - mu*df[active]
        # note that sparsity level is size(active) which is one of [k, k+1, ..., 3*k]
        update_x!(b, b0, df, active, mu, k=lt)

        # now apply hard threshold on model to original desired sparsity k
        project_k!(b,bk,bidxs,k)

        # refit nonzeroes in b?
        if refit

            # update boolean vector of nonzeroes
            copy!(idxs0,idxs)
            update_indices!(idxs, b, p=p)

            # update active set of x, if necessary
            (mm_iter == 1 || !isequal(idxs,idxs0)) && update_xk!(xk, x, idxs, k=k, n=n, p=p)

            # attempt refit but guard against possible instabilities or singularities
            # these are more frequent if lambda is small
            # if refitting destabilizes, then leave b alone
            try
                bk2, nt_iter, bktrk = fit_logistic(xk, y, lambda, n=n, p=k, d2b=d2b, x2=xk2, b=bk, b0=bk0, ntb=ntb, db=db, Xb=Xb, lxb=lxb, l2xb=l2xb, tol=tolrefit, max_iter=max_step, quiet=true)
                b[idxs] = bk2
            catch e
                warn("in refitting, caught error: ", e)
                warn("skipping refit")
                nt_iter = 0
                bktrk   = 0
            end # end try-catch for refit
        end # end refit

        # need norm of top 3*k components of gradient
        normdf = df_norm(df, dfidxs, 1, num_df)

        # guard against numerical instabilities in gradient
        isnan(normdf) && throw(error("Gradient contains NaN, something went wrong..."))
        isinf(normdf) && throw(error("Gradient contains Inf, something went wrong..."))

        # check for convergence
        converged_obj  = abs(loss - loss0) < tol
        converged_grad = normdf < tolG
        converged      = converged_obj || converged_grad
        stuck          = abs(loss - loss00) < tol
#        stuck          = abs(loss - loss00) < tol && converged_grad

        # output algorithm progress
        quiet || @printf("%d\t%d\t%d\t%3.7f\t%3.7f\n",mm_iter,nt_iter,bktrk,loss,normdf)

        # check for convergence
        # if converged and in feasible set, then algorithm converged before maximum iteration
        # perform final computations and output return variables
        if converged

            # send elements below tol to zero
            threshold!(b, tol, n=p)

            # if requested, apply final refit without regularization
            if refit
                copy!(idxs0,idxs)
                update_indices!(idxs, b, p=p)
                !isequal(idxs,idxs0) && update_xk!(xk, x, idxs, k=k, n=n, p=p)
                bk2,nt_iter,bktrk = fit_logistic(xk, y, zero(T), n=n, p=k, d2b=d2b, x2=xk2, b=bk, b0=bk0, ntb=ntb, db=db, Xb=Xb, lxb=lxb, l2xb=l2xb, tol=tolrefit, max_iter=max_step, quiet=true)
                b[idxs] = bk2
            end

            # stop time
            mm_time = toq()

            if !quiet
                println("\nL0_log has converged successfully.")
                @printf("Results:\nIterations: %d\n", mm_iter)
                @printf("Final Loss: %3.7f\n", loss)
                @printf("Norm of active gradient: %3.7f\n", normdf)
                @printf("Total Compute Time: %3.3f sec\n", mm_time)
            end

            # these are output variables for function
            # wrap them into a Dict and return
            output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => loss, "iter" => mm_iter, "beta" => copy(b), "active" => copy(active))

            return output
        end # end convergence check
    end # end main GraSP iterations
end # end L0_log


### MENTION ADDITION OF "ACTIVE" VECTOR OUTPUT? ###
"""
If used with a `BEDFile` object `x`, then the additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs()`.
- `means`, a vector of SNP means. Defaults to `mean(T, x, shared=true, pids=procs()`.
- `invstds`, a vector of SNP precisions. Defaults to `invstd(x, means, shared=true, pids=procs()`.
"""
function L0_log{T <: Float}(
    x        :: BEDFile,
    y        :: SharedVector{T},
    k        :: Int;
    n        :: Int              = length(y),
    p        :: Int              = size(x,2),
    pids     :: DenseVector{Int} = procs(),
    means    :: SharedVector{T}  = mean(T,x, shared=true, pids=pids),
    invstds  :: SharedVector{T}  = invstd(x,means, shared=true, pids=pids),
    xk       :: DenseMatrix{T}   = zeros(T, n,k),
    xk2      :: DenseMatrix{T}   = zeros(T, n,k),
    d2b      :: DenseMatrix{T}   = zeros(T, k,k),
    b        :: SharedVector{T}  = SharedArray(T, p, pids=pids),
    b0       :: SharedVector{T}  = SharedArray(T, p, pids=pids),
    df       :: SharedVector{T}  = SharedArray(T, p, pids=pids),
    Xb       :: SharedVector{T}  = SharedArray(T, n, pids=pids),
    lxb      :: SharedVector{T}  = SharedArray(T, n, pids=pids),
    l2xb     :: DenseVector{T}   = zeros(T, n),
    bk       :: DenseVector{T}   = zeros(T, k),
    bk2      :: DenseVector{T}   = zeros(T, k),
    bk0      :: DenseVector{T}   = zeros(T, k),
    ntb      :: DenseVector{T}   = zeros(T, k),
    db       :: DenseVector{T}   = zeros(T, k),
    dfk      :: DenseVector{T}   = zeros(T, k),
    active   :: DenseVector{Int} = collect(1:p),
    bidxs    :: DenseVector{Int} = collect(1:p),
    dfidxs   :: DenseVector{Int} = collect(1:p),
    mask_n   :: DenseVector{Int} = ones(Int, n),
    idxs     :: BitArray{1}      = falses(p),
    idxs2    :: BitArray{1}      = falses(p),
    idxs0    :: BitArray{1}      = falses(p),
    lambda   :: Float            = convert(T, sqrt(log(p)/n)),
    mu       :: Float            = one(T),
    tol      :: Float            = convert(T, 1e-6),
    tolG     :: Float            = convert(T, 1e-3),
    tolrefit :: Float            = convert(T, 1e-6),
    max_iter :: Int              = 100,
    max_step :: Int              = 100,
    mn       :: Int              = sum(mask_n),
    refit    :: Bool             = true,
    quiet    :: Bool             = true,
)

    # start timer
    tic()

    # check arguments
    n        == size(x,1) || throw(ArgumentError("Length n = $n of response vector y does not match number of rows = $(size(x,1)) in x"))
    k        >  p         && throw(ArgumentError("Value of argument k = $k exceeds number of predictors p = $p"))
    lambda   <  zero(T)   && throw(ArgumentError("Value of argument lambda = $lambda must be nonnegative"))
    mu       >  zero(T)   || throw(ArgumentError("Value of argument mu must be positive"))
    tol      >  eps(T)    || throw(ArgumentError("Value of argument tol must exceed machine precision"))
    tolG     >  eps(T)    || throw(ArgumentError("Value of argument tolG must exceed machine precision"))
    tolrefit >  eps(T)    || throw(ArgumentError("Value of argument tolrefit must exceed machine precision"))
    max_iter >= 0         || throw(ArgumentError("Value of max_iter must be nonnegative\n"))
    max_step >= 0         || throw(ArgumentError("Value of max_step must be nonnegative\n"))

    # enforce all 0/1 in mask_n
    sum((mask_n .== 1) $ (mask_n .== 0)) == n || throw(ArgumentError("Argument mask_n can only contain 1s and 0s"))

    # initialize return values
    mm_iter   = 0               # number of iterations of L0_reg
    mm_time   = zero(T)         # compute time *within* L0_reg
    loss      = oftype(mu, Inf) # loss function value

    # initialize algorithm parameters
    num_df    = min(p,3*k)      # largest 3*k active components of gradient
    short_df  = min(p,2*k)      # largest 2*k active components of gradient
    converged = false           # is algorithm converged?
    normdf    = oftype(mu, Inf) # norm of active portion of gradient
    loss0     = oftype(mu, Inf) # previous loss function value
    bktrk     = 0               # track loss and number of backtracking steps
    nt_iter   = 0               # number of Newton iterations
    lt        = length(active)  # size of current active set?

    # formatted output to monitor algorithm progress
    if !quiet
         println("\nBegin MM algorithm\n")
         println("Iter\tSteps\tHalves\tLoss\t\tGrad Norm")
         println("0\t0\t0\tInf\t\tInf")
    end

    # main GraSP iterations
    for mm_iter = 1:max_iter

        # notify and break if maximum iterations are reached.
        if mm_iter >= max_iter

            # warn about hitting maximum iterations
            warn("L0_log has hit maximum iterations $(max_iter)!\nCurrent loss: $(loss)\n")

            # send elements below tol to zero
            threshold!(b, tol, n=p)

            # if requested, apply final refit without regularization
            if refit
                copy!(idxs0,idxs)
                update_indices!(idxs, b, p=p)
                !isequal(idxs,idxs0) && decompress_genotypes!(xk, x, idxs, mask_n, means=means,invstds=invstds)
                bk2,bktrk = fit_logistic(xk, y, mask_n, zero(T), n=n, p=k, d2b=d2b, x2=xk2, b=bk, b0=bk0, ntb=ntb, db=db, Xb=Xb, lxb=lxb, l2xb=l2xb, tol=tolrefit, max_iter=max_step, quiet=true)
                b[idxs] = bk2
            end

            # stop timer
            mm_time = toq()

            # these are output variables for function
            # wrap them into a Dict and return
            output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => loss, "iter" => mm_iter, "beta" => copy(b), "active" => copy(active))

            return output
        end

        # save previous loss, iterate
        loss0 = loss
        copy!(b0,b)

        # size of current active set?
        if mm_iter > 1
            active = union(dfidxs[1:short_df],bidxs[1:k])
        end
        lt = length(active)

        # update x*b
        # no need to compute anything if b = 0
        if all(b .== zero(T))
            fill!(Xb,zero(T))
        else
            fill!(idxs2,false)
            idxs2[active] = true
            xb!(Xb,x,b,idxs2,sum(idxs2),mask_n, means=means, invstds=invstds, pids=pids)
        end

        # recompute active loss = (-dot(y,Xb) + sum(log(1.0 + exp(Xb)))) / n + 0.5*lambda*sumabs2(b[active])
        # special case: b = 0 --> Xb = 0 --> loss = n*log(1 + exp(0))/n + 0.5*lambda*norm(0)
        if all(Xb .== zero(T))
            loss = log(one(T) + one(T))
        else
            loss = logistic_loglik(Xb,y,b,bidxs,mask_n,0,lambda,k, n=n, mn=mn)
        end

        # guard against numerical instabilities in loss function
        isnan(loss) && throw(error("Loss function is NaN, something went wrong..."))
        isinf(loss) && throw(error("Loss function is Inf, something went wrong..."))

        # recompute active gradient df[active] = (x[:,active]'*(logistic(Xb) - y)) / n + lambda*b[active]
        # arrange calculations differently if active set is entire support 1, 2, ..., p
        if lt == p
            logistic_grad!(df, lxb, x, y, b, Xb, means, invstds, mask_n, lambda, n=n, p=p, pids=pids, mn=mn)
        else
            logistic_grad!(df, lxb, x, y, b, Xb, means, invstds, active, mask_n, lt, lambda, n=n, mn=mn)
        end

        # identify 3*k dominant directions in gradient
        selectperm!(dfidxs, df, 1:num_df, by=abs, rev=true, initialized=true)

        # clean b and fill, b[active] = b0[active] - mu*df[active]
        # note that sparsity level is size(active) which is one of [k, k+1, ..., 3*k]
        update_x!(b, b0, df, active, mu, k=lt)

        # now apply hard threshold on model to enforce original desired sparsity k
        project_k!(b,bk,bidxs,k)

        # refit nonzeroes in b?
        if refit

            # update boolean vector of nonzeroes
            copy!(idxs0,idxs)
            update_indices!(idxs, b, p=p)

            # update active set of x, if necessary
            (mm_iter == 1 || !isequal(idxs,idxs0)) && decompress_genotypes!(xk, x, idxs, mask_n, means=means, invstds=invstds)

            # attempt refit but guard against possible instabilities or singularities
            # these are more frequent if lambda is small
            # if refitting destabilizes, then leave b alone
            try
                bk2, nt_iter, bktrk = fit_logistic(xk, y, mask_n, lambda, n=n, p=k, d2b=d2b, x2=xk2, b=bk, b0=bk0, ntb=ntb, db=db, Xb=Xb, lxb=lxb, l2xb=l2xb, tol=tolrefit, max_iter=max_step, quiet=true, mn=mn)
                b[idxs] = bk2
            catch e
                warn("in refitting, caught error: ", e)
                warn("skipping refit")
                bktrk   = 0
                nt_iter = 0
            end # end try-catch for refit
        end # end if-else for refit

        # need norm of largest 3*k components of gradient
        normdf = df_norm(df, dfidxs, 1, num_df)

        # guard against numerical instabilities in gradient
        isnan(normdf) && throw(error("Gradient contains NaN, something went wrong..."))
        isinf(normdf) && throw(error("Gradient contains Inf, something went wrong..."))

        # check for convergence
        converged_obj  = abs(loss - loss0) < tol
        converged_grad = normdf < tolG
        converged      = converged_obj || converged_grad

        # output algorithm progress
        quiet || @printf("%d\t%d\t%d\t%3.7f\t%3.7f\n",mm_iter,nt_iter,bktrk,loss,normdf)

        # check for convergence
        # if converged and in feasible set, then algorithm converged before maximum iteration
        # perform final computations and output return variables
        if converged

            # send elements below tol to zero
            threshold!(b, tol, n=p)

            # if requested, apply final refit without regularization
            if refit
                copy!(idxs0,idxs)
                update_indices!(idxs, b, p=p)
                !isequal(idxs,idxs0) && decompress_genotypes!(xk, x, idxs, mask_n, means=means,invstds=invstds)
                bk2, nt_iter, bktrk = fit_logistic(xk, y, zero(T), n=n, p=k, d2b=d2b, x2=xk2, b=bk, b0=bk0, ntb=ntb, db=db, Xb=Xb, lxb=lxb, l2xb=l2xb, tol=tolrefit, max_iter=max_step, quiet=true)
                b[idxs] = bk2
            end

            # stop time
            mm_time = toq()

            if !quiet
                println("\nL0_log has converged successfully.")
                @printf("Results:\nIterations: %d\n", mm_iter)
                @printf("Final Loss: %3.7f\n", loss)
                @printf("Norm of active gradient: %3.7f\n", normdf)
                @printf("Total Compute Time: %3.3f sec\n", mm_time)
            end

            # these are output variables for function
            # wrap them into a Dict and return
            output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => loss, "iter" => mm_iter, "beta" => copy(b), "active" => copy(active))

            return output
        end # end convergence check
    end # end main GraSP iterations
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
    n        :: Int     = length(y),
    p        :: Int     = size(x,2),
    lambdas  :: DenseVector{T} = ones(length(path)) * convert(T, sqrt(log(p) / n)),
    tol      :: Float   = convert(T, 1e-6),
    tolG     :: Float   = convert(T, 1e-3),
    tolrefit :: Float   = convert(T, 1e-6),
    max_iter :: Int     = 100,
    max_step :: Int     = 100,
    refit    :: Bool    = true,
    quiet    :: Bool    = true,
)

    # how many models will we compute?
    num_models = length(path)

    # preallocate space for intermediate steps of algorithm calculations
    b      = zeros(T, size(x,2))     # statistical model beta
    b0     = zeros(T, p)             # previous iterate beta0
    df     = zeros(T, p)             # (negative) gradient
    Xb     = zeros(T, n)             # x*b
    lxb    = zeros(T, n)             # logistic(x*b) = 1 ./ (1 + exp(-x*b))
    l2xb   = zeros(T, n)             # lxb * (1 - lxb)
    bidxs  = collect(1:p)            # indices that sort b
    dfidxs = collect(1:p)            # indices that sort df
    active = collect(1:p)            # union of active subsets of b and df
    idxs   = falses(p)               # nonzero components of b
    idxs0  = falses(p)               # store previous nonzero indicators for b
    betas  = spzeros(T,p,num_models) # a matrix to store calculated models

    # compute the path
    for i = 1:num_models

        # model size?
        q = path[i]
        quiet || println("Current model size: $q")

        # store projection of beta onto largest k nonzeroes in magnitude
        bk     = zeros(T,q)

        # current regularization parameter?
        lambda = lambdas[i]

        # these arrays change in size from iteration to iteration
        # we must allocate them for every new model size
        xk     = zeros(T, n, q) # store q columns of x for refitting
        xk2    = zeros(T, n, q) # copy of xk also used in refitting
        d2b    = zeros(T, q, q) # Hessian of q active components of b
        bk0    = zeros(T, q)    # copy of bk used in refitting
        bk2    = zeros(T, q)    # copy of bk used in refitting
        ntb    = zeros(T, q)    # Newton step for bk used in refitting
        db     = zeros(T, q)    # gradient of bk used in refitting
        dfk    = zeros(T, q)    # size q subset of df used in refitting

        # now compute current model
        output = L0_log(x,y,q, n=n, p=p, b=b, tol=tol, tolG=tolG, tolrefit=tolrefit, refit=refit, max_iter=max_iter, max_step=max_step, quiet=quiet, b0=b0, df=df, Xb=Xb, lxb=lxb, l2xb=l2xb, bidxs=bidxs, dfidxs=dfidxs, active=active, idxs=idxs, idxs0=idxs0, bk=bk, xk=xk, xk2=xk2, d2b=d2b, bk0=bk0, ntb=ntb, db=db, dfk=dfk, lambda=lambda, bk2=bk2)

        # extract and save model
        copy!(b, output["beta"])
        active = copy(output["active"])
        betas[:,i] = sparsevec(b)

        # run garbage collector to clear temporary arrays for next step of path
        xk  = false
        xk2 = false
        d2b = false
        bk  = false
        bk0 = false
        ntb = false
        db  = false
        dfk = false
        gc()
    end

    # return a sparsified copy of the models
    return betas
end

"""
    iht_path_log(x::BEDFile, y, path)

If used with a `BEDFile` object `x`, then the additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs()`.
- `means`, a vector of SNP means. Defaults to `mean(T, x, shared=true, pids=procs()`.
- `invstds`, a vector of SNP precisions. Defaults to `invstd(x, means, shared=true, pids=procs()`.
"""
function iht_path_log{T <: Float}(
    x        :: BEDFile,
    y        :: SharedVector{T},
    path     :: DenseVector{Int};
    pids     :: DenseVector{Int} = procs(),
    means    :: SharedVector{T}  = mean(T,x, shared=true, pids=pids),
    invstds  :: SharedVector{T}  = invstd(x,means, shared=true, pids=pids),
    mask_n   :: DenseVector{Int} = ones(Int,length(y)),
    n        :: Int              = length(y),
    p        :: Int              = size(x,2),
    lambdas  :: DenseVector{T}   = ones(length(path)) * conver(T, sqrt(log(p) / n)),
    tol      :: Float            = convert(T, 1e-6),
    tolG     :: Float            = convert(T, 1e-3),
    tolrefit :: Float            = convert(T, 1e-6),
    max_iter :: Int              = 100,
    max_step :: Int              = 100,
    refit    :: Bool             = true,
    quiet    :: Bool             = true,
)

    # how many models will we compute?
    num_models = length(path)

    # preallocate space for intermediate steps of algorithm calculations
    b      = SharedArray(T, p, pids=pids)  # statistical model
    b0     = SharedArray(T, p, pids=pids)  # previous iterate beta0
    df     = SharedArray(T, p, pids=pids)  # (negative) gradient
    Xb     = SharedArray(T, n, pids=pids)  # x*b
    lxb    = SharedArray(T, n, pids=pids)  # logistic(x*b) = 1 ./ (1 + exp(-x*b))
    l2xb   = zeros(T,n)                    # lxb * (1 - lxb)
    bidxs  = collect(1:p)                  # indices that sort b
    dfidxs = collect(1:p)                  # indices that sort df
    active = collect(1:p)                  # union of active subsets of b and df
    idxs   = falses(p)                     # nonzero components of b
    idxs2  = falses(p)                     # nonzero components of b
    idxs0  = falses(p)                     # store previous nonzero indicators for b
    betas  = spzeros(T,p,num_models)       # a matrix to store calculated models

    # compute the path
    for i = 1:num_models

        # model size?
        q = path[i]

        # store projection of beta onto largest k nonzeroes in magnitude
        bk     = zeros(T,q)

        # current regularization parameter?
        lambda = lambdas[i]

        # these arrays change in size from iteration to iteration
        # we must allocate them for every new model size
        xk     = zeros(T, n, q) # store q columns of x for refitting
        xk2    = zeros(T, n, q) # copy of xk also used in refitting
        d2b    = zeros(T, q, q) # Hessian of q active components of b
        bk0    = zeros(T, q)    # copy of bk used in refitting
        ntb    = zeros(T, q)    # Newton step for bk used in refitting
        db     = zeros(T, q)    # gradient of bk used in refitting
        dfk    = zeros(T, q)    # size q subset of df used in refitting

        # now compute current model
        output = L0_log(x,y,q, n=n, p=p, b=b, tol=tol, tolG=tolG, tolrefit=tolrefit, refit=refit, max_iter=max_iter, max_step=max_step, quiet=quiet, b0=b0, df=df, Xb=Xb, lxb=lxb, l2xb=l2xb, bidxs=bidxs, dfidxs=dfidxs, active=active, idxs=idxs, idxs0=idxs0, bk=bk, xk=xk, xk2=xk2, d2b=d2b, bk0=bk0, ntb=ntb, db=db, dfk=dfk, means=means, invstds=invstds, idxs2=idxs2, mask_n=mask_n, lambda=lambda)

        # extract and save model
        copy!(b, output["beta"])
        active = copy(output["active"])
        betas[:,i] = sparsevec(sdata(b))

        # run garbage compiler to clear temporary arrays for next step of path
        xk  = false
        xk2 = false
        d2b = false
        bk  = false
        bk0 = false
        ntb = false
        db  = false
        dfk = false
        gc()
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
- `deviance` is an `ASCIIString` that specifies the crossvalidation criterion, either `"deviance"` for the logistic loglikelihood or `"class"` for misclassification error. Defaults to `"deviance"`.

Output:

- `errors` is a vector of out-of-sample errors (MSEs) for the current fold.
"""
function one_fold_log{T <: Float}(
    x         :: DenseMatrix{T},
    y         :: DenseVector{T},
    path      :: DenseVector{Int},
    folds     :: DenseVector{Int},
    fold      :: Int;
    n         :: Int  = length(y),
    p         :: Int  = size(x,2),
    lambdas   :: DenseVector{T} = ones(length(path)) * convert(T, sqrt(log(p) / n)),
    criterion :: ASCIIString    = "deviance",
    tol       :: Float = convert(T, 1e-6),
    tolG      :: Float = convert(T, 1e-3),
    tolrefit  :: Float = convert(T, 1e-6),
    max_iter  :: Int   = 100,
    max_step  :: Int   = 100,
    refit     :: Bool  = true,
    quiet     :: Bool  = true,
)
    # ensure that crossvalidation criterion is valid
    criterion in ["deviance", "class"] || throw(ArgumentError("Argument criterion must be 'deviance' or 'class'"))

    # for deviance, will need an extra vector of "falses"
    # this is used when computing deviances with logistic_loglik()
    if criterion == "deviance"
        notrues = falses(p)
    end

    # make vector of indices for folds
    test_idx = folds .== fold
    test_size = sum(test_idx)

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # convert test_idx to numeric
    test_idx = convert(Vector{Int}, test_idx)

    # allocate the arrays for the training set
    x_train = x[train_idx,:]
    y_train = y[train_idx]

    # compute the regularization path on the training set
    betas = iht_path_log(x_train,y_train,path, tol=tol, tolG=tolG, tolrefit=tolrefit, max_iter=max_iter, max_step=max_step, refit=refit, quiet=quiet, lambdas=lambdas)

    Xbetas = x*betas
    Xb     = zeros(T, n)
    nbetas = size(Xbetas,2)
    errors = zeros(T, nbetas)

    # compute the mean out-of-sample error for the TEST set
    if criterion == "class"
#        errors = vec(sumabs2(broadcast(-, round(y[test_idx]), round(logistic(x[test_idx,:] * betas))), 1)) ./ length(test_idx)
        for i = 1:nbetas
            # pull correct x*b
            update_col!(Xb,Xbetas,i,n=n,p=nbetas)

            # need logistic probabilities
            logistic!(lxb, Xb, n=n)

            # mask data from training set
            # training set consists of data NOT in fold
            mask!(lxb, test_idx, 0, zero(T), n=n)

            # compute misclassification error
            errors[i] = mce(lxb, y, test_idx, n=n, mn=test_size)
        end
    else # else -> criterion == "deviance"
        # use k = 0, lambda = 0.0, sortidx = falses(p) to ensure that regularizer is not included in deviance
#        errors[i] = 2.0*logistic_loglik(Xb,y,b,falses(p),test_idx,0,0.0,0, n=n, mn=test_size)
#        errors[i] = 2.0*logistic_loglik(Xb,y,b,indices,test_idx,0,lambda,k, n=n, mn=test_size)
#        errors[i] = 2.0*logistic_loglik(Xb[:,i],y,full(betas[:,i]),indices,test_idx,0,0.0,0, n=n, mn=test_size)
        for i = 1:nbetas
            update_col!(Xb,Xbetas,i,n=n,p=nbetas)
#            mask!(Xb, test_idx, 0, zero(T), n=n)
            Xb[test_idx .== 0] = zero(T)
            errors[i] = - 1 / (2*n) * ( dot(y,Xb) - sum(log(1.0 + exp(Xb))) )# + 0.5*lambda*sumabs2(b[indices])
#            errors[i] = 2.0*logistic_loglik(Xb,y,b,notrues,test_idx,0,zero(T),0, n=n)
        end
    end
    return errors
end


function one_fold_log{T <: Float}(
    x         :: BEDFile,
    y         :: DenseVector{T},
    path      :: DenseVector{Int},
    folds     :: DenseVector{Int},
    fold      :: Int;
    n         :: Int              = length(y),
    p         :: Int              = size(x,2),
    lambdas   :: DenseVector{T}   = ones(length(path)) * convert(T, sqrt(log(p) / n)),
    pids      :: DenseVector{Int} = procs(),
    means     :: DenseVector{T}   = mean(T,x, shared=true, pids=pids),
    invstds   :: DenseVector{T}   = invstd(x,means, shared=true, pids=pids),
    criterion :: ASCIIString      = "deviance",
    tol       :: Float            = convert(T, 1e-6),
    tolG      :: Float            = convert(T, 1e-3),
    tolrefit  :: Float            = convert(T, 1e-6),
    max_iter  :: Int              = 100,
    max_step  :: Int              = 100,
    refit     :: Bool             = true,
    quiet     :: Bool             = true,
)

    # ensure that crossvalidation criterion is valid
    criterion in ["deviance", "class"] || throw(ArgumentError("Argument criterion must be 'deviance' or 'class'"))

    # for deviance, will need an extra vector of "falses"
    # this is used when computing deviances with logistic_loglik()
    if criterion == "deviance"
        notrues = falses(p)
    end

    # make vector of indices for folds
    test_idx = folds .== fold
    test_size = sum(test_idx)

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # allocate the arrays for the training set
    train_idx = convert(Vector{Int}, train_idx)
    test_idx  = convert(Vector{Int}, train_idx)

    # compute the regularization path on the training set
    betas = iht_path_log(x,y,path, tol=tol, tolG=tolG, tolrefit=tolrefit, max_iter=max_iter, max_step=max_step, refit=refit, quiet=quiet, mask_n=train_idx, pids=pids, means=means, invstds=invstds, lambdas=lambdas)

    # tidy up
    gc()

    # preallocate vector for output
    errors = zeros(T, length(path))

    # problem dimensions?
    n = length(folds)
    p = size(x,2)

    # allocate an index vector for b
    indices = falses(p)

    # allocate temporary arrays for the test set
    Xb  = SharedArray(T, n, pids=pids)
    b   = SharedArray(T, p, pids=pids)
    lxb = SharedArray(T, n, pids=pids)

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
        xb!(Xb,x,b,indices,path[i],test_idx, means=means, invstds=invstds, pids=pids)

        # compute out-of-sample error as misclassification (L1 norm) averaged over size of test set
        if criterion == "class"

            # compute estimated responses
            logistic!(lxb,Xb,n=n)

            # mask data from training set
            # training set consists of data NOT in fold
            mask!(lxb, test_idx, 0, zero(T), n=n)
#            lxb[test_idx .== 0] = zero(T)

            # compute misclassification error
            # errors[i] = sum(abs(lxb[test_idx .== 1] - y[test_idx .== 1])) / test_size
            errors[i] = mce(lxb, y, test_idx, n=n, mn=test_size)
        else # else -> criterion == "deviance"

            # use k = 0, lambda = 0.0, sortidx = falses(p) to ensure that regularizer is not included in deviance
            errors[i] = logistic_loglik(Xb,y,b,notrues,test_idx,0,zero(T),0, n=n, mn=mn) / 2
#            errors[i] = 2.0*logistic_loglik(Xb,y,b,indices,test_idx,0,lambda,k, n=n, mn=mn)
#            errors[i] = -2.0 / n * ( dot(y[test_idx .== 1],Xb[test_idx .== 1]) - sum(log(1.0 + exp(Xb[test_idx .== 1]))) ) # + 0.5*lambda*sumabs2(b[indices])
        end

    end

    return errors
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
- `deviance` is an `ASCIIString` that specifies the crossvalidation criterion, either `"deviance"` for the logistic loglikelihood or `"class"` for misclassification error. Defaults to `"deviance"`.

Output:

- `errors` is the averaged MSE over all folds.

If called with `refit = true`, then the output also includes, for best model size `k_star`:

- `b`, a vector of `k_star` floats
- `bidx`, a vector of `k_star` indices indicating the support of the best model.
"""
function cv_log{T <: Float}(
    x         :: DenseMatrix{T},
    y         :: DenseVector{T},
    path      :: DenseVector{Int},
    q         :: Int;
    pids      :: DenseVector{Int} = procs(),
    n         :: Int              = length(y),
    p         :: Int              = size(x,2),
    lambdas   :: SharedVector{T}  = SharedArray(T, (length(path),), pids=pids, init = S -> S[localindexes(S)] = sqrt(log(p) / n)),
    folds     :: DenseVector{Int} = cv_get_folds(n,q),
    criterion :: ASCIIString      = "deviance",
    tol       :: Float            = convert(T, 1e-6),
    tolG      :: Float            = convert(T, 1e-3),
    tolrefit  :: Float            = convert(T, 1e-6),
    max_iter  :: Int              = 100,
    max_step  :: Int              = 100,
    quiet     :: Bool             = true,
    refit     :: Bool             = true
)
    # ensure that crossvalidation criterion is valid
    criterion in ["deviance", "class"] || throw(ArgumentError("Argument criterion must be 'deviance' or 'class'"))

    # how many elements are in the path?
    num_models = length(path)

    # compute crossvalidation deviances
    errors = pfold_log(x, y, path, folds, q, n=n, p=p, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids, tolG=tolG, tolrefit=tolrefit, refit=refit, criterion=criterion, lambdas=lambdas, pids=pids)

    # what is the best model size?
    k = convert(Int, floor(mean(path[errors .== minimum(errors)])))

    # print results
    if !quiet
        println("\n\nCrossvalidation Results:")
        println("k\tError")
        @inbounds for i = 1:num_models
            println(path[i], "\t", errors[i])
        end
        println("\nThe lowest crossvalidation error is achieved at k = ", k)
    end

    # recompute ideal model
    if refit

        # initialize parameter vector
        b = zeros(T, p)

        # get lambda value for best model
        lambda = lambdas[path .== k][1]

        # use L0_log to extract model
        # with refit = true, L0_log will continuously refit predictors
        # no final refitting code necessary
        output = L0_log(x,y,k, b=b, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, tolG=tolG, tolrefit=tolrefit, refit=refit, lambda=lambda)

        # which components of beta are nonzero?
        bidx = find( x -> x .!= zero(T), b)

        return errors, b[bidx], bidx
    end

    return errors
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
    x         :: SharedMatrix{T},
    y         :: SharedVector{T},
    path      :: SharedVector{Int},
    folds     :: SharedVector{Int},
    numfolds  :: Int;
    n         :: Int              = length(y),
    p         :: Int              = size(x,2),
    pids      :: DenseVector{Int} = procs(),
    lambdas   :: SharedVector{T}  = SharedArray(T, (length(path),), pids=pids, init = S -> S[localindexes(S)] = sqrt(log(p) / n)), # type stable for Float32?
    criterion :: ASCIIString      = "deviance",
    tol       :: Float            = convert(T, 1e-6),
    tolG      :: Float            = convert(T, 1e-3),
    tolrefit  :: Float            = convert(T, 1e-6),
    max_iter  :: Int              = 100,
    max_step  :: Int              = 100,
    quiet     :: Bool             = true,
    refit     :: Bool             = true,
    header    :: Bool             = false
)
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
    results = cell(numfolds)

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
                        current_fold > numfolds && break

                        # report distribution of fold to worker and device
                        quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker.\n\n")

                        # launch job on worker
                        # worker loads data from file paths and then computes the errors in one fold
#                        sendto([worker], criterion=criterion, max_iter=max_iter, max_step=max_step)
                        results[current_fold] = remotecall_fetch(worker) do
                                one_fold_log(x, y, path, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, tolG=tolG, tolrefit=tolrefit, refit=refit, criterion=criterion, lambdas=lambdas)
                        end # end remotecall_fetch()
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return reduce(+, results[1], results) ./ numfolds
end


"""
    pfold_log(xfile, xtfile, x2file, yfile, meanfile, invstdfile, path, folds, numfolds[, pids=procs()])

This function is the parallel execution kernel in `cv_log()`. It is not meant to be called outside of `cv_iht()`.
It will distribute `numfolds` crossvalidation folds across the processes supplied by the optional argument `pids` and call `one_fold_log()` for each fold.
Each fold will compute a regularization path given by `path`.
`pfold()` collects the vectors of MSEs returned by calling `one_fold_log()` for each process, reduces them, and returns their average across all folds.
"""
function pfold_log(
    T          :: Type,
    xfile      :: ASCIIString,
    xtfile     :: ASCIIString,
    x2file     :: ASCIIString,
    yfile      :: ASCIIString,
    meanfile   :: ASCIIString,
    invstdfile :: ASCIIString,
    path       :: DenseVector{Int},
    folds      :: DenseVector{Int},
    numfolds   :: Int;
    n          :: Int              = length(y),
    p          :: Int              = size(x,2),
    pids       :: DenseVector{Int} = procs(),
    lambdas    :: DenseVector{T}   = SharedArray(T, (length(path),), pids=pids, init = S -> S[localindexes(S)] = sqrt(log(p) / n)), # type stable for Float32?
    criterion  :: ASCIIString      = "deviance",
    tol        :: Float            = convert(T, 1e-6),
    tolG       :: Float            = convert(T, 1e-3),
    tolrefit   :: Float            = convert(T, 1e-6),
    max_iter   :: Int              = 100,
    max_step   :: Int              = 100,
    quiet      :: Bool             = true,
    refit      :: Bool             = true,
    header     :: Bool             = false
)
    # ensure that crossvalidation criterion is valid
    criterion in ["deviance", "class"] || throw(ArgumentError("Argument criterion must be 'deviance' or 'class'"))
    T <: Float || throw(ArgumentError("Argument T must be either Float32 or T"))

    # how many CPU processes can pfold use?
    np = length(pids)

    # report on CPU processes
    quiet || println("pfold: np = ", np)
    quiet || println("pids = ", pids)

    # set up function to share state (indices of folds)
    i = 1
    nextidx() = (idx=i; i+=1; idx)

    # preallocate cell array for results
    results = cell(numfolds)

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
                        current_fold > numfolds && break

                        # report distribution of fold to worker and device
                        quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker.\n\n")

                        # launch job on worker
                        # worker loads data from file paths and then computes the errors in one fold
                        results[current_fold] = remotecall_fetch(worker) do
                                pids = [worker]
                                x = BEDFile(T, xfile, xtfile, x2file, pids=pids, header=header)
                                n = x.n
                                p = size(x,2)
                                y = SharedArray(abspath(yfile), T, (n,), pids=pids)
                                means = SharedArray(abspath(meanfile), T, (p,), pids=pids)
                                invstds = SharedArray(abspath(invstdfile), T, (p,), pids=pids)

                                one_fold_log(x, y, path, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, means=means, invstds=invstds, pids=pids, tol=tol, tolG=tolG, tolrefit=tolrefit, refit=refit, criterion=criterion, lambdas=lambdas)
                        end # end remotecall_fetch()
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return reduce(+, results[1], results) ./ numfolds
end

# default type for pfold_log is Float64
pfold_log(xfile::ASCIIString, xtfile::ASCIIString, x2file::ASCIIString, yfile::ASCIIString, meanfile::ASCIIString, invstdfile::ASCIIString, path::DenseVector{Int}, folds::DenseVector{Int}, numfolds::Int; n::Int=length(y), p::Int=size(x,2), pids::DenseVector{Int}=procs(), lambdas::DenseVector{Float64}=SharedArray(Float64, (length(path),), pids=pids, init = S -> S[localindexes(S)] = sqrt(log(p) / n)), criterion::ASCIIString="deviance", tol::Float64=1e-6, tolG::Float64=1e-3, tolrefit::Float64=1e-6, max_iter::Int=100, max_step::Int=100, quiet::Bool=true, refit::Bool=true, header::Bool=false)
= pfold_log(Float64, xfile, xtfile, x2file, yfile, meanfile, invstdfile, path, folds, numfolds, n=n, p=p, pids=pids, lambdas=lambdas, criterion=criterion, tol=tol, tolG=tolG, tolrefit=tolrefit, max_iter=max_iter, max_step=max_step, quiet=quiet, refit=refit, header=header)

"""
    cv_log(xfile,xtfile,x2file,yfile,meanfile,invstdfile,path,kernfile,folds,numfolds [, pids=procs()])

This variant of `cv_log()` performs `numfolds` crossvalidation with a `BEDFile` object loaded by `xfile`, `xtfile`, and `x2file`,
with column means stored in `meanfile` and column precisions stored in `invstdfile`.
The continuous response is stored in `yfile` with data particioned by the `Int` vector `folds`.
The folds are distributed across the processes given by `pids`.
"""
function cv_log(
    T             :: Type,
    xfile         :: ASCIIString,
    xtfile        :: ASCIIString,
    x2file        :: ASCIIString,
    yfile         :: ASCIIString,
    meanfile      :: ASCIIString,
    invstdfile    :: ASCIIString,
    path          :: DenseVector{Int},
    folds         :: DenseVector{Int},
    numfolds      :: Int;
    pids          :: DenseVector{Int}   = procs(),
    lambdas       :: DenseVector{Float} = SharedArray(Float, (length(path),), pids=pids, init = S -> S[localindexes(S)] = one(T)),
    criterion     :: ASCIIString        = "deviance",
    tol           :: Float              = convert(T, 1e-6),
    tolG          :: Float              = convert(T, 1e-3),
    tolrefit      :: Float              = convert(T, 1e-6),
    max_iter      :: Int                = 100,
    max_step      :: Int                = 100,
    quiet         :: Bool               = true,
    refit         :: Bool               = true,
    header        :: Bool               = false
)
    # ensure that crossvalidation criterion is valid
    criterion in ["deviance", "class"] || throw(ArgumentError("Argument criterion must be 'deviance' or 'class'"))
    T <: Float || throw(ArgumentError("Argument T must be either Float32 or T"))

    # how many elements are in the path?
    num_models = length(path)

    # want to compute a path for each fold
    # the folds are computed asynchronously
    # only use the worker processes
    errors = pfold_log(xfile, xtfile, x2file, yfile, meanfile, invstdfile, path, folds, numfolds, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids, header=header, tolG=tolG, tolrefit=tolrefit, refit=refit, criterion=criterion, lambdas=lambdas)

    # what is the best model size?
    k = convert(Int, floor(mean(path[errors .== minimum(errors)])))

    # print results
    if !quiet
        println("\n\nCrossvalidation Results:")
        println("k\tMCE")
        @inbounds for i = 1:num_models
            println(path[i], "\t", errors[i])
        end
        println("\nThe lowest MCE is achieved at k = ", k)
    end

    # recompute ideal model
    if refit

        # load data on *all* processes
        x       = BEDFile(xfile, xtfile, x2file, header=header, pids=pids)
        n       = x.n
        p       = size(x,2)
        y       = SharedArray(abspath(yfile), T, (n,), pids=pids)
        means   = SharedArray(abspath(meanfile), T, (p,), pids=pids)
        invstds = SharedArray(abspath(invstdfile), T, (p,), pids=pids)

        # initialize parameter vector as SharedArray
        b = SharedArray(T, p)

        # get lambda value for best model
        lambda = lambdas[path .== k][1]

        # use L0_reg to extract model
        output = L0_log(x,y,k,n=n, p=p, b=b, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, means=means, invstds=invstds, tolG=tolG, tolrefit=tolrefit, refit=refit, lambda=lambda)

        # which components of beta are nonzero?
        inferred_model = b .!= zero(T)
        bidx = find( x -> x .!= zero(T), b)

        return errors, b[bidx], bidx
    end
    return errors
end

cv_log(xfile::ASCIIString, xtfile::ASCIIString, x2file::ASCIIString, yfile::ASCIIString, meanfile::ASCIIString, invstdfile::ASCIIString, path::DenseVector{Int}, folds::DenseVector{Int}, numfolds::Int; pids::DenseVector{Int}=procs(), lambdas::DenseVector{Float64} = SharedArray(Float, (length(path),), pids=pids, init = S -> S[localindexes(S)] = one(Float64)),  criterion::ASCIIString="deviance", tol::Float64=1e-6, tolG::Float64=1e-3, tolrefit::Float64=1e-6, max_iter::Int=100, max_step::Int=100, quiet::Bool=true, refit::Bool=true, header::Bool=false)
=cv_log(Float64, xfile, xtfile, x2file, yfile, meanfile, invstdfile, path, folds, numfolds, pids=pids, lambdas=lambdas, criterion=criterion, tol=tol, tolG=tolG, tolrefit=tolrefit, max_iter=max_iter, max_step=max_step, quiet=quiet, refit=refit, header=header)
