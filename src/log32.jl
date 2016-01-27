function L0_log(
    x        :: DenseMatrix{Float32},
    y        :: DenseVector{Float32},
    k        :: Int;
    n        :: Int                  = length(y),
    p        :: Int                  = size(x,2),
    xk       :: DenseMatrix{Float32} = zeros(Float32, n,k),
    xk2      :: DenseMatrix{Float32} = zeros(Float32, n,k),
    d2b      :: DenseMatrix{Float32} = zeros(Float32, k,k),
    b        :: DenseVector{Float32} = zeros(Float32, p),
    b0       :: DenseVector{Float32} = zeros(Float32, p),
    df       :: DenseVector{Float32} = zeros(Float32, p),
    Xb       :: DenseVector{Float32} = zeros(Float32, n),
    lxb      :: DenseVector{Float32} = zeros(Float32, n),
    l2xb     :: DenseVector{Float32} = zeros(Float32, n),
    bk       :: DenseVector{Float32} = zeros(Float32, k),
    bk2      :: DenseVector{Float32} = zeros(Float32, k),
    bk0      :: DenseVector{Float32} = zeros(Float32, k),
    ntb      :: DenseVector{Float32} = zeros(Float32, k),
    db       :: DenseVector{Float32} = zeros(Float32, k),
    dfk      :: DenseVector{Float32} = zeros(Float32, k),
    active   :: DenseVector{Int}     = collect(1:p),
    bidxs    :: DenseVector{Int}     = collect(1:p),
    dfidxs   :: DenseVector{Int}     = collect(1:p),
    idxs     :: BitArray{1}          = falses(p),
    idxs0    :: BitArray{1}          = falses(p),
    lambda   :: Float32              = sqrt(log(p)/n),
    mu       :: Float32              = one(Float32),
    tol      :: Float32              = 1f-6,
    tolG     :: Float32              = 1f-3,
    tolrefit :: Float32              = 1f-6,
    max_iter :: Int                  = 100,
    max_step :: Int                  = 50,
    refit    :: Bool                 = true,
    quiet    :: Bool                 = true,
)

    # start timer
    tic()

    # check arguments
    n        == size(x,1)     || throw(ArgumentError("Length n = $n of response vector y does not match number of rows = $(size(x,1)) in x"))
    k        >  p             && throw(ArgumentError("Value of argument k = $k exceeds number of predictors p = $p"))
    lambda   <  zero(Float32) && throw(ArgumentError("Value of argument lambda = $lambda must be nonnegative"))
    mu       >  zero(Float32) || throw(ArgumentError("Value of argument mu must be positive"))
    tol      >  eps(Float32)  || throw(ArgumentError("Value of argument tol must exceed machine precision"))
    tolG     >  eps(Float32)  || throw(ArgumentError("Value of argument tolG must exceed machine precision"))
    tolrefit >  eps(Float32)  || throw(ArgumentError("Value of argument tolrefit must exceed machine precision"))
    max_iter >= 0             || throw(ArgumentError("Value of max_iter must be nonnegative\n"))
    max_step >= 0             || throw(ArgumentError("Value of max_step must be nonnegative\n"))

    # initialize return values
    mm_iter   = 0              # number of iterations of L0_reg
    mm_time   = zero(Float32)  # compute time *within* L0_reg
    loss      = Inf            # loss function value

    # initialize algorithm parameters
    num_df    = min(p,3*k)     # largest 3*k active components of gradient
    short_df  = min(p,2*k)     # largest 2*k active components of gradient
    converged = false          # is algorithm converged?
    stuck     = false          # is algorithm stuck in a cycle?
    normdf    = Inf            # norm of active portion of gradient
    loss0     = Inf            # penultimate loss function value
    loss00    = Inf            # antepenultimate loss function value
    bktrk     = 0              # number of Newton backtracking steps
    nt_iter   = 0              # number of Newton iterations
    lt        = length(active) # size of current active set?


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
                bk2, nt_iter, bktrk = fit_logistic(xk, y, zero(Float32), n=n, p=k, d2b=d2b, x2=xk2, b=bk, b0=bk0, ntb=ntb, db=db, Xb=Xb, lxb=lxb, l2xb=l2xb, tol=tolrefit, max_iter=max_step, quiet=true)
                b[idxs] = bk2
            end

            # stop timer
            mm_time = toq()

            # these are output variables for function
            # wrap them into a Dict and return
            output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => loss, "iter" => mm_iter, "beta" => copy(b), "active" => active)

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
        if all(b .== zero(Float32))
            fill!(Xb,zero(Float32))
        else
#            update_xb!(Xb,x,b,active,lt,p=p,n=n)
            update_xb!(Xb,x,b,bidxs,k,p=p,n=n)
        end

        # recompute active loss = (-dot(y,Xb) + sum(log(1.0 + exp(Xb)))) / n + 0.5*lambda*sumabs2(b[active])
        # special case: b = 0 --> Xb = 0 --> loss = n*log(1 + exp(0))/n + 0.5*lambda*norm(0)
        if all(Xb .== zero(Float32))
            loss = log(one(Float32) + one(Float32))
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
                bk2,nt_iter,bktrk = fit_logistic(xk, y, zero(Float32), n=n, p=k, d2b=d2b, x2=xk2, b=bk, b0=bk0, ntb=ntb, db=db, Xb=Xb, lxb=lxb, l2xb=l2xb, tol=tolrefit, max_iter=max_step, quiet=true)
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
            output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => loss, "iter" => mm_iter, "beta" => copy(b), "active" => active)

            return output
        end # end convergence check
    end # end main GraSP iterations
end # end L0_log


function L0_log(
    x        :: BEDFile,
    y        :: SharedVector{Float32},
    k        :: Int;
    n        :: Int                   = length(y),
    p        :: Int                   = size(x,2),
    pids     :: DenseVector{Int}      = procs(),
    means    :: SharedVector{Float32} = mean(Float32,x, shared=true, pids=pids),
    invstds  :: SharedVector{Float32} = invstd(x,means, shared=true, pids=pids),
    xk       :: DenseMatrix{Float32}  = zeros(Float32, n,k),
    xk2      :: DenseMatrix{Float32}  = zeros(Float32, n,k),
    d2b      :: DenseMatrix{Float32}  = zeros(Float32, k,k),
    b        :: SharedVector{Float32} = SharedArray(Float32, p, pids=pids),
    b0       :: SharedVector{Float32} = SharedArray(Float32, p, pids=pids),
    df       :: SharedVector{Float32} = SharedArray(Float32, p, pids=pids),
    Xb       :: SharedVector{Float32} = SharedArray(Float32, n, pids=pids),
    lxb      :: SharedVector{Float32} = SharedArray(Float32, n, pids=pids),
    l2xb     :: DenseVector{Float32}  = zeros(Float32, n),
    bk       :: DenseVector{Float32}  = zeros(Float32, k),
    bk2      :: DenseVector{Float32}  = zeros(Float32, k),
    bk0      :: DenseVector{Float32}  = zeros(Float32, k),
    ntb      :: DenseVector{Float32}  = zeros(Float32, k),
    db       :: DenseVector{Float32}  = zeros(Float32, k),
    dfk      :: DenseVector{Float32}  = zeros(Float32, k),
    active   :: DenseVector{Int}      = collect(1:p),
    bidxs    :: DenseVector{Int}      = collect(1:p),
    dfidxs   :: DenseVector{Int}      = collect(1:p),
    mask_n   :: DenseVector{Int}      = ones(Int, n),
    idxs     :: BitArray{1}           = falses(p),
    idxs2    :: BitArray{1}           = falses(p),
    idxs0    :: BitArray{1}           = falses(p),
    lambda   :: Float32               = sqrt(log(p)/n),
    mu       :: Float32               = one(Float32),
    tol      :: Float32               = 1f-6,
    tolG     :: Float32               = 1f-3,
    tolrefit :: Float32               = 1f-6,
    max_iter :: Int                   = 100,
    max_step :: Int                   = 50,
    mn       :: Int                   = sum(mask_n),
    refit    :: Bool                  = true,
    quiet    :: Bool                  = true,
)

    # start timer
    tic()

    # check arguments
    n        == size(x,1)     || throw(ArgumentError("Length n = $n of response vector y does not match number of rows = $(size(x,1)) in x"))
    k        >  p             && throw(ArgumentError("Value of argument k = $k exceeds number of predictors p = $p"))
    lambda   <  zero(Float32) && throw(ArgumentError("Value of argument lambda = $lambda must be nonnegative"))
    mu       >  zero(Float32) || throw(ArgumentError("Value of argument mu must be positive"))
    tol      >  eps(Float32)  || throw(ArgumentError("Value of argument tol must exceed machine precision"))
    tolG     >  eps(Float32)  || throw(ArgumentError("Value of argument tolG must exceed machine precision"))
    tolrefit >  eps(Float32)  || throw(ArgumentError("Value of argument tolrefit must exceed machine precision"))
    max_iter >= 0             || throw(ArgumentError("Value of max_iter must be nonnegative\n"))
    max_step >= 0             || throw(ArgumentError("Value of max_step must be nonnegative\n"))

    # enforce all 0/1 in mask_n
    sum((mask_n .== 1) $ (mask_n .== 0)) == n || throw(ArgumentError("Argument mask_n can only contain 1s and 0s"))

    # initialize return values
    mm_iter   = 0              # number of iterations of L0_reg
    mm_time   = zero(Float32)  # compute time *within* L0_reg
    loss      = Inf            # loss function value

    # initialize algorithm parameters
    num_df    = min(p,3*k)     # largest 3*k active components of gradient
    short_df  = min(p,2*k)     # largest 2*k active components of gradient
    converged = false          # is algorithm converged?
    normdf    = Inf            # norm of active portion of gradient
    loss0     = Inf            # previous loss function value
    bktrk     = 0              # track loss and number of backtracking steps
    nt_iter   = 0              # number of Newton iterations
    lt        = length(active) # size of current active set?

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
                bk2,bktrk = fit_logistic(xk, y, mask_n, zero(Float32), n=n, p=k, d2b=d2b, x2=xk2, b=bk, b0=bk0, ntb=ntb, db=db, Xb=Xb, lxb=lxb, l2xb=l2xb, tol=tolrefit, max_iter=max_step, quiet=true)
                b[idxs] = bk2
            end

            # stop timer
            mm_time = toq()

            # these are output variables for function
            # wrap them into a Dict and return
            output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => loss, "iter" => mm_iter, "beta" => copy(b), "active" => active)

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
        if all(b .== zero(Float32))
            fill!(Xb,zero(Float32))
        else
            fill!(idxs2,false)
            idxs2[active] = true
            xb!(Xb,x,b,idxs2,sum(idxs2),mask_n, means=means, invstds=invstds, pids=pids)
        end

        # recompute active loss = (-dot(y,Xb) + sum(log(1.0 + exp(Xb)))) / n + 0.5*lambda*sumabs2(b[active])
        # special case: b = 0 --> Xb = 0 --> loss = n*log(1 + exp(0))/n + 0.5*lambda*norm(0)
        if all(Xb .== zero(Float32))
            loss = log(one(Float32) + one(Float32))
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
                bk2, nt_iter, bktrk = fit_logistic(xk, y, zero(Float32), n=n, p=k, d2b=d2b, x2=xk2, b=bk, b0=bk0, ntb=ntb, db=db, Xb=Xb, lxb=lxb, l2xb=l2xb, tol=tolrefit, max_iter=max_step, quiet=true)
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
            output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => loss, "iter" => mm_iter, "beta" => copy(b), "active" => active)

            return output
        end # end convergence check
    end # end main GraSP iterations
end # end L0_log



function iht_path_log(
    x        :: DenseMatrix{Float32},
    y        :: DenseVector{Float32},
    path     :: DenseVector{Int};
    n        :: Int     = length(y),
    p        :: Int     = size(x,2),
    lambdas  :: DenseVector{Float32} = ones(length(path)) * sqrt(log(p) / n),
    tol      :: Float32 = 1f-6,
    tolG     :: Float32 = 1f-3,
    tolrefit :: Float32 = 1f-6,
    max_iter :: Int     = 100,
    max_step :: Int     = 50,
    refit    :: Bool    = true,
    quiet    :: Bool    = true,
)

#    # size of problem?
#    (n,p) = size(x)

    # how many models will we compute?
    num_models = length(path)

    # preallocate space for intermediate steps of algorithm calculations
    b      = zeros(Float32, size(x,2))     # statistical model beta
    b0     = zeros(Float32, p)             # previous iterate beta0
    df     = zeros(Float32, p)             # (negative) gradient
    Xb     = zeros(Float32, n)             # x*b
    lxb    = zeros(Float32, n)             # logistic(x*b) = 1 ./ (1 + exp(-x*b))
    l2xb   = zeros(Float32, n)             # lxb * (1 - lxb)
    bidxs  = collect(1:p)                  # indices that sort b
    dfidxs = collect(1:p)                  # indices that sort df
    active = collect(1:p)                  # union of active subsets of b and df
    idxs   = falses(p)                     # nonzero components of b
    idxs0  = falses(p)                     # store previous nonzero indicators for b
    betas  = spzeros(Float32,p,num_models) # a matrix to store calculated models

    # compute the path
    for i = 1:num_models

        # model size?
        q = path[i]
        quiet || println("Current model size: $q")

        # store projection of beta onto largest k nonzeroes in magnitude
        bk     = zeros(Float32,q)

        # current regularization parameter?
        lambda = lambdas[i]

        # these arrays change in size from iteration to iteration
        # we must allocate them for every new model size
        xk     = zeros(Float32, n, q)  # store q columns of x for refitting
        xk2    = zeros(Float32, n, q)  # copy of xk also used in refitting
        d2b    = zeros(Float32, q, q)  # Hessian of q active components of b
        bk0    = zeros(Float32, q)    # copy of bk used in refitting
        ntb    = zeros(Float32, q)    # Newton step for bk used in refitting
        db     = zeros(Float32, q)    # gradient of bk used in refitting
        dfk    = zeros(Float32, q)    # size q subset of df used in refitting

        # now compute current model
        output = L0_log(x,y,q, n=n, p=p, b=b, tol=tol, tolG=tolG, tolrefit=tolrefit, refit=refit, max_iter=max_iter, max_step=max_step, quiet=quiet, b0=b0, df=df, Xb=Xb, lxb=lxb, l2xb=l2xb, bidxs=bidxs, dfidxs=dfidxs, active=active, idxs=idxs, idxs0=idxs0, bk=bk, xk=xk, xk2=xk2, d2b=d2b, bk0=bk0, ntb=ntb, db=db, dfk=dfk, lambda=lambda)

        # extract and save model
        copy!(b, output["beta"])
        active = output["active"]
        betas[:,i] = sparsevec(b)

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

function iht_path_log(
    x        :: BEDFile,
    y        :: SharedVector{Float32},
    path     :: DenseVector{Int};
    pids     :: DenseVector{Int}      = procs(),
    means    :: SharedVector{Float32} = mean(Float32,x, shared=true, pids=pids),
    invstds  :: SharedVector{Float32} = invstd(x,means, shared=true, pids=pids),
    mask_n   :: DenseVector{Int}      = ones(Int,length(y)),
    n        :: Int                   = length(y),
    p        :: Int                   = size(x,2),
    lambdas  :: DenseVector{Float32}  = ones(length(path)) * sqrt(log(p) / n),
    tol      :: Float32               = 1f-6,
    tolG     :: Float32               = 1f-3,
    tolrefit :: Float32               = 1f-6,
    max_iter :: Int                   = 100,
    max_step :: Int                   = 50,
    refit    :: Bool                  = true,
    quiet    :: Bool                  = true,
)

#    # size of problem?
#    n = length(y)
#    p = size(x,2)

    # how many models will we compute?
    num_models = length(path)

    # preallocate space for intermediate steps of algorithm calculations
    b      = SharedArray(Float32, p, pids=pids)  # statistical model
    b0     = SharedArray(Float32, p, pids=pids)  # previous iterate beta0
    df     = SharedArray(Float32, p, pids=pids)  # (negative) gradient
    Xb     = SharedArray(Float32, n, pids=pids)  # x*b
    lxb    = SharedArray(Float32, n, pids=pids)  # logistic(x*b) = 1 ./ (1 + exp(-x*b))
    l2xb   = zeros(Float32,n)                    # lxb * (1 - lxb)
    bidxs  = collect(1:p)                        # indices that sort b
    dfidxs = collect(1:p)                        # indices that sort df
    active = collect(1:p)                        # union of active subsets of b and df
    idxs   = falses(p)                           # nonzero components of b
    idxs2  = falses(p)                           # nonzero components of b
    idxs0  = falses(p)                           # store previous nonzero indicators for b
    betas  = spzeros(Float32,p,num_models)       # a matrix to store calculated models

    # compute the path
    for i = 1:num_models

        # model size?
        q = path[i]

        # store projection of beta onto largest k nonzeroes in magnitude
        bk     = zeros(Float32,q)

        # current regularization parameter?
        lambda = lambdas[i]

        # these arrays change in size from iteration to iteration
        # we must allocate them for every new model size
        xk     = zeros(Float32, n, q)  # store q columns of x for refitting
        xk2    = zeros(Float32, n, q)  # copy of xk also used in refitting
        d2b    = zeros(Float32, q, q)  # Hessian of q active components of b
        bk0    = zeros(Float32, q)    # copy of bk used in refitting
        ntb    = zeros(Float32, q)    # Newton step for bk used in refitting
        db     = zeros(Float32, q)    # gradient of bk used in refitting
        dfk    = zeros(Float32, q)    # size q subset of df used in refitting

        # now compute current model
        output = L0_log(x,y,q, n=n, p=p, b=b, tol=tol, tolG=tolG, tolrefit=tolrefit, refit=refit, max_iter=max_iter, max_step=max_step, quiet=quiet, b0=b0, df=df, Xb=Xb, lxb=lxb, l2xb=l2xb, bidxs=bidxs, dfidxs=dfidxs, active=active, idxs=idxs, idxs0=idxs0, bk=bk, xk=xk, xk2=xk2, d2b=d2b, bk0=bk0, ntb=ntb, db=db, dfk=dfk, means=means, invstds=invstds, idxs2=idxs2, mask_n=mask_n, lambda=lambda)

        # extract and save model
        copy!(b, output["beta"])
        active = output["active"]
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


function one_fold_log(
    x         :: DenseMatrix{Float32},
    y         :: DenseVector{Float32},
    path      :: DenseVector{Int},
    folds     :: DenseVector{Int},
    fold      :: Int;
    n         :: Int                   = length(y),
    p         :: Int                   = size(x,2),
    lambdas   :: DenseVector{Float32}  = ones(length(path)) * sqrt(log(p) / n),
    criterion :: ASCIIString = "deviance",
    tol       :: Float32     = 1f-6,
    tolG      :: Float32     = 1f-3,
    tolrefit  :: Float32     = 1f-6,
    max_iter  :: Int         = 100,
    max_step  :: Int         = 50,
    refit     :: Bool        = true,
    quiet     :: Bool        = true,
)
    # ensure that crossvalidation criterion is valid
    criterion in ["deviance", "class"] || throw(ArgumentError("Argument criterion must be 'deviance' or 'class'"))

    # make vector of indices for folds
    test_idx = folds .== fold
    test_size = sum(test_idx)

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # allocate the arrays for the training set
    x_train = x[train_idx,:]
    y_train = y[train_idx]

    # compute the regularization path on the training set
    betas = iht_path_log(x_train,y_train,path, tol=tol, tolG=tolG, tolrefit=tolrefit, max_iter=max_iter, max_step=max_step, refit=refit, quiet=quiet, lambdas=lambdas)

    # compute the mean out-of-sample error for the TEST set
    errors = vec(sumabs2(broadcast(-, round(y[test_idx]), round(logistic(x[test_idx,:] * betas))), 1)) ./ length(test_idx)

    return errors
end


function one_fold_log(
    x         :: BEDFile,
    y         :: DenseVector{Float32},
    path      :: DenseVector{Int},
    folds     :: DenseVector{Int},
    fold      :: Int;
    n         :: Int                  = length(y),
    p         :: Int                  = size(x,2),
    lambdas   :: DenseVector{Float32} = ones(length(path)) * sqrt(log(p) / n),
    pids      :: DenseVector{Int}     = procs(),
    means     :: DenseVector{Float32} = mean(Float32,x, shared=true, pids=pids),
    invstds   :: DenseVector{Float32} = invstd(x,means, shared=true, pids=pids),
    criterion :: ASCIIString          = "deviance",
    tol       :: Float32              = 1f-6,
    tolG      :: Float32              = 1f-3,
    tolrefit  :: Float32              = 1f-6,
    max_iter  :: Int                  = 100,
    max_step  :: Int                  = 50,
    refit     :: Bool                 = true,
    quiet     :: Bool                 = true,
)

    # ensure that crossvalidation criterion is valid
    criterion in ["deviance", "class"] || throw(ArgumentError("Argument criterion must be 'deviance' or 'class'"))

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
    errors = zeros(Float32, length(path))

    # problem dimensions?
    n = length(folds)
    p = size(x,2)

    # allocate an index vector for b
    indices = falses(p)

    # allocate temporary arrays for the test set
    Xb  = SharedArray(Float32, n, pids=pids)
    b   = SharedArray(Float32, p, pids=pids)
    lxb = SharedArray(Float32, n, pids=pids)

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
        if criteron == "class"

            # compute estimated responses
            logistic!(lxb,Xb,n=n)

            # mask data from training set
            # training set consists of data NOT in fold
            mask!(lxb, test_idx, 0, zero(Float32), n=n)
#            lxb[test_idx .== 0] = zero(Float32)

            # compute misclassification error
#            errors[i] = sum(abs(lxb[test_idx .== 1] - y[test_idx .== 1])) / test_size 
            errors[i] = mce(lxb, y, test_idx, n=n, mn=test_size)
#        elseif criteron == "deviance"
        elseif criteron == "deviance"
            # use k = 0, lambda = 0.0, sortidx = falses(p) to ensure that regularizer is not included in deviance 
#            errors[i] = 2.0*logistic_loglik(Xb,y,b,falses(p),test_idx,0,0.0,0, n=n, mn=mn)
#            errors[i] = 2.0*logistic_loglik(Xb,y,b,indices,test_idx,0,lambda,k, n=n, mn=mn)
            errors[i] = -2.0f0 / n * ( dot(y[test_idx .== 1],Xb[test_idx .== 1]) - log(one(Float32) + exp(Xb[test_idx .== 1])) ) + 0.5f0*lambda*sumabs2(b[indices]) 
        end

    end

    return errors
end


function cv_log(
    x         :: DenseMatrix{Float32},
    y         :: DenseVector{Float32},
    path      :: DenseVector{Int},
    q         :: Int;
    n         :: Int                  = length(y),
    p         :: Int                  = size(x,2),
    lambdas   :: DenseVector{Float32} = ones(length(path)) * sqrt(log(p) / n),
    folds     :: DenseVector{Int} = cv_get_folds(n,q),
    criterion :: ASCIIString      = "deviance",
    tol       :: Float32          = 1f-4,
    tolG      :: Float32          = 1f-3,
    tolrefit  :: Float32          = 1f-6,
    max_iter  :: Int              = 100,
    max_step  :: Int              = 50,
    quiet     :: Bool             = true,
    refit     :: Bool             = true
)
    # ensure that crossvalidation criterion is valid
    criterion in ["deviance", "class"] || throw(ArgumentError("Argument criterion must be 'deviance' or 'class'"))

    # how many elements are in the path?
    num_models = length(path)

    # preallocate vectors used in xval
    myrefs = cell(q)
    errors = zeros(Float32, num_models)    # vector to save mean squared errors

    # want to compute a path for each fold
    # the folds are computed asynchronously
    # the @sync macro ensures that we wait for all of them to finish before proceeding
    @sync for i = 1:q

        quiet || print_with_color(:blue, "spawning fold $i\n")

        # one_fold_log returns a vector of out-of-sample errors (MCE for logistic regression)
        # @spawn(one_fold_log(...)) sends calculation to any available processor and returns RemoteRef to out-of-sample error
        myrefs[i] = @spawn(one_fold_log(x, y, path, folds, i, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, tolG=tolG, tolrefit=tolrefit, refit=refit, criterion=criterion, lambdas=lambdas))
    end

    # average the errors 
    @inbounds for i = 1:q
        errors += fetch(myrefs[i]);
    end
    errors ./= q

    # what is the best model size?
    k = convert(Int, floor(mean(path[errors .== minimum(errors)])))

    # print results
    if !quiet
        println("\n\nCrossvalidation Results:")
        println("k\tError")
        @inbounds for i = 1:num_models
            println(path[i], "\t", errors[i])
        end
        println("\nThe lowest MCE is achieved at k = ", k)
    end

    # recompute ideal model
    if refit

        # initialize parameter vector
        b = zeros(Float32, p)

        # use L0_log to extract model
        # with refit = true, L0_log will continuously refit predictors
        # no final refitting code necessary
        output = L0_log(x,y,k, b=b, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, tolG=tolG, tolrefit=tolrefit, refit=refit)

        # which components of beta are nonzero?
        bidx = find( x -> x .!= zero(Float32), b)

        return errors, b[bidx], bidx
    end

    return errors
end


function pfold_log(
	xfile      :: ASCIIString,
	xtfile     :: ASCIIString,
	x2file     :: ASCIIString,
	yfile      :: ASCIIString,
	meanfile   :: ASCIIString,
	invstdfile :: ASCIIString,
	path       :: DenseVector{Int},
	folds      :: DenseVector{Int},
	numfolds   :: Int;
    n          :: Int                  = length(y),
    p          :: Int                  = size(x,2),
	pids       :: DenseVector{Int}     = procs(),
    lambdas    :: DenseVector{Float32} = SharedArray(Float32, (length(path),), pids=pids, init = S -> S[localindexes(S)] = one(Float32)) * sqrt(log(p) / n),
    criterion  :: ASCIIString      = "deviance",
    tol        :: Float32          = 1f-6,
    tolG       :: Float32          = 1f-3,
    tolrefit   :: Float32          = 1f-6,
	max_iter   :: Int              = 100,
	max_step   :: Int              = 50,
	quiet      :: Bool             = true,
	refit      :: Bool             = true,
	header     :: Bool             = false
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
						results[current_fold] = remotecall_fetch(worker) do
								pids = [worker]
								x = BEDFile(Float32, xfile, xtfile, x2file, pids=pids, header=header)
								n = x.n
								p = size(x,2)
								y = SharedArray(abspath(yfile), Float32, (n,), pids=pids)
								means = SharedArray(abspath(meanfile), Float32, (p,), pids=pids)
								invstds = SharedArray(abspath(invstdfile), Float32, (p,), pids=pids)

								one_fold_log(x, y, path, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, means=means, invstds=invstds, pids=pids, tol=tol, tolG=tolG, tolrefit=tolrefit, refit=refit, criterion=criterion)
						end # end remotecall_fetch()
					end # end while
				end # end @async
			end # end if
		end # end for
	end # end @sync

	# return reduction (row-wise sum) over results
	return reduce(+, results[1], results) ./ numfolds
end


function cv_log(
	xfile         :: ASCIIString,
	xtfile        :: ASCIIString,
	x2file        :: ASCIIString,
	yfile         :: ASCIIString,
	meanfile      :: ASCIIString,
	invstdfile    :: ASCIIString,
	path          :: DenseVector{Int},
	folds         :: DenseVector{Int},
	numfolds      :: Int;
	pids          :: DenseVector{Int} = procs(),
    criterion     :: ASCIIString      = "deviance",
	tol           :: Float32          = 1f-6,
    tolG          :: Float32          = 1f-3,
    tolrefit      :: Float32          = 1f-6,
	max_iter      :: Int              = 100,
	max_step      :: Int              = 50,
	quiet         :: Bool             = true,
	refit         :: Bool             = true,
	header        :: Bool             = false
)
    # ensure that crossvalidation criterion is valid
    criterion in ["deviance", "class"] || throw(ArgumentError("Argument criterion must be 'deviance' or 'class'"))

	# how many elements are in the path?
	num_models = length(path)

	# want to compute a path for each fold
	# the folds are computed asynchronously
	# only use the worker processes
	errors = pfold_log(xfile, xtfile, x2file, yfile, meanfile, invstdfile, path, folds, numfolds, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids, header=header, tolG=tolG, tolrefit=tolrefit, refit=refit, criterion=criterion)

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
		y       = SharedArray(abspath(yfile), Float32, (n,), pids=pids)
		means   = SharedArray(abspath(meanfile), Float32, (p,), pids=pids)
		invstds = SharedArray(abspath(invstdfile), Float32, (p,), pids=pids)

		# initialize parameter vector as SharedArray
		b = SharedArray(Float32, p)

		# use L0_reg to extract model
		output = L0_log(x,y,k,n=n, p=p, b=b, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, means=means, invstds=invstds, tolG=tolG, tolrefit=tolrefit, refit=refit)

		# which components of beta are nonzero?
		inferred_model = b .!= zero(Float32)
		bidx = find( x -> x .!= zero(Float32), b)

		return errors, b[bidx], bidx
	end
	return errors
end
