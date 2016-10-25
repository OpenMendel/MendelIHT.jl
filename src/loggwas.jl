### 17 OCT 2016: MUST RECODE THIS FUNCTION
"""
    fit_logistic(x, y, mask_n, λ)

If called with an `Int` vector `mask_n`, then `fit_logistic()` will refit logistic effect sizes while masking components `y[i]` where `mask_n[i] = 0`.
"""
function fit_logistic{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    mask_n   :: DenseVector{Int},
    λ   :: T;
    n        :: Int    = length(y),
    p        :: Int    = size(x,2),
    mn       :: Int    = sum(mask_n),
    d2b      :: DenseMatrix{T} = zeros(T, p,p),
    x2       :: DenseMatrix{T} = zeros(T, n,p),
    b        :: DenseVector{T} = zeros(T, p),
    b0       :: DenseVector{T} = zeros(T, p),
    ntb      :: DenseVector{T} = zeros(T, p),
    db       :: DenseVector{T} = zeros(T, p),
    xβ       :: DenseVector{T} = zeros(T, n),
    lxβ      :: DenseVector{T} = zeros(T, n),
    l2xb     :: DenseVector{T} = zeros(T, n),
    tol      :: Float = convert(T, 1e-8),
    max_iter :: Int   = 50,
    quiet    :: Bool  = true,
)

    # if b is not warm-started, then ensure that it is not entirely zero
    if all(b .== 0) 
        b[1] = logit(mean(y[mask_n .== 1]))
    end

    # initialize intermediate arrays for calculations
    BLAS.gemv!('N', one(T), x, b, zero(T), xβ)
    log2xb!(lxβ, l2xb, xβ, n=n)
    mask!(lxβ, mask_n, 0, zero(T), n=n)
#    lxβ[mask_n .== 0] = zero(T)
    copy!(b0,b)
    fill!(db, zero(T))

    # track objective
    old_obj = Inf
    new_obj = logistic_loglik(xβ,y,b,mask_n,0,λ,p, n=n)

    # output progress to console
    quiet || println("Iter\tHalves\tObjective")

    i = 0
    bktrk = 0
    # enter loop for Newton's method
    for i = 1:max_iter

        # db = (x'*(lxβ - y)) / n + λ*b
        BLAS.axpy!(n, -one(T), sdata(y), 1, sdata(lxβ), 1)
        mask!(lxβ, mask_n, 0, zero(T), n=n)
#        lxβ[mask_n .== 0] = zero(T)
        BLAS.gemv!('T', one(T), sdata(x), sdata(lxβ), zero(T), sdata(db))
#        BLAS.scal!(p, 1/n, sdata(db), 1)
        BLAS.scal!(p, 1/mn, sdata(db), 1)
        BLAS.axpy!(p, λ, sdata(b), 1, sdata(db), 1)

        # d2b = (x'*diagm(l2xb)*x)/n + λ*I
        # note that log2xb!() already performs division by n on l2xb
        copy!(x2,x)
        mask!(l2xb, mask_n, 0, zero(T), n=n)
#        l2xb[mask_n .== 0] = zero(T)
        BLAS.scal!(p, n/mn, sdata(l2xb), 1) # rescale to number of unmasked samples
        scale!(sdata(l2xb), sdata(x2))
        BLAS.gemm!('T', 'N', one(T), sdata(x), sdata(x2), zero(T), sdata(d2b))
        d2b += λ*I

        # b = b0 - ntb = b0 - inv(d2b)*db
        #   = b0 - inv[ x' diagm(pi) diagm(1 - pi) x + λ*I] [x' (pi - y) + λ*b]
        ntb = d2b\db
        copy!(b,b0)
        BLAS.axpy!(p,-one(T),ntb,1,b,1)

        # compute objective
        new_obj = logistic_loglik(xβ,y,b,mask_n,0,λ,p, n=n)

        # backtrack
        j = 0
        while (new_obj > old_obj + tol) && (j < 50)

            # increment iterator
            j += 1

            # b = b0 - 0.5*ntb
            copy!(b,b0)
#            BLAS.axpy!(p,-(0.5^j),ntb,1,b,1)
            BLAS.axpy!(p,-one(T) / (2^j),ntb,1,b,1)

            # recalculate objective
            new_obj = logistic_loglik(xβ,y,b,mask_n,0,λ,p, n=n)

        end

        # accumulate total backtracking steps
        bktrk += j

        # track distance between iterates
        dist = euclidean(sdata(b),sdata(b0)) / (norm(sdata(b0),2) + one(T))

        # track progress
        quiet || println(i, "\t", j, "\t", dist)

        # check for convergence
        # if converged, then return b
        dist < tol && return b, i, div(bktrk,i)

        # unconverged at this point, so update intermediate arrays
        BLAS.gemv!('N', one(T), sdata(x), sdata(b), zero(T), sdata(xβ))
        log2xb!(lxβ, l2xb, xβ, n=n)

        # save previous beta
        copy!(b0, b)
        old_obj = new_obj
    end

    warn("fit_logistic failed to converge in $(max_iter) iterations, exiting...")
    return b, i, div(bktrk,max_iter)
end


### MENTION ADDITION OF "ACTIVE" VECTOR OUTPUT? ###
"""
If used with a `BEDFile` object `x`, then the additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs()`.
- `means`, a vector of SNP means. Defaults to `mean(T, x, shared=true, pids=procs()`.
- `invstds`, a vector of SNP precisions. Defaults to `invstd(x, means, shared=true, pids=procs()`.
"""
function L0_log{T <: Float}(
    x        :: BEDFile{T},
    y        :: SharedVector{T},
    k        :: Int;
    n        :: Int              = length(y),
    p        :: Int              = size(x,2),
    pids     :: Vector{Int} = procs(),
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
    λ   :: T    = convert(T, sqrt(log(p)/n)),
    mu       :: T    = one(T),
    tol      :: T    = convert(T, 1e-6),
    tolG     :: T    = convert(T, 1e-3),
    tolrefit :: T    = convert(T, 1e-6),
    max_iter :: Int  = 100,
    max_step :: Int  = 100,
    mn       :: Int  = sum(mask_n),
    refit    :: Bool = true,
    quiet    :: Bool = true,
)


    # start timer
    tic()

    # check arguments
    n        == size(x,1) || throw(ArgumentError("Length n = $n of response vector y does not match number of rows = $(size(x,1)) in x"))
    k        >  p         && throw(ArgumentError("Value of argument k = $k exceeds number of predictors p = $p"))
    λ   <  zero(T)   && throw(ArgumentError("Value of argument λ = $λ must be nonnegative"))
    mu       >  zero(T)   || throw(ArgumentError("Value of argument mu must be positive"))
    tol      >  eps(T)    || throw(ArgumentError("Value of argument tol must exceed machine precision"))
    tolG     >  eps(T)    || throw(ArgumentError("Value of argument tolG must exceed machine precision"))
    tolrefit >  eps(T)    || throw(ArgumentError("Value of argument tolrefit must exceed machine precision"))
    max_iter >= 0         || throw(ArgumentError("Value of max_iter must be nonnegative\n"))
    max_step >= 0         || throw(ArgumentError("Value of max_step must be nonnegative\n"))

    # enforce all 0/1 in mask_n
    sum((mask_n .== 1) $ (mask_n .== 0)) == n || throw(ArgumentError("Argument mask_n can only contain 1s and 0s"))

    # initialize return values
    iter   = 0               # number of iterations of L0_reg
    exec_time   = zero(T)         # compute time *within* L0_reg
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
    for iter = 1:max_iter

        # notify and break if maximum iterations are reached.
        if iter >= max_iter

            # warn about hitting maximum iterations
            warn("L0_log has hit maximum iterations $(max_iter)!\nCurrent loss: $(loss)\n")

            # send elements below tol to zero
            threshold!(b, tol, n=p)

            # if requested, apply final refit without regularization
            if refit
                copy!(idxs0,idxs)
                update_indices!(idxs, b, p=p)
                !isequal(idxs,idxs0) && decompress_genotypes!(xk, x, idxs, mask_n) 
                bk2,bktrk = fit_logistic(xk, y, mask_n, zero(T), n=n, p=k, d2b=d2b, x2=xk2, b=bk, b0=bk0, ntb=ntb, db=db, Xb=Xb, lxb=lxb, l2xb=l2xb, tol=tolrefit, max_iter=max_step, quiet=true)
                b[idxs] = bk2
            end

            # stop timer
            exec_time = toq()

            # these are output variables for function
            # wrap them into a Dict and return
            output = Dict{String, Any}("time" => exec_time, "loss" => loss, "iter" => iter, "beta" => copy(b), "active" => copy(active))

            return output
        end

        # save previous loss, iterate
        loss0 = loss
        copy!(b0,b)

        # size of current active set?
        if iter > 1
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
            A_mul_B!(v.xb, x, v.b, v.idxs2, sum(v.idxs2), mask_n, pids=pids)
        end

        # recompute active loss = (-dot(y,Xb) + sum(log(1.0 + exp(Xb)))) / n + 0.5*λ*sumabs2(b[active])
        # special case: b = 0 --> Xb = 0 --> loss = n*log(1 + exp(0))/n + 0.5*λ*norm(0)
        if all(Xb .== zero(T))
            loss = log(one(T) + one(T))
        else
            loss = logistic_loglik(Xb,y,b,bidxs,mask_n,0,λ,k, n=n, mn=mn)
        end

        # guard against numerical instabilities in loss function
        isnan(loss) && throw(error("Loss function is NaN, something went wrong..."))
        isinf(loss) && throw(error("Loss function is Inf, something went wrong..."))

        # recompute active gradient df[active] = (x[:,active]'*(logistic(Xb) - y)) / n + λ*b[active]
        # arrange calculations differently if active set is entire support 1, 2, ..., p
        if lt == p
            logistic_grad!(df, lxb, x, y, b, Xb, x.means, x.precs, mask_n, λ, n=n, p=p, pids=pids, mn=mn)
        else
            logistic_grad!(df, lxb, x, y, b, Xb, x.means, x.precs, active, mask_n, lt, λ, n=n, mn=mn)
        end

        # identify 3*k dominant directions in gradient
        selectperm!(dfidxs, df, 1:num_df, by=abs, rev=true, initialized=true)

        # clean b and fill, b[active] = b0[active] - mu*df[active]
        # note that sparsity level is size(active) which is one of [k, k+1, ..., 3*k]
        update_x!(b, b0, df, active, mu, k=lt)

        # now apply hard threshold on model to enforce original desired sparsity k
        project_k!(b,k)

        # refit nonzeroes in b?
        if refit

            # update boolean vector of nonzeroes
            copy!(idxs0,idxs)
            update_indices!(idxs, b, p=p)

            # update active set of x, if necessary
            (iter == 1 || !isequal(idxs,idxs0)) && decompress_genotypes!(xk, x, idxs, mask_n) 

            # attempt refit but guard against possible instabilities or singularities
            # these are more frequent if λ is small
            # if refitting destabilizes, then leave b alone
            try
                bk2, nt_iter, bktrk = fit_logistic(xk, y, mask_n, λ, n=n, p=k, d2b=d2b, x2=xk2, b=bk, b0=bk0, ntb=ntb, db=db, Xb=Xb, lxb=lxb, l2xb=l2xb, tol=tolrefit, max_iter=max_step, quiet=true, mn=mn)
                b[idxs] = bk2
            catch e
#                warn("in refitting, caught error: ", e)
#                warn("skipping refit")
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
        quiet || @printf("%d\t%d\t%d\t%3.7f\t%3.7f\n",iter,nt_iter,bktrk,loss,normdf)

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
                !isequal(idxs,idxs0) && decompress_genotypes!(xk, x, idxs, mask_n) 
                bk2, nt_iter, bktrk = fit_logistic(xk, y, zero(T), n=n, p=k, d2b=d2b, x2=xk2, b=bk, b0=bk0, ntb=ntb, db=db, Xb=Xb, lxb=lxb, l2xb=l2xb, tol=tolrefit, max_iter=max_step, quiet=true)
                b[idxs] = bk2
            end

            # stop time
            exec_time = toq()

            if !quiet
                println("\nL0_log has converged successfully.")
                @printf("Results:\nIterations: %d\n", iter)
                @printf("Final Loss: %3.7f\n", loss)
                @printf("Norm of active gradient: %3.7f\n", normdf)
                @printf("Total Compute Time: %3.3f sec\n", exec_time)
            end

            # these are output variables for function
            # wrap them into a Dict and return
            output = Dict{String, Any}("time" => exec_time, "loss" => loss, "iter" => iter, "beta" => copy(b), "active" => copy(active))

            return output
        end # end convergence check
    end # end main GraSP iterations
end # end L0_log


###
### 13 Oct 2016: everything below this line is the dark side...
###

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
    λs  :: DenseVector{T}   = ones(length(path)) * convert(T, sqrt(log(p) / n)),
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
        λ = λs[i]

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
        output = L0_log(x,y,q, n=n, p=p, b=b, tol=tol, tolG=tolG, tolrefit=tolrefit, refit=refit, max_iter=max_iter, max_step=max_step, quiet=quiet, b0=b0, df=df, Xb=Xb, lxb=lxb, l2xb=l2xb, bidxs=bidxs, dfidxs=dfidxs, active=active, idxs=idxs, idxs0=idxs0, bk=bk, xk=xk, xk2=xk2, d2b=d2b, bk0=bk0, ntb=ntb, db=db, dfk=dfk, means=means, invstds=invstds, idxs2=idxs2, mask_n=mask_n, λ=λ)

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


function one_fold_log{T <: Float}(
    x         :: BEDFile,
    y         :: DenseVector{T},
    path      :: DenseVector{Int},
    folds     :: DenseVector{Int},
    fold      :: Int;
    n         :: Int              = length(y),
    p         :: Int              = size(x,2),
    λs   :: DenseVector{T}   = ones(length(path)) * convert(T, sqrt(log(p) / n)),
    pids      :: DenseVector{Int} = procs(),
    means     :: DenseVector{T}   = mean(T,x, shared=true, pids=pids),
    invstds   :: DenseVector{T}   = invstd(x,means, shared=true, pids=pids),
    criterion :: String      = "deviance",
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
    betas = iht_path_log(x,y,path, tol=tol, tolG=tolG, tolrefit=tolrefit, max_iter=max_iter, max_step=max_step, refit=refit, quiet=quiet, mask_n=train_idx, pids=pids, means=means, invstds=invstds, λs=λs)


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

            # use k = 0, λ = 0.0, sortidx = falses(p) to ensure that regularizer is not included in deviance
            errors[i] = logistic_loglik(Xb,y,b,notrues,test_idx,0,zero(T),0, n=n, mn=mn) / 2
#            errors[i] = 2.0*logistic_loglik(Xb,y,b,indices,test_idx,0,λ,k, n=n, mn=mn)
#            errors[i] = -2.0 / n * ( dot(y[test_idx .== 1],Xb[test_idx .== 1]) - sum(log(1.0 + exp(Xb[test_idx .== 1]))) ) # + 0.5*λ*sumabs2(b[indices])
        end

    end

    return errors
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
    xfile      :: String,
    xtfile     :: String,
    x2file     :: String,
    yfile      :: String,
    meanfile   :: String,
    invstdfile :: String,
    path       :: DenseVector{Int},
    folds      :: DenseVector{Int},
    numfolds   :: Int;
    n          :: Int                = length(y),
    p          :: Int                = size(x,2),
    pids       :: DenseVector{Int}   = procs(),
    λs    :: DenseVector{Float} = SharedArray(T, (length(path),), pids=pids, init = S -> S[localindexes(S)] = sqrt(log(p) / n)), # type stable for Float32?
    criterion  :: String        = "deviance",
    tol        :: Float              = convert(T, 1e-6),
    tolG       :: Float              = convert(T, 1e-3),
    tolrefit   :: Float              = convert(T, 1e-6),
    max_iter   :: Int                = 100,
    max_step   :: Int                = 100,
    quiet      :: Bool               = true,
    refit      :: Bool               = true,
    header     :: Bool               = false
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

                                one_fold_log(x, y, path, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, means=means, invstds=invstds, pids=pids, tol=tol, tolG=tolG, tolrefit=tolrefit, refit=refit, criterion=criterion, λs=λs)

                                # have each worker explicitly kill arrays
                                x       = false
                                y       = false
                                means   = false
                                invstds = false
                                gc()
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
pfold_log(xfile::String, xtfile::String, x2file::String, yfile::String, meanfile::String, invstdfile::String, path::DenseVector{Int}, folds::DenseVector{Int}, numfolds::Int; n::Int=length(y), p::Int=size(x,2), pids::DenseVector{Int}=procs(), λs::DenseVector{Float64}=SharedArray(Float64, (length(path),), pids=pids, init = S -> S[localindexes(S)] = sqrt(log(p) / n)), criterion::String="deviance", tol::Float64=1e-6, tolG::Float64=1e-3, tolrefit::Float64=1e-6, max_iter::Int=100, max_step::Int=100, quiet::Bool=true, refit::Bool=true, header::Bool=false) = pfold_log(Float64, xfile, xtfile, x2file, yfile, meanfile, invstdfile, path, folds, numfolds, n=n, p=p, pids=pids, λs=λs, criterion=criterion, tol=tol, tolG=tolG, tolrefit=tolrefit, max_iter=max_iter, max_step=max_step, quiet=quiet, refit=refit, header=header)

"""
    cv_log(xfile,xtfile,x2file,yfile,meanfile,invstdfile,path,kernfile,folds,numfolds [, pids=procs()])

This variant of `cv_log()` performs `numfolds` crossvalidation with a `BEDFile` object loaded by `xfile`, `xtfile`, and `x2file`,
with column means stored in `meanfile` and column precisions stored in `invstdfile`.
The continuous response is stored in `yfile` with data particioned by the `Int` vector `folds`.
The folds are distributed across the processes given by `pids`.
"""
function cv_log(
    T             :: Type,
    xfile         :: String,
    xtfile        :: String,
    x2file        :: String,
    yfile         :: String,
    meanfile      :: String,
    invstdfile    :: String,
    path          :: DenseVector{Int},
    folds         :: DenseVector{Int},
    numfolds      :: Int;
    pids          :: DenseVector{Int}   = procs(),
    λs       :: DenseVector{Float} = SharedArray(T, (length(path),), pids=pids, init = S -> S[localindexes(S)] = one(T)),
    criterion     :: String        = "deviance",
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
    errors = pfold_log(xfile, xtfile, x2file, yfile, meanfile, invstdfile, path, folds, numfolds, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids, header=header, tolG=tolG, tolrefit=tolrefit, refit=refit, criterion=criterion, λs=λs)

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

        # get λ value for best model
        λ = λs[path .== k][1]

        # use L0_reg to extract model
        output = L0_log(x,y,k,n=n, p=p, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, means=means, invstds=invstds, tolG=tolG, tolrefit=tolrefit, refit=refit, λ=λ, pids=pids)
        b = copy(output["beta"])

        # which components of beta are nonzero?
        bidx = find(b)

        return errors, b[bidx], bidx
    end
    return errors
end

cv_log(xfile::String, xtfile::String, x2file::String, yfile::String, meanfile::String, invstdfile::String, path::DenseVector{Int}, folds::DenseVector{Int}, numfolds::Int; pids::DenseVector{Int}=procs(), λs::DenseVector{Float64} = SharedArray(Float, (length(path),), pids=pids, init = S -> S[localindexes(S)] = one(Float64)),  criterion::String="deviance", tol::Float64=1e-6, tolG::Float64=1e-3, tolrefit::Float64=1e-6, max_iter::Int=100, max_step::Int=100, quiet::Bool=true, refit::Bool=true, header::Bool=false)=cv_log(Float64, xfile, xtfile, x2file, yfile, meanfile, invstdfile, path, folds, numfolds, pids=pids, λs=λs, criterion=criterion, tol=tol, tolG=tolG, tolrefit=tolrefit, max_iter=max_iter, max_step=max_step, quiet=quiet, refit=refit, header=header)
