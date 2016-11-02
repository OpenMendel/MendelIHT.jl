### 17 OCT 2016: MUST RECODE THIS FUNCTION
#"""
#    fit_logistic(x, y, mask_n, λ)
#
#If called with an `Int` vector `mask_n`, then `fit_logistic()` will refit logistic effect sizes while masking components `y[i]` where `mask_n[i] = 0`.
#"""
#function fit_logistic{T <: Float}(
#    x        :: DenseMatrix{T},
#    y        :: DenseVector{T},
#    mask_n   :: DenseVector{Int},
#    λ   :: T;
#    n        :: Int    = length(y),
#    p        :: Int    = size(x,2),
#    mn       :: Int    = sum(mask_n),
#    d2b      :: DenseMatrix{T} = zeros(T, p,p),
#    x2       :: DenseMatrix{T} = zeros(T, n,p),
#    b        :: DenseVector{T} = zeros(T, p),
#    b0       :: DenseVector{T} = zeros(T, p),
#    ntb      :: DenseVector{T} = zeros(T, p),
#    db       :: DenseVector{T} = zeros(T, p),
#    xβ       :: DenseVector{T} = zeros(T, n),
#    lxβ      :: DenseVector{T} = zeros(T, n),
#    l2xb     :: DenseVector{T} = zeros(T, n),
#    tol      :: Float = convert(T, 1e-8),
#    max_iter :: Int   = 50,
#    quiet    :: Bool  = true,
#)
#
#    # if b is not warm-started, then ensure that it is not entirely zero
#    if all(b .== 0) 
#        b[1] = logit(mean(y[mask_n .== 1]))
#    end
#
#    # initialize intermediate arrays for calculations
#    BLAS.gemv!('N', one(T), x, b, zero(T), xβ)
#    log2xb!(lxβ, l2xb, xβ, n=n)
#    mask!(lxβ, mask_n, 0, zero(T), n=n)
##    lxβ[mask_n .== 0] = zero(T)
#    copy!(b0,b)
#    fill!(db, zero(T))
#
#    # track objective
#    old_obj = Inf
#    new_obj = logistic_loglik(xβ,y,b,mask_n,0,λ,p, n=n)
#
#    # output progress to console
#    quiet || println("Iter\tHalves\tObjective")
#
#    i = 0
#    bktrk = 0
#    # enter loop for Newton's method
#    for i = 1:max_iter
#
#        # db = (x'*(lxβ - y)) / n + λ*b
#        BLAS.axpy!(n, -one(T), sdata(y), 1, sdata(lxβ), 1)
#        mask!(lxβ, mask_n, 0, zero(T), n=n)
##        lxβ[mask_n .== 0] = zero(T)
#        BLAS.gemv!('T', one(T), sdata(x), sdata(lxβ), zero(T), sdata(db))
##        BLAS.scal!(p, 1/n, sdata(db), 1)
#        BLAS.scal!(p, 1/mn, sdata(db), 1)
#        BLAS.axpy!(p, λ, sdata(b), 1, sdata(db), 1)
#
#        # d2b = (x'*diagm(l2xb)*x)/n + λ*I
#        # note that log2xb!() already performs division by n on l2xb
#        copy!(x2,x)
#        mask!(l2xb, mask_n, 0, zero(T), n=n)
##        l2xb[mask_n .== 0] = zero(T)
#        BLAS.scal!(p, n/mn, sdata(l2xb), 1) # rescale to number of unmasked samples
#        scale!(sdata(l2xb), sdata(x2))
#        BLAS.gemm!('T', 'N', one(T), sdata(x), sdata(x2), zero(T), sdata(d2b))
#        d2b += λ*I
#
#        # b = b0 - ntb = b0 - inv(d2b)*db
#        #   = b0 - inv[ x' diagm(pi) diagm(1 - pi) x + λ*I] [x' (pi - y) + λ*b]
#        ntb = d2b\db
#        copy!(b,b0)
#        BLAS.axpy!(p,-one(T),ntb,1,b,1)
#
#        # compute objective
#        new_obj = logistic_loglik(xβ,y,b,mask_n,0,λ,p, n=n)
#
#        # backtrack
#        j = 0
#        while (new_obj > old_obj + tol) && (j < 50)
#
#            # increment iterator
#            j += 1
#
#            # b = b0 - 0.5*ntb
#            copy!(b,b0)
##            BLAS.axpy!(p,-(0.5^j),ntb,1,b,1)
#            BLAS.axpy!(p,-one(T) / (2^j),ntb,1,b,1)
#
#            # recalculate objective
#            new_obj = logistic_loglik(xβ,y,b,mask_n,0,λ,p, n=n)
#
#        end
#
#        # accumulate total backtracking steps
#        bktrk += j
#
#        # track distance between iterates
#        dist = euclidean(sdata(b),sdata(b0)) / (norm(sdata(b0),2) + one(T))
#
#        # track progress
#        quiet || println(i, "\t", j, "\t", dist)
#
#        # check for convergence
#        # if converged, then return b
#        dist < tol && return b, i, div(bktrk,i)
#
#        # unconverged at this point, so update intermediate arrays
#        BLAS.gemv!('N', one(T), sdata(x), sdata(b), zero(T), sdata(xβ))
#        log2xb!(lxβ, l2xb, xβ, n=n)
#
#        # save previous beta
#        copy!(b0, b)
#        old_obj = new_obj
#    end
#
#    warn("fit_logistic failed to converge in $(max_iter) iterations, exiting...")
#    return b, i, div(bktrk,max_iter)
#end


### MENTION ADDITION OF "ACTIVE" VECTOR OUTPUT? ###
"""
If used with a `BEDFile` object `x`, then the additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs()`.
- `means`, a vector of SNP means. Defaults to `mean(T, x, shared=true, pids=procs()`.
- `invstds`, a vector of SNP precisions. Defaults to `invstd(x, means, shared=true, pids=procs()`.
"""
function L0_log{T <: Float, V <: SharedVector}(
    x        :: BEDFile{T},
    y        :: V,
    k        :: Int;
    pids     :: Vector{Int} = procs(x),
    mask_n   :: DenseVector{Int} = ones(Int, n),
    lambda   :: T    = convert(T, sqrt(log(p)/n)),
    mu       :: T    = one(T),
    tol      :: T    = convert(T, 1e-6),
    tolG     :: T    = convert(T, 1e-3),
    tolrefit :: T    = convert(T, 1e-6),
    max_iter :: Int  = 100,
    max_step :: Int  = 100,
    refit    :: Bool = true,
    quiet    :: Bool = true,
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
    mn = sum(mask_n),

    # initialize return values
    iter      = 0               # number of iterations of L0_reg
    exec_time = zero(T)         # compute time *within* L0_reg
    loss      = convert(T, Inf) # loss function value

    # initialize algorithm parameters
    num_df    = min(p,3*k)      # largest 3*k active components of gradient
    short_df  = min(p,2*k)      # largest 2*k active components of gradient
    converged = false           # is algorithm converged?
    normdf    = convert(T, Inf) # norm of active portion of gradient
    loss0     = convert(T, Inf) # previous loss function value
    bktrk     = 0               # track loss and number of backtracking steps
    nt_iter   = 0               # number of Newton iterations
    lt        = length(active)  # size of current active set?

    # formatted output to monitor algorithm progress
    quiet || print_header_log()

    # main GraSP iterations
    for iter = 1:max_iter

        # notify and break if maximum iterations are reached.
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
        loss0 = loss
        copy!(v.b0, v.b)

        # size of current active set?
        if iter > 1
            active = union(v.dfidxs[1:short_df], v.bidxs[1:k])
        end
        lt = length(active)

        # update x*b
        # no need to compute anything if b = 0
        if all(v.b .== zero(T))
            fill!(v.xb,zero(T))
        else
            fill!(x.idxs2,false)
            v.idxs2[active] = true
            A_mul_B!(v.xb, x, v.b, v.idxs2, sum(v.idxs2), mask_n, pids=pids)
        end

        # recompute active loss = (-dot(y,Xb) + sum(log(1.0 + exp(Xb)))) / n + 0.5*lambda*sumabs2(b[active])
        # special case: b = 0 --> Xb = 0 --> loss = n*log(1 + exp(0))/n + 0.5*lambda*norm(0)
        if all(v.xb .== 0)
            loss = log(2)
        else
            #loss = logistic_loglik(v.xb, y, v.b, v.bidxs,mask_n,0,lambda,k, n=n)
            loss = logistic_loglik(v.xb, y, v.b, v.active, mask_n, 0, lambda, k)
        end

        # guard against numerical instabilities in loss function
        iter < 2 && check_finiteness(loss)

        # recompute active gradient df[active] = (x[:,active]'*(logistic(Xb) - y)) / n + lambda*b[active]
        # arrange calculations differently if active set is entire support 1, 2, ..., p
        if lt == p
            logistic_grad!(v.df, v.lxb, x, y, v.b, v.xb, mask_n, lambda, pids=pids)
        else
            logistic_grad!(v.df, v.lxb, x, y, v.b, v.xb, v.active, mask_n, lt, lambda)
        end

        # identify 3*k dominant directions in gradient
        selectperm!(dfidxs, df, 1:num_df, by=abs, rev=true, initialized=true)

        # clean b and fill, b[active] = b0[active] - mu*df[active]
        # note that sparsity level is size(active) which is one of [k, k+1, ..., 3*k]
        #update_x!(v.b, v.b0, v.df, v.active, mu, k=lt)
        fill!(v.b, zero(T))
        v.b[v.active] = v.b0[v.active] .- mu.*v.df[v.active]

        # now apply hard threshold on model to enforce original desired sparsity k
        project_k!(b,k)

        # refit nonzeroes in b?
        if refit

            # update boolean vector of nonzeroes
            copy!(v.idxs0, v.idxs)
            update_indices!(v.idxs, v.b)

            # update active set of x, if necessary
            (iter == 1 || !isequal(v.idxs, v.idxs0)) && decompress_genotypes!(v.xk, x, v.idxs, mask_n)

            # attempt refit but guard against possible instabilities or singularities
            # these are more frequent if lambda is small
            # if refitting destabilizes, then leave b alone
            try
                nt_iter, bktrk = fit_logistic!(v, y, lambda, tol=tolrefit, max_iter=max_step, quiet=true)
                v.b[v.idxs] = v.bk
            catch e
#                warn("in refitting, caught error: ", e)
#                warn("skipping refit")
                bktrk   = 0
                nt_iter = 0
            end # end try-catch for refit
        end # end if-else for refit

        # need norm of largest 3*k components of gradient
        normdf = df_norm(v.df, v.dfidxs, 1, num_df)

        # guard against numerical instabilities in gradient
        check_finiteness(normdf)

        # check for convergence
        converged_obj  = abs(loss - loss0) < tol
        converged_grad = normdf < tolG
        converged      = converged_obj || converged_grad
        stuck          = abs(loss - loss00) < tol

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
                decompress_genotypes!(v.xk, x, v.idxs, mask_n) 
                try
                    nt_iter, bktrk = fit_logistic!(v, y, zero(T), tol=tolrefit, max_iter=max_step, quiet=true)
                    v.b[v.idxs] = v.bk
                catch e
                    nt_iter = 0
                    bktrk   = 0
                end
            end

            # stop time
            exec_time = toq()

            # announce convergence
            !quiet && print_log_convergence(iter, loss, exec_time, normdf)

            # these are output variables for function
            # wrap them into a Dict and return
            return IHTLogResults(exec_time, iter, loss, copy(v.b), copy(v.active))

        end # end convergence check
    end # end main GraSP iterations

    # return null result
    return IHTLogResults(zero(T), 0, zero(T), zeros(T, 1), zeros(Int, 1)) 
end # end L0_log



"""
    iht_path_log(x::BEDFile, y, path)

If used with a `BEDFile` object `x`, then the additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs()`.
"""
function iht_path_log{T <: Float}(
    x        :: BEDFile,
    y        :: SharedVector{T},
    path     :: DenseVector{Int};
    pids     :: Vector{Int}    = procs(x),
    mask_n   :: BitArray{1}    = trues(y), 
    lambdas  :: DenseVector{T} = default_lambda(x, y, length(path)),
    mu       :: T    = one(T),
    tol      :: T    = convert(T, 1e-6),
    tolG     :: T    = convert(T, 1e-3),
    tolrefit :: T    = convert(T, 1e-6),
    max_iter :: Int  = 100,
    max_step :: Int  = 100,
    refit    :: Bool = true,
    quiet    :: Bool = true,
)

    # dimensions of problems?
    n,p = size(x)

    # how many models will we compute?
    nmodels = length(path)

    # preallocate temporary arrays
    v = IHTLogVariables(x, y, 1)

    # betas will be a sparse matrix with models
    betas = spzeros(T, p, nmodels)

    # compute the path
    for i = 1:nmodels

        # model size?
        q = path[i]
        quiet || println("Current model size: $q")

        # update temporary variables with new threshold size q
        update_variables!(v, x, q)

        # store projection of beta onto largest k nonzeroes in magnitude
        project_k!(v.b, q)

        # current regularization parameter?
        λ = lambdas[i]

        # now compute current model
        output = L0_log(x, y, q, v=v, lambda=λ, mu=mu, tol=tol, tolG=tolG, tolrefit=tolrefit, max_iter=max_iter, max_step=max_step, refit=refit, quiet=quiet)

        # extract and save model
        betas[:,i] = sparsevec(sdata(output.beta))

        # run garbage compiler to clear temporary arrays for next step of path
        #gc()
    end

    # return a sparsified copy of the models
    return betas
end



function one_fold_log{T <: Float}(
    x         :: BEDFile{T},
    y         :: DenseVector{T},
    path      :: DenseVector{Int},
    folds     :: DenseVector{Int},
    fold      :: Int;
    lambdas   :: DenseVector{T} = default_lambda(x, y, length(path)),
    pids      :: Vector{Int}    = procs(x),
    criterion :: String = "deviance",
    tol       :: T      = convert(T, 1e-6),
    tolG      :: T      = convert(T, 1e-3),
    tolrefit  :: T      = convert(T, 1e-6),
    max_iter  :: Int    = 100,
    max_step  :: Int    = 100,
    refit     :: Bool   = true,
    quiet     :: Bool   = true,
)
    # problem dimensions?
    n,p = size(x)
    nmodels = length(path)

    # ensure that crossvalidation criterion is valid
    criterion in ["deviance", "class"] || throw(ArgumentError("Argument criterion must be 'deviance' or 'class'"))

    # make vector of indices for folds
    test_idx  = folds .== fold
    test_size = sum(test_idx)

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # allocate the arrays for the training set
#    train_idx = convert(Vector{Int}, train_idx)
#    test_idx  = convert(Vector{Int}, train_idx)

    # compute the regularization path on the training set
    betas = iht_path_log(x, y, path, tol=tol, tolG=tolG, tolrefit=tolrefit, max_iter=max_iter, max_step=max_step, refit=refit, quiet=quiet, lambdas=lambdas, mask_n=train_idx)


    # allocate an index vector for b
    idx = falses(p)

    # allocate temporary arrays for the test set
    xb  = SharedArray(T, n, pids=pids) :: SharedVector{T}
    b   = SharedArray(T, p, pids=pids) :: SharedVector{T}
    lxb = zeros(T, (n, nmodels)) 
    xβ  = zeros(T, (n, nmodels))

    # compute the mean out-of-sample error for the TEST set
    # do this for every computed model in regularization path
    for i = 1:size(betas,2)

        # which model are we analyzing?
        m = path[i]

        # pull ith model in dense vector format
        b2 = full(vec(betas[:,i]))

        # copy it into SharedArray b
        copy!(b,b2)

        # indices stores Boolean indexes of nonzeroes in b
        update_indices!(idx, b)

        # compute estimated responses
        A_mul_B!(xb, x, b, idx, m, test_idx, pids=pids)
        setindex!(xβ, xb, :, m) 
    end

    # mask data from training set
    fill!(view(xb, train_idx, :), zero(T))
    logistic!(lxb, xb)
    fill!(view(lxb, train_idx, :), zero(T))
    ytest = copy(y)
    fill!(view(ytest, train_idx, :), zero(T))

    # compute out-of-sample error as misclassification (L1 norm) averaged over size of test set
    errors = zeros(T, nmodels) 
    if criterion == "class" # compute misclassification error
        errors .= vec(sumabs(lxb .- ytest, 1)) ./ sum(!train_idx)
    else # else -> criterion == "deviance" = 2*(neg logistic loss with no reg)
        errors .= - 2 ./ test_size .* ( vec( sum(xβ .* ytest, 1) .- sum(log(one(T) .+ exp(xβ)), 1) ) )
    end

    return errors :: Vector{T}
end


"""
    pfold_log(xfile, xtfile, x2file, yfile, meanfile, precfile, path, folds, q[, pids=procs()])

This function is the parallel execution kernel in `cv_log()`. It is not meant to be called outside of `cv_iht()`.
It will distribute `q` crossvalidation folds across the processes supplied by the optional argument `pids` and call `one_fold_log()` for each fold.
Each fold will compute a regularization path given by `path`.
`pfold()` collects the vectors of MSEs returned by calling `one_fold_log()` for each process, reduces them, and returns their average across all folds.
"""
function pfold_log(
    T        :: Type,
    xfile    :: String,
    xtfile   :: String,
    x2file   :: String,
    yfile    :: String,
    meanfile :: String,
    precfile :: String;
    n        :: Int = begin
                  # find n from the corresponding FAM file, then make folds
                  famfile = xfile[1:(endof(xfile)-3)] * "fam"
                  countlines(famfile)
                  end,
    p        :: Int = begin 
                  # find p from the corresponding BIM file, then make path
                  bimfile = xfile[1:(endof(xfile)-3)] * "bim"
                  countlines(bimfile)
                end,
    path      :: DenseVector{Int} = collect(1:min(20,p)), 
    folds     :: DenseVector{Int} = cv_get_folds(n, q)
    q         :: Int = cv_get_num_folds(3,5), 
    pids      :: Vector{Int} = procs(),
    lambdas   :: DenseVector{Float} = default_lambda(n, p, q), 
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
    results = SharedArray(T, (length(path),q), pids=pids) :: SharedMatrix{T}

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
                            processes = [worker]
                            x = BEDFile(T, xfile, xtfile, x2file, meanfile, precfile, pids=processes, header=header)
                            n = x.geno.n
                            y = SharedArray(abspath(yfile), T, (n,), pids=processes) :: SharedVector{T}
                            one_fold_log(x, y, path, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=processes, tol=tol, tolG=tolG, tolrefit=tolrefit, refit=refit, criterion=criterion, lambdas=lambdas)

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

# default type for pfold_log is Float64
pfold_log(xfile::String, xtfile::String, x2file::String, yfile::String, meanfile::String, precfile::String; n::Int=begin famfile=xfile[1:(endof(xfile)-3)]*"fam"; countlines(famfile) end, p::Int=begin bimfile=xfile[1:(endof(xfile)-3)]*"bim"; countlines(bimfile) end, q::Int = cv_get_num_folds(3,5), path::DenseVector{Int} = collect(1:min(20,p)), folds::DenseVector{Int} = cv_get_folds(n,q), lambdas::DenseVector{Float64} = default_lambda(n,p,q), pids::Vector{Int}=procs(), criterion::String="deviance", tol::Float64=1e-6, tolG::Float64=1e-3, tolrefit::Float64=1e-6, max_iter::Int=100, max_step::Int=100, quiet::Bool=true, refit::Bool=true, header::Bool=false) = pfold_log(Float64, xfile, xtfile, x2file, yfile, meanfile, precfile, path, folds, q, n=n, p=p, pids=pids, lambdas=lambdas, criterion=criterion, tol=tol, tolG=tolG, tolrefit=tolrefit, max_iter=max_iter, max_step=max_step, quiet=quiet, refit=refit, header=header)

"""
    cv_log(xfile,xtfile,x2file,yfile,meanfile,precfile,path,kernfile,folds,q [, pids=procs()])

This variant of `cv_log()` performs `q` crossvalidation with a `BEDFile` object loaded by `xfile`, `xtfile`, and `x2file`,
with column means stored in `meanfile` and column precisions stored in `precfile`.
The continuous response is stored in `yfile` with data particioned by the `Int` vector `folds`.
The folds are distributed across the processes given by `pids`.
"""
function cv_log(
    T         :: Type,
    xfile     :: String,
    xtfile    :: String,
    x2file    :: String,
    yfile     :: String,
    meanfile  :: String,
    precfile  :: String;
    n         :: Int = begin
                  # find n from the corresponding FAM file, then make folds
                  famfile = xfile[1:(endof(xfile)-3)] * "fam"
                  countlines(famfile)
                 end,
    p         :: Int = begin 
                  # find p from the corresponding BIM file, then make path
                  bimfile = xfile[1:(endof(xfile)-3)] * "bim"
                  countlines(bimfile)
                 end,
    q         :: Int = cv_get_num_folds(3,5), 
    path      :: DenseVector{Int} = collect(1:min(20,p)),
    folds     :: DenseVector{Int} = cv_get_folds(n,q),
    pids      :: DenseVector{Int} = procs(),
    lambdas   :: DenseVector      = default_lambda(n,p,q), # type unstable? 
    criterion :: String = "deviance",
    tol       :: Float  = convert(T, 1e-6),
    tolG      :: Float  = convert(T, 1e-3),
    tolrefit  :: Float  = convert(T, 1e-6),
    max_iter  :: Int    = 100,
    max_step  :: Int    = 100,
    quiet     :: Bool   = true,
    refit     :: Bool   = true,
    header    :: Bool   = false
)
    # ensure that crossvalidation criterion is valid
    criterion in ["deviance", "class"] || throw(ArgumentError("Argument criterion must be 'deviance' or 'class'"))
    T <: Float || throw(ArgumentError("Argument T must be either Float32 or T"))

    # how many elements are in the path?
    nmodels = length(path)

    # want to compute a path for each fold
    # the folds are computed asynchronously
    # only use the worker processes
    errors = pfold_log(T, xfile, xtfile, x2file, yfile, meanfile, precfile, n=n, p=p, path=path, folds=folds, q=q, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids, header=header, tolG=tolG, tolrefit=tolrefit, refit=refit, criterion=criterion, lambdas=lambdas)

    # what is the best model size?
    k = path[indmin(errors)] :: Int

    # print results
    !quiet && print_cv_results(errors, path, k)

    # recompute ideal model
    # load data on *all* processes
    x = BEDFile(T, xfile, xtfile, x2file, header=header, pids=pids)
    n = x.geno.n
    y = SharedArray(abspath(yfile), T, (n,), pids=pids) :: SharedVector{T}

    # get lambda value for best model
    λ = lambdas[path .== k][1]

    # use L0_reg to extract model
    output = L0_log(x, y, k, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, tolG=tolG, tolrefit=tolrefit, refit=refit, lambda=λ, pids=pids)

    # which components of beta are nonzero?
    bidx = find(output.beta)

    return IHTCrossvalidationResults(errors, sdata(path), output.beta[bidx], bidx, k)

end

# default XV function type is Float64
cv_log(xfile::String, xtfile::String, x2file::String, yfile::String, meanfile::String, precfile::String; n::Int = begin famfile=xfile[1:(endof(xfile)-3)]*"fam"; countlines(famfile) end, p::Int = begin bimfile=xfile[1:(endof(xfile)-3)]*"bim"; countlines(bimfile) end, q::Int=cv_get_num_folds(3,5), path::DenseVector{Int} = collect(1:min(20,p)), folds::DenseVector{Int} = cv_get_folds(n,q), pids::Vector{Int} = procs(), lambdas::DenseVector{Float64} = default_lambda(n,p,q),  criterion::String="deviance", tol::Float64=1e-6, tolG::Float64=1e-3, tolrefit::Float64=1e-6, max_iter::Int=100, max_step::Int=100, quiet::Bool=true, refit::Bool=true, header::Bool=false)=cv_log(Float64, xfile, xtfile, x2file, yfile, meanfile, precfile, path=path, folds=folds, q=q, pids=pids, lambdas=lambdas, criterion=criterion, tol=tol, tolG=tolG, tolrefit=tolrefit, max_iter=max_iter, max_step=max_step, quiet=quiet, refit=refit, header=header)
