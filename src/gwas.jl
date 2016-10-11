"""
    iht!(v::IHTVariables, x::BEDFile, y, k)

If used with a `BEDFile` object `x`, then the temporary arrays `b0`, `Xb`, `Xb0`, and `r` housed in the `IHTVariables` object `v` are all initialized as `SharedArray`s of the proper dimensions.
The additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs(x)`.
"""
function iht!{T <: Float}(
    v     :: IHTVariables{T},
    x     :: BEDFile{T},
    y     :: DenseVector{T},
    k     :: Int;
    pids  :: DenseVector{Int} = procs(x),
    iter  :: Int = 1,
    nstep :: Int = 50,
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
    mu = (sumabs2(v.gk) / sumabs2(v.xgk)) :: T
#    mu = _iht_stepsize(v, k) :: T

    # notify problems with step size
    isfinite(mu) || throw(error("Step size is not finite, is active set all zero?"))
    mu <= eps(typeof(mu))  && warn("Step size $(mu) is below machine precision, algorithm may not converge correctly")

    # compute gradient step
    _iht_gradstep(v, mu, k)

    # update xb
    PLINK.A_mul_B!(v.xb, x, v.b, v.idx, k, pids=pids)

    # calculate omega
    omega_top, omega_bot = _iht_omega(v)

    # backtrack until mu sits below omega and support stabilizes
    mu_step = 0
    while _iht_backtrack(v, omega_top, omega_bot, mu, mu_step, nstep)

        # stephalving
        mu /= 2

        # warn if mu falls below machine epsilon
        mu <= eps(typeof(mu)) && warn("Step size equals zero, algorithm may not converge correctly")

        # recompute gradient step
        copy!(v.b, v.b0)
        _iht_gradstep(v, mu, k)

        # recompute xb
        PLINK.A_mul_B!(v.xb, x, v.b, v.idx, k, pids=pids)

        # calculate omega
        omega_top, omega_bot = _iht_omega(v)

        # increment the counter
        mu_step += 1
    end

    return mu::T, mu_step::Int
end

"""
    L0_reg(x::BEDFile, y, k)

If used with a `BEDFile` object `x`, then the temporary floating point arrays are all initialized as `SharedArray`s of the proper dimensions.
The additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs()`.
- `mask_n`, an `Int` vector used as a bitmask for crossvalidation purposes. Defaults to a vector of ones.
"""
function L0_reg{T <: Float}(
    x        :: BEDFile{T},
    y        :: DenseVector{T},
    k        :: Int;
    pids     :: DenseVector{Int} = procs(),
    temp     :: IHTVariables{T}  = IHTVariables(x, y, k),
    mask_n   :: DenseVector{Int} = ones(Int, size(y)),
    tol      :: Float = convert(T, 1e-4),
    max_iter :: Int   = 100,
    max_step :: Int   = 50,
    quiet    :: Bool  = true
)

    # start timer
    tic()

    # first handle errors
    k        >= 0      || throw(ArgumentError("Value of k must be nonnegative!\n"))
    max_iter >= 0      || throw(ArgumentError("Value of max_iter must be nonnegative!\n"))
    max_step >= 0      || throw(ArgumentError("Value of max_step must be nonnegative!\n"))
    tol      >  eps(T) || throw(ArgumentError("Value of global tol must exceed machine precision!\n"))
    n = length(y)
    sum((mask_n .== 1) $ (mask_n .== 0)) == n || throw(ArgumentError("Argument mask_n can only contain 1s and 0s"))

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

    # update xb, r, and gradient
#    initialize_xb_r_grad!(temp, x, y, k, pids=pids)
    if sum(temp.idx) == 0
        fill!(temp.xb, zero(T))
        copy!(temp.r, y)
        mask!(temp.r, mask_n, 0, zero(T))
    else
        A_mul_B!(temp.xb, x, temp.b, temp.idx, k, mask_n, pids=pids)
        difference!(temp.r, y, temp.xb)
        mask!(temp.r, mask_n, 0, zero(T))
    end

    # calculate the gradient
    At_mul_B!(temp.df, x, temp.r, mask_n, pids=pids)

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
        copy!(temp.b0, temp.b)   # b0 = b
        copy!(temp.xb0, temp.xb) # Xb0 = Xb
        loss = next_loss

        # now perform IHT step
        (mu, mu_step) = iht!(temp, x, y, k, nstep=max_step, iter=mm_iter, pids=pids)

        # the IHT kernel gives us an updated x*b
        # use it to recompute residuals and gradient
#        update_r_grad!(temp, x, y, pids=pids)
        difference!(temp.r, y, temp.xb)
        mask!(temp.r, mask_n, 0, zero(T), n=n)

        # use updated residuals to recompute the gradient on the GPU
        At_mul_B!(temp.df, x, temp.r, mask_n, pids=pids)

        # update loss, objective, and gradient
        next_loss = sumabs2(sdata(temp.r)) / 2

        # guard against numerical instabilities
        # ensure that objective is finite
        # if not, throw error
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

- `pids`, a vector of process IDs. Defaults to `procs(x)`.
- `mask_n`, an `Int` vector used as a bitmask for crossvalidation purposes. Defaults to a vector of ones.
"""
function iht_path{T <: Float}(
    x        :: BEDFile{T},
    y        :: DenseVector{T},
    path     :: DenseVector{Int};
    pids     :: DenseVector{Int} = procs(x),
    mask_n   :: DenseVector{Int} = ones(Int, size(y)),
    tol      :: Float = convert(T, 1e-4),
    max_iter :: Int   = 100,
    max_step :: Int   = 50,
    quiet    :: Bool  = true
)

    # size of problem?
    n,p = size(x)

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
        output = L0_reg(x, y, q, temp=temp, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids, mask_n=mask_n)

        # ensure that we correctly index the nonzeroes in b
        update_indices!(temp.idx, output.beta)
        fill!(temp.idx0, false)

        # put model into sparse matrix of betas
        betas[:,i] = sparsevec(output.beta)
    end

    # return a sparsified copy of the models
    return betas
end


"""
    one_fold(x::BEDFile, y, path, folds, fold)

If used with a `BEDFile` object `x`, then the additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs(x)`.
"""
function one_fold{T <: Float}(
    x        :: BEDFile{T},
    y        :: DenseVector{T},
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    fold     :: Int;
    pids     :: DenseVector{Int} = procs(x),
    tol      :: Float = convert(T, 1e-4),
    max_iter :: Int   = 100,
    max_step :: Int   = 50,
    quiet    :: Bool  = true
)
    # dimensions of problem
    n,p = size(x)

    # make vector of indices for folds
    test_idx = folds .== fold
    test_size = sum(test_idx)

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx
    mask_n    = convert(Vector{Int}, train_idx)
    mask_test = convert(Vector{Int}, test_idx)

    # compute the regularization path on the training set
    betas = iht_path(x, y, path, mask_n=mask_n, max_iter=max_iter, quiet=quiet, max_step=max_step, pids=pids, tol=tol)

    # tidy up
    gc()

    # preallocate vector for output
    myerrors = zeros(T, length(path))

    # allocate an index vector for b
    indices = falses(p)

    # allocate the arrays for the test set
    xb = SharedArray(T, n, init = S -> S[localindexes(S)] = zero(T), pids=pids)
    b  = SharedArray(T, p, init = S -> S[localindexes(S)] = zero(T), pids=pids)
    r  = SharedArray(T, n, init = S -> S[localindexes(S)] = zero(T), pids=pids)

    # compute the mean out-of-sample error for the TEST set
    # do this for every computed model in regularization path
    for i = 1:size(betas,2)

        # pull ith model in dense vector format
        b2 = full(vec(betas[:,i]))

        # copy it into SharedArray b
        copy!(b,b2)

        # indices stores Boolean indexes of nonzeroes in b
        update_indices!(indices, b)

        # compute estimated response Xb with $(path[i]) nonzeroes
        A_mul_B!(xb, x, b, indices, path[i], mask_test, pids=pids)

        # compute residuals
        difference!(r, y, xb)

        # mask data from training set
        # training set consists of data NOT in fold:
        # r[folds .!= fold] = zero(Float64)
        mask!(r, mask_test, 0, zero(T))

        # compute out-of-sample error as squared residual averaged over size of test set
        myerrors[i] = sumabs2(r) / test_size / 2
    end

    return myerrors :: Vector{T}
end

function pfold(
    T          :: Type,
    xfile      :: String,
    xtfile     :: String,
    x2file     :: String,
    yfile      :: String,
    meanfile   :: String,
    precfile   :: String,
    path       :: DenseVector{Int},
    folds      :: DenseVector{Int},
    pids       :: DenseVector{Int},
    q          :: Int;
    max_iter   :: Int  = 100,
    max_step   :: Int  = 50,
    quiet      :: Bool = true,
    header     :: Bool = false
)

    # ensure correct type
    T <: Float || throw(ArgumentError("Argument T must be either Float32 or Float64"))

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
                                pids = [worker]
                                x = BEDFile(T, xfile, xtfile, x2file, meanfile, precfile, pids=pids, header=header)
                                y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids)

                                one_fold(x, y, path, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids)
                        end # end remotecall_fetch()
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return (reduce(+, results[1], results) ./ q) :: Vector{T}
end

# default type for pfold is Float64
pfold(xfile::String, xtfile::String, x2file::String, yfile::String, meanfile::String, precfile::String, path::DenseVector{Int}, folds::DenseVector{Int}, pids::DenseVector{Int}, q::Int; max_iter::Int=100, max_step::Int =50, quiet::Bool=true, header::Bool=false) = pfold(Float64, xfile, xtfile, x2file, yfile, meanfile, precfile, path, folds, pids, q, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)

function pfold(
    T        :: Type,
    xfile    :: String,
    x2file   :: String,
    yfile    :: String,
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    pids     :: DenseVector{Int},
    q        :: Int;
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true,
    header   :: Bool = false
)

    # ensure correct type
    T <: Float || throw(ArgumentError("Argument T must be either Float32 or Float64"))

    # do not allow crossvalidation with fewer than 3 folds
    q > 2 || throw(ArgumentError("Number of folds q = $q must be at least 3."))

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
                                pids = [worker]
                                x = BEDFile(T, xfile, x2file, pids=pids, header=header)
                                y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids)

                                one_fold(x, y, path, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids)
                        end # end remotecall_fetch()
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return (reduce(+, results[1], results) ./ q) :: Vector{T}
end

# default for previous function is Float64
pfold(xfile::String, x2file::String, yfile::String, path::DenseVector{Int}, folds::DenseVector{Int}, pids::DenseVector{Int}, q::Int; max_iter::Int = 100, max_step::Int = 50, quiet::Bool = true, header::Bool = false) = pfold(Float64, xfile, x2file, yfile, path, folds, pids, q, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)



"""
    cv_iht(xfile, xtfile, x2file, yfile, meanfile, precfile, [q=max(3, min(CPU_CORES,5)), path=collect(1:min(p,20)), folds=cv_get_folds(n,q), pids=procs()])

This variant of `cv_iht()` performs `q`-fold crossvalidation with a `BEDFile` object loaded by `xfile`, `xtfile`, and `x2file`,
with column means stored in `meanfile` and column precisions stored in `precfile`.
The continuous response is stored in `yfile` with data particioned by the `Int` vector `folds`.
The folds are distributed across the processes given by `pids`.
The dimensions `n` and `p` are inferred from BIM and FAM files corresponding to the BED file path `xpath`.
"""
function cv_iht(
    T        :: Type,
    xfile    :: String,
    xtfile   :: String,
    x2file   :: String,
    yfile    :: String,
    meanfile :: String,
    precfile :: String;
    q        :: Int = max(3, min(CPU_CORES, 5)),
    path     :: DenseVector{Int} = begin
           # find p from the corresponding BIM file, then make path
            bimfile = xfile[1:(endof(xfile)-3)] * "bim"
            p       = countlines(bimfile)
            collect(1:min(20,p))
            end,
    folds    :: DenseVector{Int} = begin
           # find n from the corresponding FAM file, then make folds
            famfile = xfile[1:(endof(xfile)-3)] * "fam"
            n       = countlines(famfile)
            cv_get_folds(n, q)
            end,
    pids     :: DenseVector{Int} = procs(),
    tol      :: Float = convert(T, 1e-4),
    max_iter :: Int   = 100,
    max_step :: Int   = 50,
    quiet    :: Bool  = true,
    header   :: Bool  = false
)

    # enforce type
    T <: Float || throw(ArgumentError("Argument T must be either Float32 or Float64"))

    # how many elements are in the path?
    num_models = length(path)

    # compute folds in parallel
    mses = pfold(T, xfile, xtfile, x2file, yfile, meanfile, precfile, path, folds, pids, q, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)

    # what is the best model size?
    k = convert(Int, floor(mean(path[mses .== minimum(mses)])))

    # print results
    !quiet && print_cv_results(mses, path, k)

    # recompute ideal model
    # first load data on *all* processes
    x = BEDFile(T, xfile, xtfile, x2file, meanfile, precfile, header=header, pids=pids)
    y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids)

    # first use L0_reg to extract model
    output = L0_reg(x, y, k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, pids=pids)

    # which components of beta are nonzero?
    inferred_model = output.beta .!= zero(T)
    bidx = find(inferred_model)

    # allocate the submatrix of x corresponding to the inferred model
    x_inferred = zeros(T, x.geno.n, sum(inferred_model))
    decompress_genotypes!(x_inferred, x, inferred_model)

    # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
    xty = BLAS.gemv('T', one(T), x_inferred, y)
    xtx = BLAS.gemm('T', 'N', one(T), x_inferred, x_inferred)
    b   = zeros(T, length(bidx))
    try
        b = (xtx \ xty) :: Vector{T}
    catch e
        warn("in refit, caught error: ", e, "\nSetting returned values of b to -Inf")
        fill!(b, -Inf)
    end

    bids = prednames(x)[bidx]
    return IHTCrossvalidationResults{T}(mses, path, b, bidx, k, bids)
end

# encodes default type FLoat64 for previous function
### 22 Sep 2016: Julia v0.5 warns that this conflicts with cv_iht for GPUs
### since this is no longer the default interface for cv_iht with CPUs,
### then it is commented out here
#cv_iht(xfile::String, xtfile::String, x2file::String, yfile::String, meanfile::String, precfile::String; q::Int = max(3, min(CPU_CORES, 5)), path::DenseVector{Int} = begin bimfile=xfile[1:(endof(xfile)-3)] * "bim"; p=countlines(bimfile); collect(1:min(20,p)) end, folds::DenseVector{Int} = begin famfile=xfile[1:(endof(xfile)-3)] * "fam"; n=countlines(famfile); cv_get_folds(n, q) end, pids::DenseVector{Int}=procs(), tol::Float64=1e-4, max_iter::Int=100, max_step::Int=50, quiet::Bool=true, header::Bool=false) = cv_iht(Float64, xfile, xtfile, x2file, yfile, meanfile, precfile, path=path, folds=folds, q=q, pids=pids, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)

"""
    cv_iht(T::Type, xfile, x2file, yfile, [q=max(3, min(CPU_CORES,5)), path=collect(1:min(p,20)), folds=cv_get_folds(n,q), pids=procs()])

An abbreviated call to `cv_iht` that calculates means, precs, and transpose on the fly.
"""
function cv_iht(
    T        :: Type,
    xfile    :: String,
    x2file   :: String,
    yfile    :: String;
    q        :: Int = max(3, min(CPU_CORES, 5)),
    path     :: DenseVector{Int} = begin
           # find p from the corresponding BIM file, then make path
            bimfile = xfile[1:(endof(xfile)-3)] * "bim"
            p       = countlines(bimfile)
            collect(1:min(20,p))
            end,
    folds    :: DenseVector{Int} = begin
           # find n from the corresponding FAM file, then make folds
            famfile = xfile[1:(endof(xfile)-3)] * "fam"
            n       = countlines(famfile)
            cv_get_folds(n, q)
            end,
    pids     :: DenseVector{Int} = procs(),
    tol      :: Float = convert(T, 1e-4),
    max_iter :: Int   = 100,
    max_step :: Int   = 50,
    quiet    :: Bool  = true,
    header   :: Bool  = false
)

    # enforce type
    T <: Float || throw(ArgumentError("Argument T must be either Float32 or Float64"))

    # how many elements are in the path?
    num_models = length(path)

    # compute folds in parallel
    mses = pfold(T, xfile, x2file, yfile, path, folds, pids, q, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)

    # what is the best model size?
    k = convert(Int, floor(mean(path[mses .== minimum(mses)])))

    # print results
    !quiet && print_cv_results(mses, path, k)

    # recompute ideal model
    # first load data on *all* processes
    x = BEDFile(T, xfile, x2file, header=header, pids=pids)
    y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids)

    # first use L0_reg to extract model
    output = L0_reg(x, y, k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, pids=pids)

    # which components of beta are nonzero?
    inferred_model = output.beta .!= zero(T)
    bidx = find(inferred_model)

    # allocate the submatrix of x corresponding to the inferred model
    x_inferred = zeros(T, x.geno.n, sum(inferred_model))
    decompress_genotypes!(x_inferred, x, inferred_model)

    # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
    xty = BLAS.gemv('T', one(T), x_inferred, y)
    xtx = BLAS.gemm('T', 'N', one(T), x_inferred, x_inferred)
    b   = zeros(T, length(bidx))
    try
        b = (xtx \ xty) :: Vector{T}
    catch e
        warn("caught error: ", e, "\nSetting returned values of b to -Inf")
        fill!(b, -Inf)
    end

    bids = prednames(x)[bidx]
    return IHTCrossvalidationResults{T}(mses, path, b, bidx, k, bids)
end

"""
    cv_iht(xfile, x2file, yfile)

The default call to `cv_iht`. Here `xfile` points to the PLINK BED file stored on disk, `x2file` points to the nongenetic covariates stored in a delimited file, and `yfile` points to the response variable stored in a **binary** file.

Important optional arguments and defaults include:

- `q`, the number of crossvalidation folds. Defaults to `max(3, min(CPU_CORES,5))`
- `path`, an `Int` vector that contains the model sizes to test. Defaults to `collect(1:min(p,20))`, where `p` is the number of genetic predictors read from the PLINK BIM file.
- `folds`, an `Int` vector that specifies the fold structure. Defaults to `cv_get_folds(n,q)`, where `n` is the number of cases read from the PLINK FAM file.
- `pids`, an `Int` vector of process IDs. Defaults to `procs()`.
"""
cv_iht(xfile::String, x2file::String, yfile::String; q::Int = max(3, min(CPU_CORES, 5)), path::DenseVector{Int} = begin bimfile=xfile[1:(endof(xfile)-3)] * "bim"; p=countlines(bimfile); collect(1:min(20,p)) end, folds::DenseVector{Int} = begin famfile=xfile[1:(endof(xfile)-3)] * "fam"; n=countlines(famfile); cv_get_folds(n, q) end, pids::DenseVector{Int}=procs(), tol::Float64=1e-4, max_iter::Int=100, max_step::Int=50, quiet::Bool=true, header::Bool=false) = cv_iht(Float64, xfile, x2file, yfile, path=path, folds=folds, q=q, pids=pids, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)
