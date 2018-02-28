"""
    L0_reg(x::BEDFile, y, k, kernfile::String)

If supplied a `BEDFile` `x` and an OpenCL kernel file `kernfile` as an String, then `L0_reg` will attempt to accelerate the calculation of the dense gradient `x' * (y - x*b)` with a GPU device.
This variant introduces a host of extra arguments for the GPU encapsulated in an optional `PlinkGPUVariables` argument `v`.
The optional argument `v` facilitates the calculation of a regularization path by `iht_path`.
"""
function L0_reg{T <: Float, V <: DenseVector}(
    x        :: BEDFile{T},
    y        :: V,
    k        :: Int,
    kernfile :: String;
    pids     :: Vector{Int}       = procs(x),
    w     :: IHTVariables{T,V} = IHTVariables(x, y, k),
    mask_n   :: Vector{Int}       = ones(Int, size(y)),
    v        :: PLINK.PlinkGPUVariables{T} = PLINK.PlinkGPUVariables(w.df, x, y, kernfile, mask_n),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true
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
    mm_iter   = 0       # number of iterations of L0_reg
    mm_time   = zero(T) # compute time *within* L0_reg
    next_loss = zero(T) # loss function value

    # initialize floats
    current_loss = convert(T, Inf)    # tracks previous objective function value
    the_norm     = zero(T)            # norm(b - b0)
    scaled_norm  = zero(T)            # the_norm / (norm(b0) + 1)
    mu           = zero(T)            # Landweber step size, 0 < tau < 2/rho_max^2

    # initialize integers
    i       = 0         # used for iterations in loops
    mu_step = 0         # counts number of backtracking steps for mu

    # initialize booleans
    converged = false   # scaled_norm < tol?

    # update Xb, r, and gradient
    if sum(w.idx) == 0
        fill!(w.xb, zero(T))
        copy!(sdata(w.r), sdata(y))
        mask!(w.r, mask_n, 0, zero(T))
    else
        A_mul_B!(w.xb, x, w.b, w.idx, k, mask_n)
        difference!(w.r, y, w.xb)
        mask!(w.r, mask_n, 0, zero(T))
    end

    # calculate the gradient using the GPU
    At_mul_B!(w.df, x, w.r, mask_n, v)

    # update loss
    next_loss = convert(T, Inf)

    # formatted output to monitor algorithm progress
    !quiet && print_header()

    # main loop
    for mm_iter = 1:max_iter

        # notify and break if maximum iterations are reached.
        if mm_iter >= max_iter

            # alert about hitting maximum iterations
            !quiet && print_maxiter(max_iter, loss)

            # send elements below tol to zero
            threshold!(b, tol)

            # stop timer
            mm_time = toq()

            # these are output variables for function
            # wrap them into a Dict and return
            return IHTResults(mm_time, next_loss, mm_iter, copy(w.b))

            return output
        end

        # save values from previous iterate
        copy!(sdata(w.b0), sdata(w.b))   # b0 = b
        copy!(sdata(w.xb0), sdata(w.xb)) # Xb0 = Xb
        loss = next_loss

        # now perform IHT step
        (mu, mu_step) = iht!(w, x, y, k, nstep=max_step, iter=mm_iter, pids=pids)

        # update residuals
        difference!(w.r, y, w.xb)
        mask!(w.r, mask_n, 0, zero(T))

        # use updated residuals to recompute the gradient on the GPU
        At_mul_B!(w.df, x, w.r, mask_n, v)

        # update objective
        next_loss = sumabs2(sdata(w.r)) / 2

        # guard against numerical instabilities
        # ensure that objective is finite
        # if not, throw error
        check_finiteness(next_loss)

        # track convergence
        the_norm    = chebyshev(w.b, w.b0)
        scaled_norm = (the_norm / ( norm(w.b0,Inf) + 1)) :: T
        converged   = scaled_norm < tol

        # output algorithm progress
        quiet || @printf("%d\t%d\t%3.7f\t%3.7f\t%3.7f\n", mm_iter, mu_step, mu, the_norm, next_loss)

        # check for convergence
        # if converged and in feasible set, then algorithm converged before maximum iteration
        # perform final computations and output return variables
        if converged

            # send elements below tol to zero
            threshold!(w.b, tol)

            # stop time
            mm_time = toq()

            # announce convergence
            !quiet && print_convergence(mm_iter, next_loss, mm_time)

            # these are output variables for function
            return IHTResults(mm_time, next_loss, mm_iter, copy(w.b))
        end

        # algorithm is unconverged at this point, so check descent property
        # if objective increases, then abort
        if next_loss > current_loss + tol
            !quiet && print_descent_error(mm_iter, loss, next_loss)
            throw(ErrorException("Descent failure!"))
        end
    end # end main loop
end # end function



"""
    iht_path(x::BEDFile, y, k, kernfile::String)

If supplied a `BEDFile` `x` and an OpenCL kernel file `kernfile` as an String, then `iht_path` will attempt to accelerate the calculation of the dense gradient `x' * (y - x*b)` in `L0_reg` with a GPU device.
"""
function iht_path{T <: Float, V <: DenseVector}(
    x        :: BEDFile{T},
    y        :: DenseVector{T},
    path     :: DenseVector{Int},
    kernfile :: String;
    pids     :: DenseVector{Int}  = procs(x),
    w        :: IHTVariables{T,V} = IHTVariables(x, y, 1),
    mask_n   :: Vector{Int}       = ones(Int,length(y)),
    v        :: PLINK.PlinkGPUVariables{T} = PLINK.PlinkGPUVariables(w.df, x, y, kernfile, mask_n),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true
)
    # size of problem?
    n,p = size(x)

    # how many models will we compute?
    num_models = length(path)

    # also preallocate matrix to store betas
    betas = spzeros(T,p,num_models) # a matrix to store calculated models

    # compute the path
    @inbounds for i = 1:num_models

        # model size?
        q = path[i]

        # monitor progress
        quiet || print_with_color(:blue, "Computing model size $q.\n\n")

        # these arrays change in size from iteration to iteration
        # we must allocate them for every new model size
        update_variables!(w, x, q)

        # store projection of beta onto largest k nonzeroes in magnitude
        project_k!(w.b, q)

        # now compute current model
        output = L0_reg(x, y, q, kernfile, w=w, v=v, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids, mask_n=mask_n)

        # ensure that we correctly index the nonzeroes in b
        update_indices!(w.idx, output.beta)
        fill!(w.idx0, false)

        # put model into sparse matrix of betas
        betas[:,i] = sparsevec(output.beta)
    end

    return betas
end


"""
    one_fold(x::BEDFile, y, path, kernfile::String, folds, fold)

If supplied a `BEDFile` `x` and an OpenCL kernel file `kernfile` as an String, then `one_fold` will attempt to accelerate the calculation of the dense gradient `x' * (y - x*b)` in `L0_reg` with a GPU device.
"""
function one_fold{T <: Float}(
    x        :: BEDFile{T},
    y        :: DenseVector{T},
    path     :: DenseVector{Int},
    kernfile :: String,
    folds    :: DenseVector{Int},
    fold     :: Int;
    pids     :: Vector{Int} = procs(x),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    header   :: Bool = false,
    quiet    :: Bool = true
)
    # dimensions of problem
    n,p = size(x)

    # get list of available GPU devices
    # var device gets pointer to device indexed by variable devidx
    #device = cl.devices(:gpu)[devidx]

    # make vector of indices for folds
    test_idx = folds .== fold

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # how many indices are in test set?
    test_size = sum(test_idx)

    # GPU code requires Int variant of training indices, so do explicit conversion
    train_idx = convert(Vector{Int}, train_idx) :: Vector{Int}
    test_idx  = convert(Vector{Int}, test_idx)  :: Vector{Int}

    # compute the regularization path on the training set
    betas = iht_path(x, y, path, kernfile, max_iter=max_iter, quiet=quiet, max_step=max_step, mask_n=train_idx, pids=pids, tol=tol)

    # tidy up
    #gc()

    # preallocate vector for output
    myerrors = zeros(T, length(path))

    # allocate an index vector for b
    indices = falses(p)

    # allocate temporary arrays for the test set
    xb = SharedArray(T, n, pids=pids) :: SharedVector{T}
    b  = SharedArray(T, p, pids=pids) :: SharedVector{T}
    r  = SharedArray(T, n, pids=pids) :: SharedVector{T}

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
        #A_mul_B!(xb, x, b, indices, path[i], test_idx, pids=pids)
        A_mul_B!(xb, x, b, indices, path[i], test_idx)

        # compute residuals
        difference!(r, y, xb)

        # mask data from training set
        # training set consists of data NOT in fold:
        # r[folds .!= fold] = zero(Float64)
        mask!(r, test_idx, 0, zero(T))

        # compute out-of-sample error as squared residual averaged over size of test set
        myerrors[i] = sumabs2(r) / test_size / 2
    end

    return myerrors :: Vector{T}
end

# subroutine to calculate the approximate memory load of one fold on the GPU
function onefold_device_memload(x::BEDFile, wg_size::Int, y_chunks::Int; prec64::Bool = true)

    # floating point bytes multiplier depends on precision
    # prec64 = true -> use double precision (8 bytes per float)
    # prec64 = false -> use single precision (4 bytes per float)
    bytemult = ifelse(prec64, 8, 4)

    # get dimensions of problem
    n = x.n
    p = size(x,2)

    # start counting bytes
    # genotype array has bitstype Int8, one byte per component
    nx = length(x.x)

    # residual
    nr = bytemult * n

    # gradient, means, precisions
    ndf = 3 * bytemult * p

    # reduction vector
    nrv = bytemult * p * y_chunks

    # bitmask, which has bitstype Int32 (4 bytes per component)
    nbm = 4 * n

    # local memory footprint, just in case
    loc = bytemult * wg_size

    # total memory in Mb, rounded up to nearest int
    totmem = nx + ndf + nrv + nbm + loc

    return ceil(Int, totmem / 1024^2)
end

# subroutine to compute the number of folds that will fit on the GPU at one time
function compute_max_gpu_load(x::BEDFile, wg_size::Int, device::cl.Device; prec64::Bool = true)

    # number of chunks in residual
    y_chunks = div(x.n, wg_size) + (x.n % wg_size != 0 ? 1 : 0)

    # total available memory on current device
    gpu_memtot = ceil(Int, device[:global_mem_size] / 1024^2)

    # memory load of one CV fold on current device
    onefold_mem = onefold_device_memload(x,wg_size,y_chunks, prec64=prec64)

    # how many folds could we fit on the current GPU?
    max_folds = div(gpu_memtot, onefold_mem)

    return max_folds
end


"""
    pfold(xfile, xtfile, x2file,yfile, meanfile, precfile,path,kernfile,folds,q [, pids=procs(), devindices=ones(Int,q])

This function is the parallel execution kernel in `cv_iht()`. It is not meant to be called outside of `cv_iht()`.
It will distribute `q` crossvalidation folds across the processes supplied by the optional argument `pids` and call `one_fold()` for each fold.
Each fold will use the GPU device indexed by its corresponding component of the optional argument `devindices` to compute a regularization path given by `path`.
`pfold()` collects the vectors of MSEs returned by calling `one_fold()` for each process, reduces them, and returns their average across all folds.
"""
function pfold(
    T          :: Type,
    xfile      :: String,
    xtfile     :: String,
    x2file     :: String,
    yfile      :: String,
    meanfile   :: String,
    precfile   :: String,
    path       :: DenseVector{Int},
    kernfile   :: String,
    folds      :: DenseVector{Int},
    q          :: Int;
    devindices :: Vector{Int} = ones(Int,q),
    pids       :: Vector{Int} = procs(),
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

                        # grab index of GPU device
                        devidx = devindices[current_fold]

                        # report distribution of fold to worker and device
                        quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker and device $devidx.\n\n")

                        # launch job on worker
                        # worker loads data from file paths and then computes the errors in one fold
                        r = remotecall_fetch(worker) do
                                pids = [worker]
                                x = BEDFile(T, xfile, xtfile, x2file, meanfile, precfile, pids=pids, header=header) :: BEDFile{T}
                                y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids) :: SharedVector{T}

                                one_fold(x, y, path, kernfile, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids)
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


# default type for pfold is Float64
pfold(xfile::String, xtfile::String, x2file::String, yfile::String, meanfile::String, precfile::String, path::DenseVector{Int}, kernfile::String, folds::DenseVector{Int}, q::Int; devindices::Vector{Int}=ones(Int,q), pids::Vector{Int}=procs(), max_iter::Int=100, max_step::Int =50, quiet::Bool=true, header::Bool=false) = pfold(Float64, xfile, xtfile, x2file, yfile, meanfile, precfile, path, kernfile, folds, q, devindices=devindices, pids=pids, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)

"""
    cv_iht(xfile, xtfile, x2file, yfile, meanfile, precfile, kernfile)
This variant of `cv_iht()` uses a GPU to perform `q`-fold crossvalidation with a `BEDFile` object loaded by `xfile`, `xtfile`, and `x2file`,
with column means stored in `meanfile` and column precisions stored in `precfile`.
The continuous response is stored in the binary file `yfile`.
The calculations employ GPU acceleration by calling OpenCL kernels from `kernfile` with workgroup size `wg_size`.
"""
function cv_iht(
    T        :: Type,
    xfile    :: String,
    xtfile   :: String,
    x2file   :: String,
    yfile    :: String,
    meanfile :: String,
    precfile :: String,
    kernfile :: String;
    q        :: Int = cv_get_num_folds(3,5),
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
    pids     :: Vector{Int} = procs(),
    tol      :: Float = convert(T, 1e-4),
    max_iter :: Int   = 100,
    max_step :: Int   = 50,
    wg_size  :: Int   = 512,
    quiet    :: Bool  = true,
    header   :: Bool  = false
)
    # enforce type
    T <: Float || throw(ArgumentError("Argument T must be either Float32 or Float64"))

    # how many elements are in the path?
    num_models = length(path)

    # how many GPU devices are available to us?
    devs = cl.devices(:gpu)
    ndev = length(devs)

    # how many folds can we fit on a GPU at once?
    # count one less per GPU device, just in case
#   max_folds = zeros(Int, ndev)
#   @inbounds for i = 1:ndev
#       max_folds[i] = max(compute_max_gpu_load(x, wg_size, devs[i], prec64 = true) - 1, 0)
#   end

    # how many rounds of folds do we need to schedule?
#   fold_rounds = zeros(Int, ndev)
#   @inbounds for i = 1:ndev
#       fold_rounds[i] = div(q, max_folds[i]) + (q% max_folds[i] != 0 ? 1 : 0)
#   end

    # assign index of a GPU device for each fold
    # default is first GPU device (devidx = 1)
    devindices = ones(Int, q)
#   @inbounds for i = 1:q
#       devindices[i] += i % ndev
#   end

    # want to compute a path for each fold
    # the folds are computed asynchronously
    # only use the worker processes
    mses = pfold(T, xfile, xtfile, x2file, yfile, meanfile, precfile, path, kernfile, folds, q, max_iter=max_iter, max_step=max_step, quiet=quiet, devindices=devindices, pids=pids, header=header)

    # what is the best model size?
    k = path[indmin(mses)] :: Int

    # print results
    !quiet && print_cv_results(mses, path, k)

    # load data on *all* processes
    x = BEDFile(T, xfile, xtfile, x2file, meanfile, precfile, header=header, pids=pids) :: BEDFile{T}
    y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids) :: SharedVector{T}
    w = IHTVariables(x, y, k)
    m = ones(Int, size(y))
    v = PLINK.PlinkGPUVariables(w.df, x, y, kernfile, m)

    # first use L0_reg to extract model
    output = L0_reg(x, y, k, kernfile, w=w, mask_n=m, v=v, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids)

    # which components of beta are nonzero?
    inferred_model = output.beta .!= zero(T)
    bidx = find(inferred_model)

    # allocate the submatrix of x corresponding to the inferred model
    x_inferred = zeros(T, x.geno.n, sum(inferred_model))
    decompress_genotypes!(x_inferred, x, inferred_model)

    # refit best model
    b, bidx = refit_iht(x_inferred, y, k, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet)

    bids = prednames(x)[bidx]
    return IHTCrossvalidationResults{T}(mses, sdata(path), b, bidx, k, bids)
end

# default type for cv_iht is Float64
cv_iht(xfile::String, x2file::String, yfile::String, meanfile::String, precfile::String, kernfile::String; q::Int = cv_get_num_folds(3,5), path::DenseVector{Int} = begin bimfile=xfile[1:(endof(xfile)-3)] * "bim"; p=countlines(bimfile); collect(1:min(20,p)) end, folds::DenseVector{Int} = begin famfile=xfile[1:(endof(xfile)-3)] * "fam"; n=countlines(famfile); cv_get_folds(n, q) end, pids::Vector{Int}=procs(), tol::Float64=1e-4, max_iter::Int=100, max_step::Int=50, quiet::Bool=true, header::Bool=false) = cv_iht(Float64, xfile, x2file, yfile, meanfile, precfile, kernfile, path=path, folds=folds, q=q, pids=pids, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)



"""
    pfold(xfile, x2file, yfile, path, kernfile, folds, q [, pids=procs(), devindices=ones(Int,q])

`pfold` can also be called without the transposed genotype files or the mean/precision files.
In this case, it will attempt to precompute them before calling `one_fold`.
"""
function pfold(
    T          :: Type,
    xfile      :: String,
    x2file     :: String,
    yfile      :: String,
    path       :: DenseVector{Int},
    kernfile   :: String,
    folds      :: DenseVector{Int},
    q          :: Int;
    devindices :: Vector{Int} = ones(Int,q),
    pids       :: Vector{Int} = procs(),
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

                        # grab index of GPU device
                        devidx = devindices[current_fold]

                        # report distribution of fold to worker and device
                        quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker and device $devidx.\n\n")

                        # launch job on worker
                        # worker loads data from file paths and then computes the errors in one fold
                        r = remotecall_fetch(worker) do
                                pids = [worker]
                                x = BEDFile(T, xfile, x2file, pids=pids, header=header) :: BEDFile{T}
                                y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids) :: SharedVector{T}
                                one_fold(x, y, path, kernfile, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids)
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


# default type for pfold is Float64
pfold(xfile::String, x2file::String, yfile::String, path::DenseVector{Int}, kernfile::String, folds::DenseVector{Int}, q::Int; devindices::Vector{Int}=ones(Int,q), pids::Vector{Int}=procs(), max_iter::Int=100, max_step::Int =50, quiet::Bool=true, header::Bool=false) = pfold(Float64, xfile, x2file, yfile, path, kernfile, folds, q, devindices=devindices, pids=pids, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)

"""
    cv_iht(xfile,x2file,yfile,path,kernfile,folds,q [, pids=procs()])

If `cv_iht` is called without filepaths to the transposed genotype data or the means and precisions, then it will attempt to precompute them.
"""
function cv_iht(
    T        :: Type,
    xfile    :: String,
    x2file   :: String,
    yfile    :: String,
    kernfile :: String;
    q        :: Int = cv_get_num_folds(3,5),
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
    pids     :: Vector{Int} = procs(),
    tol      :: Float = convert(T, 1e-4),
    max_iter :: Int   = 100,
    max_step :: Int   = 50,
    wg_size  :: Int   = 512,
    quiet    :: Bool  = true,
    header   :: Bool  = false
)
    # enforce type
    T <: Float || throw(ArgumentError("Argument T must be either Float32 or Float64"))

    # how many elements are in the path?
    num_models = length(path)

    # how many GPU devices are available to us?
    devs = cl.devices(:gpu)
    ndev = length(devs)

    # how many folds can we fit on a GPU at once?
    # count one less per GPU device, just in case
#   max_folds = zeros(Int, ndev)
#   @inbounds for i = 1:ndev
#       max_folds[i] = max(compute_max_gpu_load(x, wg_size, devs[i], prec64 = true) - 1, 0)
#   end

    # how many rounds of folds do we need to schedule?
#   fold_rounds = zeros(Int, ndev)
#   @inbounds for i = 1:ndev
#       fold_rounds[i] = div(q, max_folds[i]) + (q% max_folds[i] != 0 ? 1 : 0)
#   end

    # assign index of a GPU device for each fold
    # default is first GPU device (devidx = 1)
    devindices = ones(Int, q)
#   @inbounds for i = 1:q
#       devindices[i] += i % ndev
#   end

    # want to compute a path for each fold
    # the folds are computed asynchronously
    # only use the worker processes
    mses = pfold(T, xfile, x2file, yfile, path, kernfile, folds, q, max_iter=max_iter, max_step=max_step, quiet=quiet, devindices=devindices, pids=pids, header=header)

    # what is the best model size?
    k = path[indmin(mses)] :: Int

    # print results
    !quiet && print_cv_results(mses, path, k)

    # recompute ideal model
    # first load data on *all* processes
    x = BEDFile(T, xfile, xtfile, x2file, header=header, pids=pids) :: BEDFile{T}
    y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids) :: SharedVector{T}
    w = IHTVariables(x, y, k)
    m = ones(Int, size(y))
    v = PLINK.PlinkGPUVariables(w.df, x, y, kernfile, m)

    # first use L0_reg to extract model
    output = L0_reg(x, y, k, kernfile, w=w, mask_n=m, v=v, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids)

    # which components of beta are nonzero?
    inferred_model = output.beta .!= zero(T)
    bidx = find(inferred_model)

    # allocate the submatrix of x corresponding to the inferred model
    x_inferred = zeros(T, x.geno.n, sum(inferred_model))
    decompress_genotypes!(x_inferred, x, inferred_model)

    # refit the best model
    b, bidx = refit_iht(x_inferred, y, k, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet)

    bids = prednames(x)[bidx]
    return IHTCrossvalidationResults{T}(mses, sdata(path), b, bidx, k, bids)
end

# default type for cv_iht is Float64
cv_iht(xfile::String, x2file::String, yfile::String, kernfile::String; q::Int = cv_get_num_folds(3,5), path::DenseVector{Int} = begin bimfile=xfile[1:(endof(xfile)-3)] * "bim"; p=countlines(bimfile); collect(1:min(20,p)) end, folds::DenseVector{Int} = begin famfile=xfile[1:(endof(xfile)-3)] * "fam"; n=countlines(famfile); cv_get_folds(n, q) end, pids::Vector{Int}=procs(), tol::Float64=1e-4, max_iter::Int=100, max_step::Int=50, quiet::Bool=true, header::Bool=false) = cv_iht(Float64, xfile, x2file, yfile, kernfile, path=path, folds=folds, q=q, pids=pids, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)
