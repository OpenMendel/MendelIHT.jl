export L0_reg_gpu
export iht_path

"A shortcut for `OpenCL` module name."
const cl = OpenCL

"""
    L0_reg(x::BEDFile, y, k, kernfile)

If supplied a `BEDFile` `x` and an OpenCL kernel file `kernfile` as an ASCIIString, then `L0_reg` will attempt to accelerate the calculation of the dense gradient `x' * (y - x*b)` with a GPU device. This variant introduces a host of extra arguments for the GPU. Most of these arguments are only meant to facilitate the calculation of a regularization path by `iht_path`. The optional arguments that a user will most likely wish to manipulate are:

- `device`, an `OpenCL.Device` object indicating the device to use in computations. Defaults to `last(OpenCL.devices(:gpu))`.
- `mask_n`, an `Int` vector of `0`s and `1`s indexing the rows of `x` and `y` that should be included or masked in the analysis. Defaults to `ones(Int,n)`, which includes all data.
- `wg_size` is the desired workgroup size for the GPU. Defaults to `512`.
"""
function L0_reg{T <: Float}(
    x           :: BEDFile{T},
    y           :: DenseVector{T},
    k           :: Int,
    kernfile    :: ASCIIString;
    pids        :: DenseVector{Int} = procs(),
    temp        :: IHTVariables{T}  = IHTVariables(x, y, k),
    n           :: Int              = length(y),
    p           :: Int              = size(x,2),
    tol         :: Float            = convert(T, 1e-4),
    max_iter    :: Int              = 100,
    max_step    :: Int              = 50,
    quiet       :: Bool             = true,
    mask_n      :: DenseVector{Int} = ones(Int,n),
    wg_size     :: Int              = 512,
    y_chunks    :: Int              = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0),
    y_blocks    :: Int              = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0),
    r_chunks    :: Int              = div(p*y_chunks, wg_size) + ((p*y_chunks) % wg_size != 0 ? 1 : 0),
    device      :: cl.Device        = last(cl.devices(:gpu)),
    ctx         :: cl.Context       = cl.Context(device),
    queue       :: cl.CmdQueue      = cl.CmdQueue(ctx),
    x_buff      :: cl.Buffer        = cl.Buffer(Int8,    ctx, (:r,  :copy), hostbuf = sdata(x.geno.x)),
    y_buff      :: cl.Buffer        = cl.Buffer(T, ctx, (:r,  :copy), hostbuf = sdata(r)),
    m_buff      :: cl.Buffer        = cl.Buffer(T, ctx, (:r,  :copy), hostbuf = sdata(x.means)),
    p_buff      :: cl.Buffer        = cl.Buffer(T, ctx, (:r,  :copy), hostbuf = sdata(x.precs)),
    df_buff     :: cl.Buffer        = cl.Buffer(T, ctx, (:rw, :copy), hostbuf = sdata(temp.df)),
    red_buff    :: cl.Buffer        = cl.Buffer(T, ctx, (:rw), p * y_chunks),
    mask_buff   :: cl.Buffer        = cl.Buffer(Int,    ctx, (:r,  :copy), hostbuf = sdata(mask_n)),
    genofloat   :: cl.LocalMem      = cl.LocalMem(T, wg_size),
    program     :: cl.Program       = cl.Program(ctx, source=kernfile) |> cl.build!,
    xtyk        :: cl.Kernel        = cl.Kernel(program, "compute_xt_times_vector"),
    rxtyk       :: cl.Kernel        = cl.Kernel(program, "reduce_xt_vec_chunks"),
    reset_x     :: cl.Kernel        = cl.Kernel(program, "reset_x"),
    wg_size32   :: Int32            = convert(Int32, wg_size),
    n32         :: Int32            = convert(Int32, n),
    p32         :: Int32            = convert(Int32, p),
    y_chunks32  :: Int32            = convert(Int32, y_chunks),
    y_blocks32  :: Int32            = convert(Int32, y_blocks),
    blocksize32 :: Int32            = convert(Int32, x.geno.blocksize),
    r_length32  :: Int32            = convert(Int32, p*y_chunks)
)

    # start timer
    tic()

    # first handle errors
    k        >= 0      || throw(ArgumentError("Value of k must be nonnegative!\n"))
    max_iter >= 0      || throw(ArgumentError("Value of max_iter must be nonnegative!\n"))
    max_step >= 0      || throw(ArgumentError("Value of max_step must be nonnegative!\n"))
    tol      >  eps(T) || throw(ArgumentError("Value of global tol must exceed machine precision!\n"))

    sum((mask_n .== 1) $ (mask_n .== 0)) == n || throw(ArgumentError("Argument mask_n can only contain 1s and 0s"))

    # initialize return values
    mm_iter   = 0       # number of iterations of L0_reg
    mm_time   = zero(T) # compute time *within* L0_reg
    next_loss = zero(T) # loss function value

    # initialize floats
    current_loss = oftype(tol,Inf)    # tracks previous objective function value
    the_norm     = zero(T)            # norm(b - b0)
    scaled_norm  = zero(T)            # the_norm / (norm(b0) + 1)
    mu           = zero(T)            # Landweber step size, 0 < tau < 2/rho_max^2

    # initialize integers
    i       = 0         # used for iterations in loops
    mu_step = 0         # counts number of backtracking steps for mu

    # initialize booleans
    converged = false   # scaled_norm < tol?

    # update Xb, r, and gradient
    if sum(temp.idx) == 0
        fill!(temp.xb, zero(T))
        copy!(temp.r, y)
        mask!(temp.r, mask_n, 0, zero(T), n=n)
    else
        A_mul_B!(temp.xb, x, temp.b, temp.idx, k, mask_n, pids=pids)
        difference!(temp.r, y, temp.xb)
        mask!(temp.r, mask_n, 0, zero(T), n=n)
    end

    # calculate the gradient using the GPU
#    xty!(df, df_buff, X, x_buff, r, y_buff, mask_n, mask_buff, queue, means, m_buff, invstds, p_buff, red_buff, xtyk, rxtyk, reset_x, wg_size, y_chunks, r_chunks, n, p, X.p2, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)
    At_mul_B!(temp.df, df_buff, x, x_buff, temp.r, y_buff, mask_n, mask_buff, queue, m_buff, p_buff, red_buff, xtyk, rxtyk, reset_x, wg_size, y_chunks, r_chunks, x.geno.n, x.geno.p, x.covar.p, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)

    # update loss
    next_loss = oftype(zero(T),Inf)

    # formatted output to monitor algorithm progress
    !quiet && print_header()

    # main loop
    for mm_iter = 1:max_iter

        # notify and break if maximum iterations are reached.
        if mm_iter >= max_iter

            # alert about hitting maximum iterations
            !quiet && print_maxiter(max_iter, loss)

            # send elements below tol to zero
            threshold!(b, tol, n=p)

            # stop timer
            mm_time = toq()

            # these are output variables for function
            # wrap them into a Dict and return
            return IHTResults(mm_time, next_loss, mm_iter, copy(temp.b))

            return output
        end

        # save values from previous iterate
        copy!(temp.b0, temp.b)             # b0 = b
        copy!(temp.xb0, temp.xb)           # Xb0 = Xb
        loss = next_loss

        # now perform IHT step
        (mu, mu_step) = iht!(temp, x, y, k, max_step=max_step, iter=mm_iter, pids=pids)

        # update residuals
        difference!(temp.r, y, temp.xb)
        mask!(temp.r, mask_n, 0, zero(T), n=n)

        # use updated residuals to recompute the gradient on the GPU
        At_mul_B!(temp.df, df_buff, x, x_buff, temp.r, y_buff, mask_n, mask_buff, queue, m_buff, p_buff, red_buff, xtyk, rxtyk, reset_x, wg_size, y_chunks, r_chunks, x.geno.n, x.geno.p, x.covar.p, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)

        # update objective
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

        # algorithm is unconverged at this point, so check descent property
        # if objective increases, then abort
        if next_loss > current_loss + tol
            !quiet && print_descent_error(mm_iter, loss, next_loss)
            throw(ErrorException("Descent failure!"))
        end
    end # end main loop
end # end function



"""
    iht_path(x::BEDFile, y , k ,kernfile)

If supplied a `BEDFile` `x` and an OpenCL kernel file `kernfile` as an ASCIIString, then `iht_path` will attempt to accelerate the calculation of the dense gradient `x' * (y - x*b)` in `L0_reg` with a GPU device. The new optional arguments include:

- `device`, an `OpenCL.Device` object indicating the device to use in computations. Defaults to `last(OpenCL.devices(:gpu))`.
- `mask_n`, an `Int` vector of `0`s and `1`s indexing the rows of `x` and `y` that should be included or masked in the analysis. Defaults to `ones(Int,n)`, which includes all data.
- `wg_size` is the desired workgroup size for the GPU. Defaults to `512`.
"""
function iht_path{T <: Float}(
    x        :: BEDFile,
    y        :: DenseVector{T},
    path     :: DenseVector{Int},
    kernfile :: ASCIIString;
    pids     :: DenseVector{Int} = procs(),
    mask_n   :: DenseVector{Int} = ones(Int,length(y)),
    device   :: cl.Device        = last(cl.devices(:gpu)),
    tol      :: Float            = convert(T, 1e-4),
    max_iter :: Int              = 100,
    max_step :: Int              = 50,
    wg_size  :: Int              = 512,
    quiet    :: Bool             = true
)
    # size of problem?
    n,p = size(x)

    # how many models will we compute?
    num_models = length(path)

    # preallocate temporary arrays
    temp = IHTVariables(x, y, 1)

    # also preallocate matrix to store betas
    betas       = spzeros(T,p,num_models) # a matrix to store calculated models

    # allocate GPU variables
    y_chunks    = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0)
    y_blocks    = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0)
    r_chunks    = div(p*y_chunks, wg_size) + ((p*y_chunks) % wg_size != 0 ? 1 : 0)
    ctx         = cl.Context(device)
    queue       = cl.CmdQueue(ctx)
    program     = cl.Program(ctx, source=kernfile) |> cl.build!
    xtyk        = cl.Kernel(program, "compute_xt_times_vector")
    rxtyk       = cl.Kernel(program, "reduce_xt_vec_chunks")
    reset_x     = cl.Kernel(program, "reset_x")
    wg_size32   = convert(Int32, wg_size)
    n32         = convert(Int32, n)
    p32         = convert(Int32, p)
    y_chunks32  = convert(Int32, y_chunks)
    y_blocks32  = convert(Int32, y_blocks)
    blocksize32 = convert(Int32, x.geno.blocksize)
    r_length32  = convert(Int32, p*y_chunks)
    x_buff      = cl.Buffer(Int8, ctx, (:r,  :copy), hostbuf = sdata(x.geno.x))
    m_buff      = cl.Buffer(T,    ctx, (:r,  :copy), hostbuf = sdata(x.means))
    p_buff      = cl.Buffer(T,    ctx, (:r,  :copy), hostbuf = sdata(x.precs))
    y_buff      = cl.Buffer(T,    ctx, (:r,  :copy), hostbuf = sdata(temp.r))
    df_buff     = cl.Buffer(T,    ctx, (:rw, :copy), hostbuf = sdata(temp.df))
    mask_buff   = cl.Buffer(Int,  ctx, (:rw, :copy), hostbuf = sdata(mask_n))
    red_buff    = cl.Buffer(T,    ctx, (:rw),        p * y_chunks)
    genofloat   = cl.LocalMem(T,  wg_size)

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
        output = L0_reg(x,y,q,kernfile, temp=temp, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, wg_size=wg_size, y_chunks=y_chunks, y_blocks=y_blocks, r_chunks=r_chunks, device=device, ctx=ctx, queue=queue, x_buff=x_buff, y_buff=y_buff, m_buff=m_buff, p_buff=p_buff, df_buff=df_buff, red_buff=red_buff, genofloat=genofloat, program=program, xtyk=xtyk, rxtyk=rxtyk, reset_x=reset_x, wg_size32=wg_size32, n32=n32, p32=p32, y_chunks32=y_chunks32, y_blocks32=y_blocks32, blocksize32=blocksize32, r_length32=r_length32, mask_n=mask_n, mask_buff=mask_buff, pids=pids)

        # ensure that we correctly index the nonzeroes in b
        update_indices!(temp.idx, output.beta)
        fill!(temp.idx0, false)

        # put model into sparse matrix of betas
        betas[:,i] = sparsevec(output.beta)
    end

    return betas
end


"""
    one_fold(x::BEDFile, y, path, kernfile, folds, fold)

If supplied a `BEDFile` `x` and an OpenCL kernel file `kernfile` as an ASCIIString, then `one_fold` will attempt to accelerate the calculation of the dense gradient `x' * (y - x*b)` in `L0_reg` with a GPU device. The new optional arguments include:

- `devidx`, an index indicating the GPU device to use in computations. The device is chosen  as `OpenCL.devices(:gpu)[devidx]`. Defaults to `1` (choose the first GPU device)
- `wg_size` is the desired workgroup size for the GPU. Defaults to `512`.
- `header` is a `Bool` to feed to `readdlm` when loading the nongenetic covariates `x.x2`. Defaults to `false` (no header).
"""
function one_fold{T <: Float}(
    x        :: BEDFile,
    y        :: DenseVector{T},
    path     :: DenseVector{Int},
    kernfile :: ASCIIString,
    folds    :: DenseVector{Int},
    fold     :: Int;
    pids     :: DenseVector{Int} = procs(),
    means    :: DenseVector{T}   = mean(T,x, shared=true, pids=pids),
    invstds  :: DenseVector{T}   = invstd(x,means, shared=true, pids=pids),
    tol      :: Float            = convert(T, 1e-4),
    max_iter :: Int              = 100,
    max_step :: Int              = 50,
    n        :: Int              = length(y),
    p        :: Int              = size(x,2),
    wg_size  :: Int              = 512,
    devidx   :: Int              = 1,
    header   :: Bool             = false,
    quiet    :: Bool             = true
)

    # get list of available GPU devices
    # var device gets pointer to device indexed by variable devidx
    device = cl.devices(:gpu)[devidx]

    # make vector of indices for folds
    test_idx = folds .== fold

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # how many indices are in test set?
    test_size = sum(test_idx)

    # GPU code requires Int variant of training indices, so do explicit conversion
    train_idx = convert(Vector{Int}, train_idx)
    test_idx  = convert(Vector{Int}, test_idx)

    # compute the regularization path on the training set
    betas = iht_path(x,y,path,kernfile, max_iter=max_iter, quiet=quiet, max_step=max_step, means=means, invstds=invstds, mask_n=train_idx, wg_size=wg_size, device=device, pids=pids, tol=tol, max_iter=max_iter, n=n, p=p)

    # tidy up
    gc()

    # preallocate vector for output
    myerrors = zeros(T, length(path))

    # allocate an index vector for b
    indices = falses(p)

    # allocate temporary arrays for the test set
    Xb = SharedArray(T, n, pids=pids)
    b  = SharedArray(T, p, pids=pids)
    r  = SharedArray(T, n, pids=pids)

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

        # compute residuals
        difference!(r,y,Xb)

        # mask data from training set
        # training set consists of data NOT in fold:
        # r[folds .!= fold] = zero(Float64)
        mask!(r, test_idx, 0, zero(T), n=n)

        # compute out-of-sample error as squared residual averaged over size of test set
        myerrors[i] = sumabs2(r) / test_size
    end

    return myerrors
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
    pfold(xfile, xtfile, x2file,yfile, meanfile, invstdfile,path,kernfile,folds,q [, pids=procs(), devindices=ones(Int,q])

This function is the parallel execution kernel in `cv_iht()`. It is not meant to be called outside of `cv_iht()`.
It will distribute `q` crossvalidation folds across the processes supplied by the optional argument `pids` and call `one_fold()` for each fold.
Each fold will use the GPU device indexed by its corresponding component of the optional argument `devindices` to compute a regularization path given by `path`.
`pfold()` collects the vectors of MSEs returned by calling `one_fold()` for each process, reduces them, and returns their average across all folds.
"""
function pfold(
    T          :: Type,
    xfile      :: ASCIIString,
    xtfile     :: ASCIIString,
    x2file     :: ASCIIString,
    yfile      :: ASCIIString,
    meanfile   :: ASCIIString,
    invstdfile :: ASCIIString,
    path       :: DenseVector{Int},
    kernfile   :: ASCIIString,
    folds      :: DenseVector{Int},
    q          :: Int;
    devindices :: DenseVector{Int} = ones(Int,q),
    pids       :: DenseVector{Int} = procs(),
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

                        # grab index of GPU device
                        devidx = devindices[current_fold]

                        # report distribution of fold to worker and device
                        quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker and device $devidx.\n\n")

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

                                one_fold(x, y, path, kernfile, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, means=means, invstds=invstds, devidx=devidx, pids=pids)
                        end # end remotecall_fetch()
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return reduce(+, results[1], results) ./ q
end


# default type for pfold is Float64
pfold(xfile::ASCIIString, xtfile::ASCIIString, x2file::ASCIIString, yfile::ASCIIString, meanfile::ASCIIString, invstdfile::ASCIIString, path::DenseVector{Int}, kernfile::ASCIIString, folds::DenseVector{Int}, q::Int; devindices::DenseVector{Int}=ones(Int,q), pids::DenseVector{Int}=procs(), max_iter::Int=100, max_step::Int =50, quiet::Bool=true, header::Bool=false) = pfold(Float64, xfile, xtfile, x2file, yfile, meanfile, invstdfile, path, kernfile, folds, q, devindices=devindices, pids=pids, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)


"""
    cv_iht(xfile,xtfile,x2file,yfile,meanfile,invstdfile,path,kernfile,folds,q [, pids=procs(), wg_size=512])

This variant of `cv_iht()` performs `q`-fold crossvalidation with a `BEDFile` object loaded by `xfile`, `xtfile`, and `x2file`,
with column means stored in `meanfile` and column precisions stored in `invstdfile`.
The continuous response is stored in `yfile` with data particioned by the `Int` vector `folds`.
The calculations employ GPU acceleration by calling OpenCL kernels from `kernfile` with workgroup size `wg_size`.
The folds are distributed across the processes given by `pids`.
"""
function cv_iht(
    T             :: Type,
    xfile         :: ASCIIString,
    xtfile        :: ASCIIString,
    x2file        :: ASCIIString,
    yfile         :: ASCIIString,
    meanfile      :: ASCIIString,
    invstdfile    :: ASCIIString,
    path          :: DenseVector{Int},
    kernfile      :: ASCIIString,
    folds         :: DenseVector{Int},
    q             :: Int;
    pids          :: DenseVector{Int} = procs(),
    tol           :: Float            = convert(T, 1e-4),
    max_iter      :: Int              = 100,
    max_step      :: Int              = 50,
    wg_size       :: Int              = 512,
    quiet         :: Bool             = true,
    compute_model :: Bool             = false,
    header        :: Bool             = false
)
    #0 <= length(path) <= p || throw(ArgumentError("Path length must be positive and cannot exceed number of predictors"))
    T <: Float             || throw(ArgumentError("Argument T must be either Float32 or Float64"))

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
    errors = pfold(T, xfile, xtfile, x2file, yfile, meanfile, invstdfile, path, kernfile, folds, q, max_iter=max_iter, max_step=max_step, quiet=quiet, devindices=devindices, pids=pids, header=header)

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

        # load data on *all* processes
        x       = BEDFile(T, xfile, xtfile, x2file, header=header, pids=pids)
        n       = x.n
        p       = size(x,2)
        y       = SharedArray(abspath(yfile), T, (n,), pids=pids)
        means   = SharedArray(abspath(meanfile), T, (p,), pids=pids)
        invstds = SharedArray(abspath(invstdfile), T, (p,), pids=pids)

        # initialize parameter vector as SharedArray
        b = SharedArray(T, p)

        # first use L0_reg to extract model
        output = L0_reg(x,y,k,kernfile, n=n, p=p, b=b, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, means=means, invstds=invstds, wg_size=wg_size, device=devs[1])

        # which components of beta are nonzero?
        inferred_model = output["beta"] .!= zero(T)
        bidx = find( x -> x .!= zero(T), b)

        # allocate the submatrix of x corresponding to the inferred model
        x_inferred = zeros(T,n,sum(inferred_model))
        decompress_genotypes!(x_inferred,x,inferred_model,means=means,invstds=invstds)

        # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
        xty = BLAS.gemv('T', one(T), x_inferred, y)
        xtx = BLAS.gemm('T', 'N', one(T), x_inferred, x_inferred)
        b = xtx \ xty
        return errors, b, bidx
    end
    return errors
end

# default type for cv_iht is Float64
cv_iht(xfile::ASCIIString, xtfile::ASCIIString, x2file::ASCIIString, yfile::ASCIIString, meanfile::ASCIIString, invstdfile::ASCIIString, path::DenseVector{Int}, kernfile::ASCIIString, folds::DenseVector{Int}, q::Int; pids::DenseVector{Int}=procs(), tol::Float=1e-4, max_iter::Int=100, max_step::Int=50, wg_size::Int=512, quiet::Bool=true, compute_model::Bool=false, header::Bool=false) = cv_iht(Float64, xfile, xtfile, x2file, yfile, meanfile, invstdfile, path, kernfile, folds, q, pids=pids, tol=tol, max_iter=max_iter, max_step=max_step, wg_size=wg_size, quiet=quiet, compute_model=compute_model, header=header)
