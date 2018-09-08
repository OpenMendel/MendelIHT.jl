"""
    iht!(v::IHTVariables, x::BEDFile, y, k)

If used with a `BEDFile` object `x`, then the temporary arrays `b0`, `Xb`, `Xb0`, and `r` housed in the `IHTVariables` object `v` are all initialized as `SharedArray`s of the proper dimensions.
The additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs(x)`.
"""
function iht!(
    v     :: IHTVariables{T},
    x     :: BEDFile{T},
    y     :: DenseVector{T},
    k     :: Int;
    pids  :: Vector{Int} = procs(x),
    iter  :: Int = 1,
    nstep :: Int = 50,
) where {T <: Float}

    # compute indices of nonzeroes in beta
    _iht_indices(v, k)

    # if support has not changed between iterations,
    # then xk and gk are the same as well
    # avoid extracting and computing them if they have not changed
    # one exception: we should always extract columns on first iteration

    if !isequal(v.idx, v.idx0) || iter < 2
        decompress_genotypes!(v.xk, x, v.idx)
    end

#GORDON
#
if iter <= 2
    print_with_color(:red, "gwas#31, Starting iht!().\n")
        print_with_color(:red, "\tsize(v.gk) = $(size(v.gk))")
        print_with_color(:red, "\tsize(v.df) = $(size(v.df))")
        print_with_color(:red, "\tsize(v.xk) = $(size(v.xk))")
        print_with_color(:red, "\tsize(v.df[v.idx]) = $(size(v.df[v.idx]))\n")
end
#

    #v.idx .= v.b .!= 0  # idea for Gordon

    # store relevant components of gradient
    v.gk .= v.df[v.idx]
    #GORDON
    #println("After v.gk .= v.df[v.idx]")
    # now compute subset of x*g
    A_mul_B!(v.xgk, v.xk, v.gk)

    # warn if xgk only contains zeros
    all(v.xgk .== zero(T)) && warn("Entire active set has values equal to 0")

    # compute step size
    μ = (sum(abs2, v.gk) / sum(abs2, v.xgk)) :: T
#    μ = _iht_stepsize(v, k) :: T

    # notify problems with step size
    @assert isfinite(μ) "Step size is not finite, is active set all zero?"
    @assert μ > eps(typeof(μ)) "Step size $(μ) is below machine precision, algorithm may not converge correctly"

    # compute gradient step
    _iht_gradstep(v, μ, k)

    # update xb
    PLINK.A_mul_B!(v.xb, x, v.b, v.idx, k)

    # calculate omega
    ω_top, ω_bot = _iht_omega(v)

    # backtrack until mu sits below omega and support stabilizes
    μ_step = 0
    while _iht_backtrack(v, ω_top, ω_bot, μ, μ_step, nstep)

        # stephalving
        μ /= 2

        # stop if mu falls below machine epsilon
        @assert μ > eps(typeof(μ)) "Step size $(μ) is below machine precision, algorithm may not converge correctly"

        # recompute gradient step
        copy!(sdata(v.b), sdata(v.b0))
        _iht_gradstep(v, μ, k)

        # recompute xb
        PLINK.A_mul_B!(v.xb, x, v.b, v.idx, k)

        # calculate omega
        ω_top, ω_bot = _iht_omega(v)

        # increment the counter
        μ_step += 1
    end

    return μ::T, μ_step::Int
end

"""
    L0_reg(x::BEDFile, y, k)

If used with a `BEDFile` object `x`, then the temporary floating point arrays are all initialized as `SharedArray`s of the proper dimensions.
The additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs()`.
- `mask_n`, an `Int` vector used as a bitmask for crossvalidation purposes. Defaults to a vector of ones.
"""
function L0_reg(
    x        :: BEDFile{T}, # note: x.covar.x is the last 2 column of the fam file
    y        :: V,
    k        :: Int;
    pids     :: Vector{Int} = procs(x),
    v        :: IHTVariables{T, V} = IHTVariables(x, y, k),
    mask_n   :: Vector{Int} = ones(Int, size(y)),
    tol      :: T     = convert(T, 1e-4),
    max_iter :: Int   = 100,
    max_step :: Int   = 50,
    quiet    :: Bool  = true
) where {T <: Float, V <: DenseVector}
    print_with_color(:red, "gwas#112, Starting L0_reg().\n")
    println("size(x) = $(size(x))")
    println("size(y) = $(size(y))")
    println("sum(mask_n) = $(sum(mask_n))")
    # start timer
    tic()

    # first handle errors
    @assert k >= 0        "Value of k must be nonnegative!\n"
    @assert max_iter >= 0 "Value of max_iter must be nonnegative!\n"
    @assert max_step >= 0 "Value of max_step must be nonnegative!\n"
    @assert tol >  eps(T) "Value of global tol must exceed machine precision!\n"
    n = length(y)
    @assert sum(xor.((mask_n .== 1),(mask_n .== 0))) == n "Argument mask_n can only contain 1s and 0s"
    @assert procs(x) == procs(y) == pids "Processes involved in arguments x, y must match those in keyword pids"

    # initialize return values
    mm_iter   = 0                 # number of iterations of L0_reg
    mm_time   = zero(T)           # compute time *within* L0_reg
    next_loss = convert(T, Inf)   # loss function value

    # initialize floats
    loss        = convert(T, Inf) # tracks previous objective function value
    the_norm    = zero(T)         # norm(b - b0)
    scaled_norm = zero(T)         # the_norm / (norm(b0) + 1)
    mu          = zero(T)         # Landweber step size, 0 < tau < 2/rho_max^2

    # initialize integers
    mu_step = 0                   # counts number of backtracking steps for mu

    # initialize booleans
    converged = false             # scaled_norm < tol?

    # update xb, r, and gradient
#    initialize_xb_r_grad!(v, x, y, k, pids=pids)
    if sum(v.idx) == 0
        fill!(v.xb, zero(T))
        copy!(sdata(v.r), sdata(y))
        mask!(v.r, mask_n, 0, zero(T))
    else
        A_mul_B!(v.xb, x, v.b, v.idx, k, mask_n)
        difference!(v.r, y, v.xb)
        #v.r .= y .- v.xb # v.r = (y - Xβ - intercept)
        #v.r[mask_n .== 0] .= 0 #bit masking, idk why we need this yet
        mask!(v.r, mask_n, 0, zero(T))
    end

    # calculate the gradient
    PLINK.At_mul_B!(v.df, x, v.r, mask_n, pids=pids)

    # formatted output to monitor algorithm progress
    !quiet && print_header()

    # main loop
    print_with_color(:green, "gwas#171, Starting max_iter loop in L0_reg().\n")
        print_with_color(:green, "\tsize(v.gk) = $(size(v.gk))")
        print_with_color(:green, "\tsize(v.df) = $(size(v.df))")
        print_with_color(:green, "\tsize(v.xk) = $(size(v.xk))")
        print_with_color(:green, "\tsize(v.df[v.idx]) = $(size(v.df[v.idx]))\n")

    for mm_iter = 1:max_iter

        # notify and break if maximum iterations are reached.
        if mm_iter >= max_iter

            # alert about hitting maximum iterations
            !quiet && print_maxiter(max_iter, loss)

            # send elements below tol to zero
            threshold!(v.b, tol)

            # stop timer
            mm_time = toq()

            # these are output variables for function
            # wrap them into a Dict and return
            return IHTResults(mm_time, next_loss, mm_iter, copy(v.b))
        end

        # save values from previous iterate
        copy!(sdata(v.b0), sdata(v.b))   # b0 = b
        copy!(sdata(v.xb0), sdata(v.xb)) # Xb0 = Xb
        loss = next_loss

        # now perform IHT step
        (μ, μ_step) = iht!(v, x, y, k, nstep=max_step, iter=mm_iter, pids=pids)

        # the IHT kernel gives us an updated x*b
        # use it to recompute residuals and gradient
#        update_r_grad!(v, x, y, pids=pids)
        v.r .= y .- v.xb
        mask!(v.r, mask_n, 0, zero(T))

        # use updated residuals to recompute the gradient on the GPU
        PLINK.At_mul_B!(v.df, x, v.r, mask_n, pids=pids)

        # update loss, objective, and gradient
        next_loss = sum(abs2, sdata(v.r)) / 2

        # guard against numerical instabilities
        # ensure that objective is finite
        # if not, throw error
        check_finiteness(next_loss)

        # track convergence
        the_norm    = chebyshev(v.b, v.b0)
        scaled_norm = (the_norm / ( norm(v.b0,Inf) + one(T))) :: T
        converged   = scaled_norm < tol

        # output algorithm progress
        quiet || @printf("%d\t%d\t%3.7f\t%3.7f\t%3.7f\n", mm_iter, μ_step, μ, the_norm, next_loss)

        # check for convergence
        # if converged and in feasible set, then algorithm converged before maximum iteration
        # perform final computations and output return variables
        if converged

            # send elements below tol to zero
            threshold!(v.b, tol)

            # stop time
            mm_time = toq()

            # announce convergence
            !quiet && print_convergence(mm_iter, next_loss, mm_time)

            # these are output variables for function
            return IHTResults(mm_time, next_loss, mm_iter, copy(v.b))
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
function iht_path(
    x        :: BEDFile{T},
    y        :: DenseVector{T},
    path     :: DenseVector{Int};
    pids     :: Vector{Int} = procs(x),
    mask_n   :: Vector{Int} = ones(Int, size(y)),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true
) where {T <: Float}
    print_with_color(:red, "gwas266, Starting iht_path().\n")
    # size of problem?
    n,p = size(x)

    # how many models will we compute?
    nmodels = length(path)

    # also preallocate matrix to store betas
    betas = spzeros(T,p,nmodels)  # a matrix to store calculated models

    # preallocate temporary arrays
    v = IHTVariables(x, y, 1)

    # compute the path
    @inbounds for i = 1:nmodels

        # model size?
        q = path[i]

        # monitor progress
        quiet || print_with_color(:blue, "Computing model size $q.\n\n")

        # these arrays change in size from iteration to iteration
        # we must allocate them for every new model size
        update_variables!(v, x, q)

        # store projection of beta onto largest k nonzeroes in magnitude
        project_k!(v.b, q)

        # now compute current model
        output = L0_reg(x, y, q, v=v, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids, mask_n=mask_n)

        # ensure that we correctly index the nonzeroes in b
        update_indices!(v.idx, output.beta)
        fill!(v.idx0, false)

        # put model into sparse matrix of betas
        betas[:,i] = sparsevec(output.beta)
    end

    # return a sparsified copy of the models
    return betas
end

"""
    iht_path(x::BEDFile, y, path)

If used with a `BEDFile` object `x`, then the temporary arrays are all initialized as `SharedArray`s of the proper dimensions.
The additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs(x)`.
- `mask_n`, an `Int` vector used as a bitmask for crossvalidation purposes. Defaults to a vector of ones.
"""
function iht_path(
    x        :: SnpLike{2},
    y        :: Array{T,1},
    path     :: DenseVector{Int};
    pids     :: Vector{Int} = procs(x),
    mask_n   :: Vector{Int} = ones(Int, size(y)),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true
) where {T <: Float}
    print_with_color(:red, "gwas266, Starting iht_path().\n")
    # size of problem?
    n,p = size(x)

    # how many models will we compute?
    nmodels = length(path)

    # also preallocate matrix to store betas
    betas = spzeros(T,p,nmodels)  # a matrix to store calculated models

    # preallocate temporary arrays
    J = 1
    v = IHTVariables(x, y, J, 1) # call Ben's code here, returns a different v

    # compute the path
    @inbounds for i = 1:nmodels

        # model size?
        q = path[i]

        # monitor progress
        quiet || print_with_color(:blue, "Computing model size $q.\n\n")

        # these arrays change in size from iteration to iteration
        # we must allocate them for every new model size
        update_variables!(v, x, J*q)

        # store projection of beta onto largest k nonzeroes in magnitude
        project_k!(v.b, q)

        # now compute current model
        #output = L0_reg(x, y, q, v=v, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=pids, mask_n=mask_n)
        #Gordon
        keyword = set_keyword_defaults!(Dict{AbstractString, Any}())
        keyword["data_type"] = ""
        keyword["predictors_per_group"] = ""
        keyword["manhattan_plot_file"] = ""
        keyword["max_groups"] = ""
        keyword["group_membership"] = ""

        keyword["prior_weights"] = ""
        keyword["pw_algorithm_value"] = 1.0     # not user defined at this time

#        file_name = xfile[1:end-4]
    # try xt_test
#        snpmatrix = SnpArray("xt_test")
        # HERE I READ THE PHENOTYPE FROM THE BEDFILE TO MATCH THE TUTORIAL RESULTS
        # I'M SURE THERE IS A BETTER WAY TO GET IT
        #phenotype is already set in tutorial_simulation.jl above  DIDN'T WORK - SAYS IT'S NOT DEFINED HERE
        #phenotype = readdlm(file_name * ".fam", header = false)[:, 6] # NO GOOD, THE PHENOTYPE HERE IS ALL ONES
#=
        x = BEDFile(T, xfile, x2file, header=header, pids=[1]) :: BEDFile{T}
        y = SharedArray{T}(abspath(yfile), (x.geno.n,), pids=[1]) :: SharedVector{T}
        phenotype = convert(Array{T,1}, y)
=#
        #println("phenotype = $(phenotype)")
        # y_copy = copy(phenotype)
        # y_copy .-= mean(y_copy)
        groups = fill(1,24000)
        k = 10
        J = 1
        snpmatrix = x
        phenotype = y
        outputg = L0_reg(snpmatrix, phenotype, J, k, groups, keyword)
        #println("outputg.beta = $(outputg.beta)")
        found = find(outputg.beta .!= 0.0)
        println("betas found in xt_test = $(found)")


        # ensure that we correctly index the nonzeroes in b
        update_indices!(v.idx, outputg.beta)
        fill!(v.idx0, false)

        # put model into sparse matrix of betas
        betas[:,i] = sparsevec(outputg.beta)
    end

    # return a sparsified copy of the models
    return betas
end


"""
    one_fold(x::BEDFile, y, path, folds, fold)

If used with a `BEDFile` object `x`, then the additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs(x)`.
"""
function one_fold(
    x        :: BEDFile{T},
    y        :: DenseVector{T},
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    fold     :: Int;
    pids     :: Vector{Int} = procs(x),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true
) where {T <: Float}
    # dimensions of problem
    n,p = size(x)
    print_with_color(:red, "gwas332, Starting one_fold().\n")

    # make vector of indices for folds
    test_idx = folds .== fold
    test_size = sum(test_idx)

    # train_idx is the vector that indexes the TRAINING set
    train_idx = .!test_idx
    mask_n    = convert(Vector{Int}, train_idx)
    mask_test = convert(Vector{Int}, test_idx)
    #Gordon
    println("test_size = $(test_size)")
    #println("train_idx = $(train_idx)")
    println()
    println()
    #println("test_idx = $(test_idx)")

    # compute the regularization path on the training set
    betas = iht_path(x, y, path, mask_n=mask_n, max_iter=max_iter, quiet=quiet, max_step=max_step, pids=pids, tol=tol)

    # tidy up
    #gc()

    # preallocate vector for output
    myerrors = zeros(T, length(path))

    # allocate an index vector for b
    indices = falses(p)

    # allocate the arrays for the test set
    xb = SharedArray{T}((n,), init = S -> S[localindexes(S)] = zero(T), pids=pids) :: SharedVector{T}
    b  = SharedArray{T}((p,), init = S -> S[localindexes(S)] = zero(T), pids=pids) :: SharedVector{T}
    r  = SharedArray{T}((n,), init = S -> S[localindexes(S)] = zero(T), pids=pids) :: SharedVector{T}

    # compute the mean out-of-sample error for the TEST set
    # do this for every computed model in regularization path
    for i = 1:size(betas,2)

        # pull ith model in dense vector format
        b2 = full(vec(betas[:,i]))

        # copy it into SharedArray b
        copy!(sdata(b),sdata(b2))

        # indices stores Boolean indexes of nonzeroes in b
        update_indices!(indices, b)

        # compute estimated response Xb with $(path[i]) nonzeroes
        #A_mul_B!(xb, x, b, indices, path[i], mask_test, pids=pids)
        A_mul_B!(xb, x, b, indices, path[i], mask_test)

        # compute residuals
        r .= y .- xb

        # mask data from training set
        # training set consists of data NOT in fold:
        # r[folds .!= fold] = zero(T)
        mask!(r, mask_test, 0, zero(T))

        # compute out-of-sample error as squared residual averaged over size of test set
        myerrors[i] = sum(abs2, r) / test_size / 2
    end

    return myerrors :: Vector{T}
end

"""
    one_fold(x::BEDFile, y, path, folds, fold)

If used with a `BEDFile` object `x`, then the additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs(x)`.
"""
function one_fold(
    x        :: SnpLike{2},
    y        :: Array{T,1},
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    fold     :: Int;
    pids     :: Vector{Int} = procs(x),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true
) where {T <: Float}
    # dimensions of problem
    n,p = size(x)
    print_with_color(:red, "gwas332, Starting one_fold().\n")

    # make vector of indices for folds
    test_idx = folds .== fold
    test_size = sum(test_idx)

    # train_idx is the vector that indexes the TRAINING set
    train_idx = .!test_idx
    mask_n    = convert(Vector{Int}, train_idx)
    mask_test = convert(Vector{Int}, test_idx)
    #Gordon
    println("test_size = $(test_size)")
    #println("train_idx = $(train_idx)")
    println()
    println()
    #println("test_idx = $(test_idx)")

    # compute the regularization path on the training set
    betas = iht_path(x, y, path, mask_n=mask_n, max_iter=max_iter, quiet=quiet, max_step=max_step, pids=pids, tol=tol)

    # tidy up
    #gc()

    # preallocate vector for output
    myerrors = zeros(T, length(path))

    # allocate an index vector for b
    indices = falses(p)

    # allocate the arrays for the test set
    xb = SharedArray{T}((n,), init = S -> S[localindexes(S)] = zero(T), pids=pids) :: SharedVector{T}
    b  = SharedArray{T}((p,), init = S -> S[localindexes(S)] = zero(T), pids=pids) :: SharedVector{T}
    r  = SharedArray{T}((n,), init = S -> S[localindexes(S)] = zero(T), pids=pids) :: SharedVector{T}

    # compute the mean out-of-sample error for the TEST set
    # do this for every computed model in regularization path
    for i = 1:size(betas,2)

        # pull ith model in dense vector format
        b2 = full(vec(betas[:,i]))

        # copy it into SharedArray b
        copy!(sdata(b),sdata(b2))

        # indices stores Boolean indexes of nonzeroes in b
        update_indices!(indices, b)

        # compute estimated response Xb with $(path[i]) nonzeroes
        #A_mul_B!(xb, x, b, indices, path[i], mask_test, pids=pids)

#        p_tmp = convert(Array{T,1}, path[i]) # Gordon - didn't help
        #A_mul_B!(xb, x, b, indices, path[i], mask_test)

        # compute residuals
        r .= y .- xb

        # mask data from training set
        # training set consists of data NOT in fold:
        # r[folds .!= fold] = zero(T)
        mask!(r, mask_test, 0, zero(T))

        # compute out-of-sample error as squared residual averaged over size of test set
        myerrors[i] = sum(abs2, r) / test_size / 2
        #myerrors[i] -= size(betas,2) # Gordon force more Betas lower becuase A_mul_B!() is needed above
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
    pids       :: Vector{Int},
    q          :: Int;
    max_iter   :: Int  = 100,
    max_step   :: Int  = 50,
    quiet      :: Bool = true,
    header     :: Bool = false
)
println("gwas409, Starting pfold(..xt..). I'm killing it here with x=xx.")
x=xx

    # ensure correct type
    @assert T <: Float "Argument T must be either Float32 or Float64"

    # do not allow crossvalidation with fewer than 3 folds
    @assert q > 2 "Number of folds q = $q must be at least 3."

    # how many CPU processes can pfold use?
    np = length(pids)

    # report on CPU processes
    quiet || println("pfold: np = ", np)
    quiet || println("pids = ", pids)

    # set up function to share state (indices of folds)
    i = 1
    nextidx() = (idx=i; i+=1; idx)

    # preallocate cell array for results
    results = SharedArray{T}((length(path),q), pids=pids) :: SharedMatrix{T}

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
                        r = remotecall_fetch(worker) do
                            processes = [worker]
                            x = BEDFile(T, xfile, x2file, meanfile, precfile, pids=processes, header=header)
                            y = SharedArray{T}(abspath(yfile), (x.geno.n,), pids=processes) :: SharedVector{T}

                            one_fold(x, y, path, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=processes)
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
pfold(xfile::String, xtfile::String, x2file::String, yfile::String, meanfile::String, precfile::String, path::DenseVector{Int}, folds::DenseVector{Int}, pids::Vector{Int}, q::Int; max_iter::Int=100, max_step::Int =50, quiet::Bool=true, header::Bool=false) = pfold(Float64, xfile, xtfile, x2file, yfile, meanfile, precfile, path, folds, pids, q, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)

function pfold(
    T        :: Type,
    xfile    :: String,
    x2file   :: String,
    yfile    :: String,
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    pids     :: Vector{Int},
    q        :: Int;
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true,
    header   :: Bool = false
)
print_with_color(:red, "gwas488, Starting pfold(..NOT xt..).\n")

    # ensure correct type
    @assert T <: Float "Argument T must be either Float32 or Float64"

    # do not allow crossvalidation with fewer than 3 folds
    @assert q > 2 "Number of folds q = $q must be at least 3."

    # how many CPU processes can pfold use?
    np = length(pids)

    # report on CPU processes
    quiet || println("pfold: np = ", np)
    quiet || println("pids = ", pids)

    # set up function to share state (indices of folds)
    i = 1
    nextidx() = (idx=i; i+=1; idx)

    # preallocate array for results
    results = SharedArray{T}((length(path),q), pids=pids) :: SharedMatrix{T}

    # master process will distribute tasks to workers
    # master synchronizes results at end before returning
#=
    println("here 500, yfile = $(yfile)")
    #Gordon - THIS CODE IS NOT NEEDED HERE
    # I WAS JUST USING IT TO CHECK RESULTS MATCH BETWEEN BEN AND KEVIN'S L0_reg() CALLS BEFORE LAUNCHING THE FOLDS
    # THIS CODE IS DUPLICATED BELOW, WHERE IT IS ACUTALLY NEEDED
#
    println("here 500, xfile = $(xfile)")

    keyword = set_keyword_defaults!(Dict{AbstractString, Any}())
    keyword["data_type"] = ""
    keyword["predictors_per_group"] = ""
    keyword["manhattan_plot_file"] = ""
    keyword["max_groups"] = ""
    keyword["group_membership"] = ""

    keyword["prior_weights"] = ""
    keyword["pw_algorithm_value"] = 1.0     # not user defined at this time

    file_name = xfile[1:end-4]
# try xt_test
    snpmatrix = SnpArray("xt_test")
    # HERE I READ THE PHENOTYPE FROM THE BEDFILE TO MATCH THE TUTORIAL RESULTS
    # I'M SURE THERE IS A BETTER WAY TO GET IT
    #phenotype is already set in tutorial_simulation.jl above  DIDN'T WORK - SAYS IT'S NOT DEFINED HERE
    #phenotype = readdlm(file_name * ".fam", header = false)[:, 6] # NO GOOD, THE PHENOTYPE HERE IS ALL ONES
    x = BEDFile(T, xfile, x2file, header=header, pids=[1]) :: BEDFile{T}
    y = SharedArray{T}(abspath(yfile), (x.geno.n,), pids=[1]) :: SharedVector{T}
    phenotype = convert(Array{T,1}, y)
    #println("phenotype = $(phenotype)")
    # y_copy = copy(phenotype)
    # y_copy .-= mean(y_copy)
    groups = fill(1,24000)
    k = 10
    J = 1
    outputg = L0_reg(snpmatrix, phenotype, J, k, groups, keyword)
    #println("outputg.beta = $(outputg.beta)")
    found = find(outputg.beta .!= 0.0)
    println("betas found in xt_test = $(found)")
# try x_test
    snpmatrix = SnpArray("x_test")
    outputg = L0_reg(snpmatrix, phenotype, J, k, groups, keyword)
    found = find(outputg.beta .!= 0.0)
    println("betas found in x_test = $(found)")

=#
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
                        println("here 524, abspath(yfile) = $(abspath(yfile))")
                        r = remotecall_fetch(worker) do
                            processes = [worker]
                            println("xfile = $(xfile)")
                            println("x2file = $(x2file)")

                            println("here 530, abspath(yfile) = $(abspath(yfile))")

                            keyword = set_keyword_defaults!(Dict{AbstractString, Any}())
                            keyword["data_type"] = ""
                            keyword["predictors_per_group"] = ""
                            keyword["manhattan_plot_file"] = ""
                            keyword["max_groups"] = ""
                            keyword["group_membership"] = ""

                            keyword["prior_weights"] = ""
                            keyword["pw_algorithm_value"] = 1.0     # not user defined at this time

                            file_name = xfile[1:end-4]
                        # try xt_test
                            snpmatrix = SnpArray("x_test")
                            # HERE I READ THE PHENOTYPE FROM THE BEDFILE TO MATCH THE TUTORIAL RESULTS
                            # I'M SURE THERE IS A BETTER WAY TO GET IT
                            #phenotype is already set in tutorial_simulation.jl above  DIDN'T WORK - SAYS IT'S NOT DEFINED HERE
                            #phenotype = readdlm(file_name * ".fam", header = false)[:, 6] # NO GOOD, THE PHENOTYPE HERE IS ALL ONES
                            x = BEDFile(T, xfile, x2file, header=header, pids=[1]) :: BEDFile{T}
                            y = SharedArray{T}(abspath(yfile), (x.geno.n,), pids=[1]) :: SharedVector{T}
                            phenotype = convert(Array{T,1}, y)
                            #println("phenotype = $(phenotype)")
                            # y_copy = copy(phenotype)
                            # y_copy .-= mean(y_copy)
                            groups = fill(1,24000)
                            k = 10
                            J = 1
                            outputg = L0_reg(snpmatrix, phenotype, J, k, groups, keyword)
                            #println("outputg.beta = $(outputg.beta)")
                            found = find(outputg.beta .!= 0.0)
                            println("betas found in x_test = $(found)")

                            x = BEDFile(T, xfile, x2file, pids=processes, header=header)
                            y = SharedArray{T}(abspath(yfile), (x.geno.n,), pids=processes) :: SharedVector{T}
                            println("here 533, abspath(yfile) = $(abspath(yfile))")
                            #y = SharedArray{T}(yfile, (x.geno.n,), pids=processes) :: SharedVector{T}

                            one_fold(snpmatrix, phenotype, path, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=processes)
                            #one_fold(x, y, path, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=processes)
                            # !!! don't put any code here, return from one_fold() is return for remotecall_fetch() !!!
                        end # end remotecall_fetch()
                        println("First test done.")
                        setindex!(results, r, :, current_fold)
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return (vec(sum(results, 2) ./ q)) :: Vector{T}
end

# default for previous function is Float64
pfold(xfile::String, x2file::String, yfile::String, path::DenseVector{Int}, folds::DenseVector{Int}, pids::Vector{Int}, q::Int; max_iter::Int = 100, max_step::Int = 50, quiet::Bool = true, header::Bool = false) = pfold(Float64, xfile, x2file, yfile, path, folds, pids, q, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)



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
    quiet    :: Bool  = true,
    header   :: Bool  = false
)

    # enforce type
    @assert T <: Float "Argument T must be either Float32 or Float64"
    println("gwas#666, Starting cv_iht(). I'm killing it here with x=xx.")
x=xx
    # how many elements are in the path?
    nmodels = length(path)

    # compute folds in parallel
    mses = pfold(T, xfile, xtfile, x2file, yfile, meanfile, precfile, path, folds, pids, q, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)

    # what is the best model size?
    k = path[indmin(errors)] :: Int

    # print results
    !quiet && print_cv_results(mses, path, k)

    # recompute ideal model
    # first load data on *all* processes
    x = BEDFile(T, xfile, xtfile, x2file, meanfile, precfile, header=header, pids=pids) :: BEDFile{T}
    y = SharedArray{T}(abspath(yfile), (x.geno.n,), pids=pids) :: SharedVector{T}

    # first use L0_reg to extract model
    output = L0_reg(x, y, k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, pids=pids)

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
    quiet    :: Bool  = true,
    header   :: Bool  = false
)

    # enforce type
    @assert T <: Float "Argument T must be either Float32 or Float64"
print_with_color(:red, "gwas#666, Starting cv_iht().\n")
    # how many elements are in the path?
    nmodels = length(path)
    println("here 664")
    # compute folds in parallel
    println("xfile = $(xfile)")
    println("x2file = $(x2file)")
    yfile = yfile[3:end]
    println("yfile = $(yfile)")
    #println("path = $(path)")
    mses = pfold(T, xfile, x2file, yfile, path, folds, pids, q, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)
    println("here 667")
    #Gordon
    println("here 500, yfile = $(yfile)")
    println("here 500, xfile = $(xfile)")

    keyword = set_keyword_defaults!(Dict{AbstractString, Any}())
    keyword["data_type"] = ""
    keyword["predictors_per_group"] = ""
    keyword["manhattan_plot_file"] = ""
    keyword["max_groups"] = ""
    keyword["group_membership"] = ""

    keyword["prior_weights"] = ""
    keyword["pw_algorithm_value"] = 1.0     # not user defined at this time

    file_name = xfile[1:end-4]
# try xt_test
    snpmatrix = SnpArray("x_test")
    # HERE I READ THE PHENOTYPE FROM THE BEDFILE TO MATCH THE TUTORIAL RESULTS
    # I'M SURE THERE IS A BETTER WAY TO GET IT
    #phenotype is already set in tutorial_simulation.jl above  DIDN'T WORK - SAYS IT'S NOT DEFINED HERE
    #phenotype = readdlm(file_name * ".fam", header = false)[:, 6] # NO GOOD, THE PHENOTYPE HERE IS ALL ONES
    x = BEDFile(T, xfile, x2file, header=header, pids=[1]) :: BEDFile{T}
    y = SharedArray{T}(abspath(yfile), (x.geno.n,), pids=[1]) :: SharedVector{T}
    phenotype = convert(Array{T,1}, y)
    #println("phenotype = $(phenotype)")
    # y_copy = copy(phenotype)
    # y_copy .-= mean(y_copy)
    groups = fill(1,24000)
    k = 10
    J = 1
    outputg = L0_reg(snpmatrix, phenotype, J, k, groups, keyword)
    #println("outputg.beta = $(outputg.beta)")
    found = find(outputg.beta .!= 0.0)
    println("betas found in x_test = $(found)")

    # what is the best model size?
    k = path[indmin(mses)] :: Int

    # print results
    !quiet && print_cv_results(mses, path, k)
    k = 10 # Gordon force 10 to be best model, because we still need new A_mul_B!() for error calcs

    # recompute ideal model
    ### first load data on *all* processes
    # first load data on *master* processes
#=
    x = BEDFile(T, xfile, x2file, header=header, pids=[1]) :: BEDFile{T}
    y = SharedArray{T}(abspath(yfile), (x.geno.n,), pids=[1]) :: SharedVector{T}
    #println("y = $(y)")
=#
    # first use L0_reg to extract model
    output = L0_reg(x, y, k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, pids=[1])

    # which components of beta are nonzero?
    inferred_model = outputg.beta .!= 0
    bidx = find(inferred_model)
    #Gordon
    println("betas found in bidx = $(bidx)")

    # allocate the submatrix of x corresponding to the inferred model
    x_inferred = zeros(T, x.geno.n, sum(inferred_model))
    decompress_genotypes!(x_inferred, x, inferred_model)

    # refit the best model
    b, bidx = refit_iht(x_inferred, y, k, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet)

    bids = prednames(x)[bidx]
    return IHTCrossvalidationResults(mses, sdata(path), b, bidx, k, bids)
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
cv_iht(xfile::String, x2file::String, yfile::String;
 q::Int = 3, #cv_get_num_folds(3,5),   # Gordon - restricted for testing
  path::DenseVector{Int} = begin bimfile=xfile[1:(endof(xfile)-3)] * "bim"; p=countlines(bimfile); [1, 10] end, # Gordon - only way it works on my Windows PC
   folds::DenseVector{Int} = begin famfile=xfile[1:(endof(xfile)-3)] * "fam"; n=countlines(famfile); cv_get_folds(n, q) end,
    pids::Vector{Int}=procs(),
     tol::Float64=1e-4,
      max_iter::Int=100,
       max_step::Int=50,
        quiet::Bool=true,
         header::Bool=false) = cv_iht(Float64, xfile, x2file, yfile, path=path, folds=folds, q=q, pids=pids, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)
"""
    difference!(Z, X, Y [, a=1.0, b=1.0])

Compute the matrix difference `Z = a*Y - b*Z`, overwriting `Z`.
"""
function difference!{T <: Float}(
    Z :: DenseMatrix{T},
    X :: DenseMatrix{T},
    Y :: DenseMatrix{T};
    a :: T = one(T),
    b :: T = one(T),
)
    m,n = size(Z)
    (m,n) == size(X) == size(Y) || throw(DimensionMismatch("Arguments, Z, X, and Y must have same size"))
    @inbounds for j = 1:n
        @inbounds for i = 1:m
            Z[i,j] = a*X[i,j] - b*Y[i,j]
        end
    end
    return nothing
end

"""
    difference!(x, y, z [, a=1.0, b=1.0, n=length(x)])

Compute the difference `x = a*y - b*z`, overwriting `x`.
"""
function difference!{T <: Float}(
    x :: DenseVector{T},
    y :: DenseVector{T},
    z :: DenseVector{T};
    a :: T = one(T),
    b :: T = one(T),
    n :: Int = length(x)
)
    @inbounds for i = 1:n
        x[i] = a*y[i] - b*z[i]
    end
    return nothing
end
