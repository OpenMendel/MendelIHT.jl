# ----------------------------------------- #
# functions to handle IHT output

# an object that houses results returned from an IHT run
immutable IHTResults{T <: Float, V <: DenseVector}
    time :: T
    loss :: T
    iter :: Int
    beta :: V

    IHTResults(time::T, loss::T, iter::Int, beta::DenseVector{T}) = new(time, loss, iter, beta)
end

# strongly typed external constructor for IHTResults
function IHTResults{T <: Float}(
    time :: T,
    loss :: T,
    iter :: Int,
    beta :: DenseVector{T}
)
    IHTResults{T, typeof(beta)}(time, loss, iter, beta)
end


# function to display IHTResults object
function Base.show(io::IO, x::IHTResults)
    println(io, "IHT results:")
    println(io, "\nCompute time (sec):   ", x.time)
    println(io, "Final loss:           ", x.loss)
    println(io, "Iterations:           ", x.iter)
    println(io, "IHT estimated ", countnz(x.beta), " nonzero coefficients.")
    print(io, DataFrame(Predictor=find(x.beta), Estimated_β=x.beta[find(x.beta)]))
    return nothing
end

# ----------------------------------------- #
# functions for handling temporary arrays

# an object to contain intermediate variables and temporary arrays
type IHTVariables{T <: Float, V <: DenseVector}
    b    :: V
    b0   :: Vector{T}
    xb   :: V
    xb0  :: Vector{T}
    xk   :: Matrix{T}
    gk   :: Vector{T}
    xgk  :: Vector{T}
    idx  :: BitArray{1}
    idx0 :: BitArray{1}
    r    :: V
    df   :: V

    IHTVariables(b::DenseVector{T}, b0::Vector{T}, xb::DenseVector{T}, xb0::Vector{T}, xk::Matrix{T},
                 gk::Vector{T}, xgk::Vector{T}, idx::BitArray{1}, idx0::BitArray{1}, r::DenseVector{T},
                 df::DenseVector{T}) = new(b, b0, xb, xb0, xk, gk, xgk, idx, idx0, r, df)
end

# strong type construction of IHTVariables
function IHTVariables{T <: Float}(
    b    :: DenseVector{T},
    b0   :: Vector{T},
    xb   :: DenseVector{T},
    xb0  :: Vector{T},
    xk   :: Matrix{T},
    gk   :: Vector{T},
    xgk  :: Vector{T},
    idx  :: BitArray{1},
    idx0 :: BitArray{1},
    r    :: DenseVector{T},
    df   :: DenseVector{T}
)
    IHTVariables{T, typeof(b)}(b, b0, xb, xb0, xk, gk, xgk, idx, idx0, r, df)
end


# minor type instability with SharedArray constructors
# a small optimization for the future, but will not dramatically change performance
# try defining b0, xb0 first and then calling "SharedVector(b0)", "SharedVector(xb0)"
function IHTVariables{T <: Float}(
    x :: SharedMatrix{T},
    y :: SharedVector{T},
    k :: Int
)
    pids = procs(x)
    V    = typeof(y)
    n, p = size(x)
#    b    = SharedArray(T, (p,), pids=pids) :: V
#    df   = SharedArray(T, (p,), pids=pids) :: V
#    xb   = SharedArray(T, (n,), pids=pids) :: V
#    r    = SharedArray(T, (n,), pids=pids) :: V
    b0   = zeros(T, p)
    b    = convert(V, SharedArray(T, size(b0), pids=pids))
    df   = convert(V, SharedArray(T, size(b0), pids=pids))
    xb   = convert(V, SharedArray(T, size(y), pids=pids))
    r    = convert(V, SharedArray(T, size(y), pids=pids))
    xb0  = zeros(T, n)
    xk   = zeros(T, n, k)
    xgk  = zeros(T, n)
    gk   = zeros(T, k)
    idx  = falses(p)
    idx0 = falses(p)
    return IHTVariables{T, V}(b, b0, xb, xb0, xk, gk, xgk, idx, idx0, r, df)
end

# strongly typed constructor for IHTVariables using regular floating point arrays
function IHTVariables{T <: Float}(
    x :: Matrix{T},
    y :: Vector{T},
    k :: Int
)
    n, p = size(x)
    b    = zeros(T, p)
    b0   = zeros(T, p)
    df   = zeros(T, p)
    xk   = zeros(T, n, k)
    xb   = zeros(T, n)
    xb0  = zeros(T, n)
    r    = zeros(T, n)
    xgk  = zeros(T, n)
    gk   = zeros(T, k)
    idx  = falses(p)
    idx0 = falses(p)
    return IHTVariables{T, typeof(y)}(b, b0, xb, xb0, xk, gk, xgk, idx, idx0, r, df)
end

# strongly typed constructor for IHTVariables using BEDFile + SharedVector
function IHTVariables{T <: Float}(
    x :: BEDFile{T},
    y :: SharedVector{T},
    k :: Int
)
    n, p = size(x)
    pids = procs(x)
    V    = typeof(y)
    b    = SharedArray(T, (p,), pids=pids) :: V
    df   = SharedArray(T, (p,), pids=pids) :: V
    xb   = SharedArray(T, (n,), pids=pids) :: V
    r    = SharedArray(T, (n,), pids=pids) :: V
    b0   = zeros(T, p)
    xb0  = zeros(T, n)
    xk   = zeros(T, n, k)
    xgk  = zeros(T, n)
    gk   = zeros(T, k)
    idx  = falses(p)
    idx0 = falses(p)
    return IHTVariables{T, V}(b, b0, xb, xb0, xk, gk, xgk, idx, idx0, r, df)
end

# function to modify fields of IHTVariables that depend on k
function update_variables!{T <: Float}(
    v :: IHTVariables{T},
    x :: DenseMatrix{T},
    k :: Int
)
    n    = size(x,1)
    v.xk = zeros(T, n, k)
    v.gk = zeros(T, k)
    return nothing
end

function update_variables!{T <: Float}(
    v :: IHTVariables{T},
    x :: BEDFile{T},
    k :: Int
)
    n    = size(x,1)
    v.xk = zeros(T, n, k)
    v.gk = zeros(T, k)
    return nothing
end


# ----------------------------------------- #
# crossvalidation routines

# subroutine to compute a default number of folds
@inline cv_get_num_folds(nmin::Int, nmax::Int) = max(nmin, min(Sys.CPU_CORES::Int, nmax))

# subroutine to refit preditors after crossvalidation
function refit_iht{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    k        :: Int;
    tol      :: T    = convert(T, 1e-6),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true,
)
    # initialize β vector and temporary arrays
    v = IHTVariables(x, y, k)

    # first use exchange algorithm to extract model
    output = L0_reg(x,y,k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol)

    # which components of β are nonzero?
    # cannot use binary indices here since we need to return Int indices
    bidx = find(v.b) :: Vector{Int}
    k2 = length(bidx)

    # allocate the submatrix of x corresponding to the inferred model
    # cannot use SubArray since result is not StridedArray?
    # issue is that bidx is Vector{Int} and not a Range object
    # use of SubArray is more memory efficient; a pity that it doesn't work!
    x_inferred = view(sdata(x), :, bidx)

    # now estimate β with the ordinary least squares estimator β = inv(x'x)x'y
    # return it with the vector of MSEs
    xty = At_mul_B(x_inferred, sdata(y)) :: Vector{T}
    xtx = At_mul_B(x_inferred, x_inferred) :: Matrix{T}
    b   = zeros(T, k2)
    try
        b = (xtx \ xty) :: Vector{T}
    catch e
        warn("caught error: ", e, "\nSetting returned values of b to -Inf")
        fill!(b, -Inf)
    end
  
    return b, bidx
end

# refitting routine for GWAS data with x', mean, prec files
function refit_iht(
    T        :: Type,
    xfile    :: String,
    xtfile   :: String,
    x2file   :: String,
    yfile    :: String,
    meanfile :: String,
    precfile :: String,
    k        :: Int;
    pids     :: DenseVector{Int} = procs(),
    tol      :: Float = convert(T, 1e-6),
    max_iter :: Int   = 100,
    max_step :: Int   = 50,
    quiet    :: Bool  = true,
    header   :: Bool  = false
)

    # initialize all variables
    x = BEDFile(T, xfile, xtfile, x2file, meanfile, precfile, pids=pids, header=header)
    y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids) :: SharedVector{T}

    # extract model with IHT
    output = L0_reg(x, y, k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol)
   
    # which components of β are nonzero?
    inferred_model = v.b .!= zero(T)
    bidx = find(inferred_model)
  
    # allocate the submatrix of x corresponding to the inferred model
    x_inferred = zeros(T, x.geno.n, sum(inferred_model))
    decompress_genotypes!(x_inferred, x, inferred_model)

    # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
    # return it with the vector of MSEs
    xty = At_mul_B(x_inferred, sdata(y))   :: Vector{T}
    xtx = At_mul_B(x_inferred, x_inferred) :: Matrix{T}
    b   = zeros(T, length(bidx))
    try
        b = (xtx \ xty) :: Vector{T}
    catch e
        warn("caught error: ", e, "\nSetting returned values of b to -Inf")
        fill!(b, -Inf)
    end

    # get predictor names before returning
    bids = prednames(x)[bidx]

    return b, bidx, bids
end


# refitting routine for GWAS data with just genotypes, covariates, y
function refit_iht(
    T        :: Type,
    xfile    :: String,
    x2file   :: String,
    yfile    :: String,
    k        :: Int;
    pids     :: DenseVector{Int} = procs(),
    tol      :: Float = convert(T, 1e-6),
    max_iter :: Int   = 100,
    quiet    :: Bool  = true,
    header   :: Bool  = false
)

    # initialize all variables
    x = BEDFile(T, xfile, x2file, pids=pids, header=header)
    y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids) :: SharedVector{T}

    # first use exchange algorithm to extract model
    L0_reg(x, y, k, max_iter=max_iter, quiet=quiet, tol=tol, window=k)

    # which components of β are nonzero?
    inferred_model = v.b .!= zero(T)
    bidx = find(inferred_model)
  
    # allocate the submatrix of x corresponding to the inferred model
    x_inferred = zeros(T, x.geno.n, sum(inferred_model))
    decompress_genotypes!(x_inferred, x, inferred_model)

    # now estimate β with the ordinary least squares estimator b = inv(x'x)x'y
    # return it with the vector of MSEs
    xty = At_mul_B(x_inferred, sdata(y))   :: Vector{T}
    xtx = At_mul_B(x_inferred, x_inferred) :: Matrix{T}
    b   = zeros(T, length(bidx))
    try
        b = (xtx \ xty) :: Vector{T}
    catch e
        warn("caught error: ", e, "\nSetting returned values of b to -Inf")
        fill!(b, -Inf)
    end

    # get predictor names before returning
    bids = prednames(x)[bidx]

    return b, bidx, bids
end




# return type for crossvalidation
immutable IHTCrossvalidationResults{T <: Float}
    mses :: Vector{T}
    path :: Vector{Int}
    b    :: Vector{T}
    bidx :: Vector{Int}
    k    :: Int
    bids :: Vector{String}

    IHTCrossvalidationResults(mses::Vector{T}, path::Vector{Int}, b::Vector{T}, bidx::Vector{Int}, k::Int, bids::Vector{String}) = new(mses, path, b, bidx, k, bids)
end

## strongly typed constructor for IHT CVR object
### 22 Sep 2016: no longer needed in Julia v0.5?
#function IHTCrossvalidationResults{T <: Float}(
#    mses :: Vector{T},
#    path :: Vector{Int},
#    b    :: Vector{T},
#    bidx :: Vector{Int},
#    k    :: Int,
#    bids :: Vector{String}
#)
#    IHTCrossvalidationResults{eltype(mses)}(mses, path, b, bidx, k, bids)
#end

## constructor for when b, bidx are not available
#function IHTCrossvalidationResults{T <: Float}(
#    mses :: Vector{T},
#    path :: Vector{Int},
#    k    :: Int
#) 
#    b    = zeros(T, 1)
#    bidx = zeros(Int, 1)
#    IHTCrossvalidationResults{T}(mses, path, b, bidx, k)
#end

# constructor for when bids are not available
# simply makes vector of "V$i" where $i are drawn from bidx
function IHTCrossvalidationResults{T <: Float}(
    mses :: Vector{T},
    path :: Vector{Int},
    b    :: Vector{T},
    bidx :: Vector{Int},
    k    :: Int
) 
    bids = convert(Vector{String}, ["V" * "$i" for i in bidx]) :: Vector{String}
    IHTCrossvalidationResults{eltype(mses)}(mses, path, b, bidx, k, bids)
end

# function to view an IHTCrossvalidationResults object
function Base.show(io::IO, x::IHTCrossvalidationResults)
    println(io, "Crossvalidation results:")
    println(io, "Minimum MSE ", minimum(x.mses), " occurs at k = $(x.k).")
    println(io, "Best model β has the following nonzero coefficients:")
    println(io, DataFrame(Predictor=x.bidx, Name=x.bids, Estimated_β=x.b))
    return nothing
end

function Gadfly.plot(x::IHTCrossvalidationResults)
    df = DataFrame(ModelSize=x.path, MSE=x.mses)
    plot(df, x="ModelSize", y="MSE", xintercept=[x.k], Geom.line, Geom.vline(color=colorant"red"), Guide.xlabel("Model size"), Guide.ylabel("MSE"), Guide.title("MSE versus model size"))
end

# ----------------------------------------- #
# subroutines for L0_reg

#function update_xb!{T <: Float}(
#   v :: IHTVariables{T},
#   x :: DenseMatrix{T},
#   k :: Int
#)
#    sum(v.idx) <= k || throw(ArgumentError("Argument indices with $(sum(indices)) trues should have at most $k of them"))
#    fill!(v.xb, zero(T))
#    numtrue = 0
#    @inbounds for j in eachindex(v.idx)
#        if v.idx[j]
#            numtrue += 1
#            @inbounds for i in eachindex(v.xb)
#                v.xb[i] += v.b[j]*x[i,j]
#            end
#        end
#        numtrue >= k && break
#    end
#    return nothing
#end

# subroutine to update residuals and gradient from data
function update_r_grad!{T}(
    v :: IHTVariables{T},
    x :: DenseMatrix{T},
    y :: DenseVector{T}
)
    #difference!(v.r, y, v.xb)
    broadcast!(-, v.r, y, v.xb) # v.r = y - v.xb
    At_mul_B!(v.df, x, v.r) # v.df = x' * v.r
    return nothing
end

function initialize_xb_r_grad!{T <: Float}(
    v :: IHTVariables{T},
    x :: DenseMatrix{T},
    y :: DenseVector{T},
    k :: Int
)
    # update x*beta
    if sum(v.idx) == 0
        fill!(v.xb, zero(T))
    else
        update_indices!(v.idx, v.b)
        update_xb!(v.xb, x, v.b, v.idx, k)
        #A_mul_B!(v.xb, view(x :, v.idx), view(v.b, v.idx) )
    end

    # update r and gradient
    update_r_grad!(v, x, y)
end

#function update_r_grad!{T}(
#    v    :: IHTVariables{T},
#    x    :: BEDFile{T},
#    y    :: DenseVector{T};
#    pids :: DenseVector{Int} = procs(x)
#)
#    #difference!(v.r, y, v.xb)
#    broadcast!(-, v.r, y, v.xb)
#    PLINK.At_mul_B!(v.df, x, v.r, pids=pids)
#    return nothing
#end
#
#function initialize_xb_r_grad!{T <: Float}(
#    v    :: IHTVariables{T},
#    x    :: BEDFile{T},
#    y    :: DenseVector{T},
#    k    :: Int;
#    pids :: DenseVector{Int} = procs(x)
#)
#    if sum(v.idx) == 0
#        fill!(v.xb, zero(T))
#        copy!(v.r, y)
#        At_mul_B!(v.df, x, v.r, pids=pids)
#    else
#        update_indices!(v.idx, v.b)
#        A_mul_B!(v.xb, x, v.b, v.idx, k, pids=pids)
#        update_r_grad!(v, x, y, pids=pids)
#    end
#    return nothing
#end


# ----------------------------------------- #
# common subroutines for IHT stepping

# this function updates the BitArray indices for b
function _iht_indices{T <: Float}(
    v :: IHTVariables{T},
    k :: Int;
)
    # which components of beta are nonzero?
    update_indices!(v.idx, v.b)

    # if current vector is 0,
    # then take largest elements of d as nonzero components for b
    if sum(v.idx) == 0
        a = select(v.df, k, by=abs, rev=true) :: T
#        threshold!(IDX, g, abs(a), n=p)
        v.idx[abs(v.df) .>= abs(a)-2*eps()] = true
        v.gk = zeros(T, sum(v.idx))
    end

    return nothing
end

# this function computes the step size for one update with iht()
function _iht_stepsize{T <: Float}(
    v :: IHTVariables{T},
    k :: Int
)
    # store relevant components of gradient
    v.gk = v.df[v.idx]

    # compute xk' * gk
    A_mul_B!(v.xgk, v.xk, v.gk)

    # warn if xgk only contains zeros
#    all(v.xgk .== zero(T)) && warn("Entire active set has values equal to 0")

    # return step size
    a = sumabs2(v.gk)  :: T
    b = sumabs2(v.xgk) :: T
    return a / b :: T
end

# this function computes one gradient step in iht()
function _iht_gradstep{T <: Float}(
    v  :: IHTVariables{T},
    mu :: T,
    k  :: Int;
)
    # take gradient step
    BLAS.axpy!(mu, sdata(v.df), sdata(v.b))

    # preserve top k components of b
    project_k!(v.b, k)

    # which indices of new beta are nonzero?
    update_indices!(v.idx, sdata(v.b))

    # must correct for equal entries at kth pivot of b
    # **note**: this uses Base.unsafe_setindex! to circumvent type stability issues
    # this is a VERY DANGEROUS DARK SIDE HACK!
    # hack randomly permutes indices of duplicates and retains one
    if sum(v.idx) > k
        a = select(v.b, k, by=abs, rev=true) :: T    # compute kth pivot
        dupes = (abs(v.b) .== abs(a)) :: BitArray{1} # find duplicates
        l = sum(dupes) :: Int                        # how many duplicates?
        l <= 1 && return nothing                     # if no duplicates, then simply return
        d = find(dupes) :: Vector{Int}               # find duplicates
        shuffle!(d)                                  # permute duplicates
        deleteat!(d, 1)                              # save first duplicate
        Base.unsafe_setindex!(v.b, zero(T), d)       # zero out other duplicates
        Base.unsafe_setindex!(v.idx, false, d)       # set corresponding indices to false
    end

    return nothing
end

# this function calculates the omega (here a / b) used for determining backtracking
function _iht_omega{T <: Float}(
    v :: IHTVariables{T}
)
    a = sqeuclidean(v.b, v.b0::Vector{T}) :: T
    b = sqeuclidean(v.xb, v.xb0::Vector{T}) :: T
    return a, b
end

# a function for determining whether or not to backtrack
function _iht_backtrack{T <: Float}(
    v       :: IHTVariables{T},
    ot      :: T,
    ob      :: T,
    mu      :: T,
    mu_step :: Int,
    nstep   :: Int
)
    mu*ob > 0.99*ot          &&
    sum(v.idx) != 0          &&
    sum(v.idx $ v.idx0) != 0 &&
    mu_step < nstep
end


# ----------------------------------------- #
# printing routines

# this prints the start of the algo
function print_header(io::IO)
     println(io, "\nBegin IHT algorithm\n")
     println(io, "Iter\tHalves\tMu\t\tNorm\t\tObjective")
     println(io, "0\t0\tInf\t\tInf\t\tInf")
end

# default IO for print_header is STDOUT
print_header() = print_header(STDOUT)

# alert when a descent error is found
function print_descent_error{T <: Float}(io::IO, iter::Int, loss::T, next_loss::T)
    print_with_color(:red, io, "\nIHT algorithm fails to descend!\n")
    print_with_color(:red, io, "Iteration: $(iter)\n")
    print_with_color(:red, io, "Current Objective: $(loss)\n")
    print_with_color(:red, io, "Next Objective: $(next_loss)\n")
    print_with_color(:red, io, "Difference in objectives: $(abs(next_loss - loss))\n")
end

# default IO for print_descent_error is STDOUT
print_descent_error{T <: Float}(iter::Int, loss::T, next_loss::T) = print_descent_error(iter, loss, next_loss)

# announce algo convergence
function print_convergence{T <: Float}(io::IO, iter::Int, loss::T, ctime::T)
    println(io, "\nIHT algorithm has converged successfully.")
    println(io, "Results:\nIterations: $(iter)")
    println(io, "Final Loss: $(loss)")
    println(io, "Total Compute Time: $(ctime)")
end

# default IO for print_convergence is STDOUT
print_convergence{T <: Float}(iter::Int, loss::T, ctime::T) = print_convergence(STDOUT, iter, loss, ctime)

# check the finiteness of an objective function value
# throw an error if value is not finite
function check_finiteness{T <: Float}(x::T)
    isnan(x) && throw(error("Objective function is NaN, aborting..."))
    isinf(x) && throw(error("Objective function is Inf, aborting..."))
end

# alert if iteration limit is reached
function print_maxiter{T <: Float}(io::IO, max_iter::Int, loss::T)
    print_with_color(:red, io, "IHT algorithm has hit maximum iterations $(max_iter)!\n")
    print_with_color(:red, io, "Current Loss: $(loss)\n")
end

# default IO for print_maxiter is STDOUT
print_maxiter{T <: Float}(max_iter::Int, loss::T) = print_maxiter(STDOUT, max_iter, loss)

# verbose printing of cv results
function print_cv_results{T <: Float}(io::IO, errors::Vector{T}, path::DenseVector{Int}, k::Int)
    println(io, "\n\nCrossvalidation Results:")
    println(io, "k\tMSE")
    for i = 1:length(errors)
        println(io, path[i], "\t", errors[i])
    end
    println(io, "\nThe lowest MSE is achieved at k = ", k)
end

# default IO for print_cv_results is STDOUT
print_cv_results{T <: Float}(errors::Vector{T}, path::DenseVector{Int}, k::Int) = print_cv_results(STDOUT, errors, path, k)

# -------------------------------------------- #
# functions related to logistic regression

# container object for temporary arrays
type IHTLogVariables{T <: Float, V <: DenseVector}
    xk       :: Matrix{T}
    xk2      :: Matrix{T}
    d2b      :: Matrix{T}
    b        :: V
    b0       :: Vector{T}
    df       :: V 
    xb       :: V 
    lxb      :: V 
    l2xb     :: Vector{T}
    bk       :: Vector{T}
    bk2      :: Vector{T}
    bk0      :: Vector{T}
    ntb      :: Vector{T}
    db       :: Vector{T}
    dfk      :: Vector{T}
    active   :: Vector{Int}
    bidxs    :: Vector{Int}
    dfidxs   :: Vector{Int}
    idxs     :: BitArray{1}
    idxs2    :: BitArray{1}
    idxs0    :: BitArray{1}

    IHTLogVariables(xk::Matrix{T}, xk2::Matrix{T}, d2b::Matrix{T}, b::DenseVector{T}, b0::Vector{T}, df::DenseVector{T}, xb::DenseVector{T}, lxb::DenseVector{T}, l2xb::Vector{T}, bk::Vector{T}, bk2::Vector{T}, bk0::Vector{T}, ntb::Vector{T}, db::Vector{T}, dfk::Vector{T}, active::Vector{Int}, bidxs::Vector{Int}, dfidxs::Vector{Int}, idxs::BitArray{1}, idxs2::BitArray{1}, idxs0::BitArray{1}) = new(xk, xk2, d2b, b, b0, df, xb, lxb, l2xb, bk, bk2, bk0, ntb, db, dfk, active, bidxs, dfidxs, idxs, idxs2, idxs0) 
end

#strongly typed constructor of IHTLogVariables
function IHTLogVariables{T <: Float}(
    xk       :: Matrix{T},
    xk2      :: Matrix{T},
    d2b      :: Matrix{T},
    b        :: DenseVector{T},
    b0       :: Vector{T},
    df       :: DenseVector{T},
    xb       :: DenseVector{T},
    lxb      :: DenseVector{T},
    l2xb     :: Vector{T},
    bk       :: Vector{T},
    bk2      :: Vector{T},
    bk0      :: Vector{T},
    ntb      :: Vector{T},
    db       :: Vector{T},
    dfk      :: Vector{T},
    active   :: Vector{Int},
    bidxs    :: Vector{Int},
    dfidxs   :: Vector{Int},
    idxs     :: BitArray{1},
    idxs2    :: BitArray{1},
    idxs0    :: BitArray{1}
)
    IHTLogVariables{T, typeof(b)}(xk, xk2, d2b, b, b0, df, xb, lxb, l2xb, bk, bk2, bk0, ntb, db, dfk, active, bidxs, dfidxs, idxs, idxs2, idxs0) 
end

# construct IHTLogVariables from data x, y, k
function IHTLogVariables{T <: Float}(
    x :: DenseMatrix{T},
    y :: DenseVector{T},
    k :: Int
)
    (n,p)  = size(x)
    xk     = zeros(T, n, k)
    xk2    = zeros(T, n, k)
    d2b    = zeros(T, k, k)
    b      = zeros(T, p)
    b0     = zeros(T, p)
    df     = zeros(T, p)
    xb     = zeros(T, n)
    lxb    = zeros(T, n)
    l2xb   = zeros(T, n)
    bk     = zeros(T, k)
    bk2    = zeros(T, k)
    bk0    = zeros(T, k)
    ntb    = zeros(T, k)
    db     = zeros(T, k)
    dfk    = zeros(T, k)
    active = collect(1:p)
    bidxs  = collect(1:p)
    dfidxs = collect(1:p)
    idxs   = falses(p)
    idxs0  = falses(p)
    idxs2  = falses(p)

    IHTLogVariables{T, typeof(b)}(xk, xk2, d2b, b, b0, df, xb, lxb, l2xb, bk, bk2, bk0, ntb, db, dfk, active, bidxs, dfidxs, idxs, idxs2, idxs0) 
end


# function to modify fields of IHTVariables that depend on k
function update_variables!{T <: Float}(
    v :: IHTLogVariables{T},
    x :: DenseMatrix{T},
    k :: Int
)
    n     = size(x,1)
    v.xk  = zeros(T, n, k)
    v.bk  = zeros(T, k)
    v.xk2 = zeros(T, n, k)
    v.d2b = zeros(T, k, k)
    v.bk0 = zeros(T, k)
    v.bk2 = zeros(T, k)
    v.ntb = zeros(T, k)
    v.db  = zeros(T, k)
    v.dfk = zeros(T, k)
    return nothing
end

###
### TODO: constructors for SharedArrays, BEDFiles!
###

# new return object for results file
immutable IHTLogResults{T <: Float, V <: DenseVector}
    time   :: T
    iter   :: Int
    loss   :: T
    beta   :: V 
    active :: Vector{Int}

    IHTLogResults(time::T, iter::Int, loss::T, β::DenseVector{T}, active::Vector{Int}) = new(time, iter, loss, β, active)
end

# strongly typed external constructor for IHTLogResults
function IHTLogResults{T <: Float}( 
    time   :: T,
    iter   :: Int,
    loss   :: T,
    β      :: DenseVector{T},
    active :: Vector{Int}
)
    IHTLogResults{T, typeof(β)}(time::T, iter::Int, loss::T, β::DenseVector{T}, active::Vector{Int})
end

# function to display IHTLogResults object
function Base.show(io::IO, x::IHTLogResults)
    println(io, "IHT results:")
    @printf(io, "\nCompute time (sec):   %3.4f\n", x.time)
    @printf(io, "Final loss:           %3.7f\n", x.loss)
    @printf(io, "Iterations:           %d\n", x.iter)
    println(io, "IHT estimated ", countnz(x.beta), " nonzero coefficients.")
    print(io, DataFrame(Predictor=find(x.beta), Estimated_β=x.beta[find(x.beta)]))
    return nothing
end


# announce algo convergence
function print_log_convergence{T <: Float}(io::IO, iter::Int, loss::T, ctime::T, nrmdf::T)
    println("\nL0_log has converged successfully.")
    @printf("Results:\nIterations: %d\n", iter)
    @printf("Final Loss: %3.7f\n", loss)
    @printf("Norm of active gradient: %3.7f\n", nrmdf)
    @printf("Total Compute Time: %3.3f sec\n", ctime)
end

# default IO for print_convergence is STDOUT
print_log_convergence{T <: Float}(iter::Int, loss::T, ctime::T, nrmdf::T) = print_log_convergence(STDOUT, iter, loss, ctime, nrmdf)

# this prints the start of the algo
function print_header_log(io::IO)
     println(io, "\nBegin IHT algorithm\n")
     println(io, "Iter\tSteps\tHalves\t\tf(β)\t\t||df(β)||")
     println(io, "0\t0\t0\t\tInf\t\tInf")
end

# default IO for print_header is STDOUT
print_header_log() = print_header_log(STDOUT)
