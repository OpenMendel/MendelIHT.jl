# an object with returned results
immutable IHTResults{T <: Float, V <: DenseVector}
    time :: T
    loss :: T
    iter :: Int
    beta :: V

    IHTResults(time::T, loss::T, iter::Int, beta::DenseVector{T}) = new(time, loss, iter, beta)
end
function IHTResults{T <: Float}(
    time :: T,
    loss :: T,
    iter :: Int,
    beta :: DenseVector{T} 
)
    IHTResults{T, typeof(beta)}(time, loss, iter, beta)
end



# display function for IHTResults object
function Base.display(x::IHTResults)
    println("\ttime: ", x.time) 
    println("\tloss: ", x.loss) 
    println("\titer: ", x.iter) 
    println("\tb:    A ", typeof(x.beta), " with ", countnz(x.beta), " nonzeroes.")
    return nothing
end

# an object to contain intermediate variables
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

# strong type construction of previous object
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
# common subroutines for IHT stepping
function _iht_indices{T <: Float}(
    v :: IHTVariables{T},
    k :: Int;
)
    # which components of beta are nonzero?
    update_indices!(v.idx, v.b)

    # if current vector is 0,
    # then take largest elements of d as nonzero components for b
    if sum(v.idx) == 0
        a = select(v.df, k, by=abs, rev=true)
#        threshold!(IDX, g, abs(a), n=p)
        v.idx[abs(v.df) .>= abs(a)-2*eps()] = true
        v.gk = zeros(T, sum(v.idx))
    end

    return nothing
end

# TODO 29 June 2016: this is type-unstable! must fix
function _iht_stepsize{T <: Float}(
    v :: IHTVariables{T},
    k :: Int
)
    # store relevant components of gradient
#    fill_perm!(v.gk, v.df, v.idx)    # gk = g[IDX]
    v.gk = v.df[v.idx]

    # compute xk' * gk
#    BLAS.gemv!('N', one(T), v.xk, v.gk, zero(T), v.xgk)
    A_mul_B!(v.xgk, v.xk, v.gk)

    # warn if xgk only contains zeros
#    all(v.xgk .== zero(T)) && warn("Entire active set has values equal to 0")

    # return step size
    a = sumabs2(v.gk)  :: T
    b = sumabs2(v.xgk) :: T
    return a / b :: T
end


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
        a = select(v.b, k, by=abs, rev=true)    # compute kth pivot
        dupes = (abs(v.b) .== abs(a))           # find duplicates
        l = sum(dupes)                          # how many duplicates?
        l <= 1 && return nothing                # if no duplicates, then simply return
        d = find(dupes)                         # find duplicates
        shuffle!(d)                             # permute duplicates
        deleteat!(d, 1)                         # save first duplicate
        Base.unsafe_setindex!(v.b, zero(T), d)  # zero out other duplicates
        Base.unsafe_setindex!(v.idx, false, d)  # set corresponding indices to false
    end 

    return nothing
end


function _iht_omega{T <: Float}(
    v :: IHTVariables{T}
)
    a = sqeuclidean(v.b, v.b0::Vector{T}) :: T
    b = sqeuclidean(v.xb, v.xb0::Vector{T}) :: T
    return a, b 
end

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
function print_header()
     println("\nBegin IHT algorithm\n")
     println("Iter\tHalves\tMu\t\tNorm\t\tObjective")
     println("0\t0\tInf\t\tInf\t\tInf")
end

function print_descent_error{T <: Float}(iter::Int, loss::T, next_loss::T)
    print_with_color(:red, "\nIHT algorithm fails to descend!\n")
    print_with_color(:red, "Iteration: $(iter)\n")
    print_with_color(:red, "Current Objective: $(loss)\n")
    print_with_color(:red, "Next Objective: $(next_loss)\n")
    print_with_color(:red, "Difference in objectives: $(abs(next_loss - loss))\n")
end

function print_convergence{T <: Float}(iter::Int, loss::T, ctime::T)
    println("\nIHT algorithm has converged successfully.")
    println("Results:\nIterations: $(iter)")
    println("Final Loss: $(loss)")
    println("Total Compute Time: $(ctime)")
end 

function check_finiteness{T <: Float}(x::T)
    isnan(x) && throw(error("Objective function is NaN, aborting..."))
    isinf(x) && throw(error("Objective function is Inf, aborting..."))
end

function print_maxiter{T <: Float}(max_iter::Int, loss::T)
    print_with_color(:red, "IHT algorithm has hit maximum iterations $(max_iter)!\n")
    print_with_color(:red, "Current Loss: $(loss)\n")
end 
