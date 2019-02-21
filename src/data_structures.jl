"""
Object to contain intermediate variables and temporary arrays. Used for cleaner code in L0_reg
"""
mutable struct IHTVariable{T <: Float}
    b     :: Vector{T}     # the statistical model for the genotype matrix, most will be 0
    b0    :: Vector{T}     # estimated model for genotype matrix in the previous iteration
    xb    :: Vector{T}     # vector that holds x*b
    xb0   :: Vector{T}     # xb in the previous iteration
    xk    :: Matrix{T}     # the n by k subset of the design matrix x corresponding to non-0 elements of b
    gk    :: Vector{T}     # numerator of step size. gk = df[idx]. 
    xgk   :: Vector{T}     # xk * gk, denominator of step size
    idx   :: BitVector     # idx[i] = 0 if b[i] = 0 and idx[i] = 1 if b[i] is not 0
    idx0  :: BitVector     # previous iterate of idx
    idc   :: BitVector     # idx[i] = 0 if c[i] = 0 and idx[i] = 1 if c[i] is not 0
    idc0  :: BitVector     # previous iterate of idc
    r     :: Vector{T}     # y - g^-1(Xβ), denoting the residuals. Arise as calculation in fisher's information matrix
    df    :: Vector{T}     # the gradient portion of the genotype part: df = ∇f(β) = snpmatrix * (y - xb - zc)
    df2   :: Vector{T}     # the gradient portion of the non-genetic covariates: covariate matrix * (y - xb - zc)
    c     :: Vector{T}     # estimated model for non-genetic variates (first entry = intercept)
    c0    :: Vector{T}     # estimated model for non-genetic variates in the previous iteration
    zc    :: Vector{T}     # z * c (covariate matrix times c)
    zc0   :: Vector{T}     # z * c (covariate matrix times c) in the previous iterate
    zdf2  :: Vector{T}     # z * df2. needed to calculate non-genetic covariate contribution for denomicator of step size 
    group :: Vector{Int64} # vector denoting group membership
    p     :: Vector{T}     # vector storing the mean of a glm: p = g^{-1}( Xβ )
end

function IHTVariables(
    x :: SnpArray,
    z :: AbstractMatrix{T},
    y :: AbstractVector{T},
    J :: Int64,
    k :: Int64;
) where {T <: Float}
    n, p  = size(x)
    q     = size(z, 2)

    b     = zeros(T, p)
    b0    = zeros(T, p)
    xb    = zeros(T, n)
    xb0   = zeros(T, n)
    xk    = zeros(T, n, J * k - 1) # subtracting 1 because the intercept will likely be selected in the first iter
    gk    = zeros(T, J * k - 1)    # subtracting 1 because the intercept will likely be selected in the first iter
    xgk   = zeros(T, n)
    idx   = falses(p)
    idx0  = falses(p)
    idc   = falses(q)
    idc0  = falses(q)
    r     = zeros(T, n)
    df    = zeros(T, p)
    df2   = zeros(T, q)
    c     = zeros(T, q)
    c0    = zeros(T, q)
    zc    = zeros(T, n)
    zc0   = zeros(T, n)
    zdf2  = zeros(T, n)
    group = ones(Int64, p + q) # both SNPs and non genetic covariates need group membership
    p     = zeros(T, n)

    return IHTVariable{T}(b, b0, xb, xb0, xk, gk, xgk, idx, idx0, idc, idc0, r, df, df2, c, c0, zc, zc0, zdf2, group, p)
end

"""
objects that house results returned from IHT run. 
The first `g` stands for group, the second `g` stands for generalized as in GLM.
"""
struct ggIHTResults{T <: Float, V <: DenseVector}
    time  :: T
    logl  :: T
    iter  :: Int
    beta  :: V
    c     :: V
    J     :: Int64
    k     :: Int64
    group :: Vector{Int64}

    ggIHTResults{T,V}(time, logl, iter, beta, c, J, k, group) where {T <: Float, V <: DenseVector{T}} = new{T,V}(time, logl, iter, beta, c, J, k, group)
end

# strongly typed external constructor for ggIHTResults
ggIHTResults(time::T, logl::T, iter::Int, beta::V, c::V, J::Int, k::Int, group::Vector{Int}) where {T <: Float, V <: DenseVector{T}} = ggIHTResults{T, V}(time, logl, iter, beta, c, J, k, group)

"""
functions to display ggIHTResults object
"""
function Base.show(io::IO, x::ggIHTResults)
    println(io, "IHT results:")
    println(io, "\nCompute time (sec):     ", x.time)
    println(io, "Final loglikelihood:    ", x.logl)
    println(io, "Iterations:             ", x.iter)
    println(io, "Max number of groups:   ", x.J)
    println(io, "Max predictors/group:   ", x.k)
    println(io, "IHT estimated ", count(!iszero, x.beta), " nonzero coefficients.")
    non_zero = findall(x -> x != 0, x.beta)
    print(io, DataFrame(Group=x.group[non_zero], Predictor=non_zero, Estimated_β=x.beta[non_zero]))
    println(io, "\n\nIntercept of model = ", x.c[1])

    return nothing
end

"""
verbose printing of cv results
"""
function print_cv_results(
    io::IO, 
    errors::Vector{T}, 
    path::DenseVector{Int}, 
    k::Int
) where {T <: Float}
    println(io, "\n\nCrossvalidation Results:")
    println(io, "k\tMSE")
    for i = 1:length(errors)
        println(io, path[i], "\t", errors[i])
    end
    println(io, "\nThe lowest MSE is achieved at k = $k \n")
end

# default IO for print_cv_results is STDOUT
print_cv_results(errors::Vector{T}, path::DenseVector{Int}, k::Int) where {T <: Float} = print_cv_results(stdout, errors, path, k)
