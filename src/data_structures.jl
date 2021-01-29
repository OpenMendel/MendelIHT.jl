"""
Single (GLM) trait IHT object, containing intermediate variables and temporary arrays
"""
mutable struct IHTVariable{T <: Float, M <: AbstractMatrix}
    # data and pre-specified parameters for fitting
    x      :: M                      # design matrix (genotypes) of subtype AbstractMatrix{T}
    y      :: Vector{T}              # response vector (phenotypes)
    z      :: VecOrMat{T}            # other non-genetic covariates 
    k      :: Int                    # sparsity parameter (this is set to 0 if doubly sparse group IHT requires different k for each group)
    J      :: Int                    # Number of non-zero groups
    ks     :: Vector{Int}            # Sparsity level for each group. This is a empty vector if each group has same sparsity.
    d      :: UnivariateDistribution # Distribution for phenotypes
    l      :: Link                   # link function linking the mean μ with linear predictors xb via l(μ) = xb
    est_r  :: Symbol                 # Can be `:MM`, `:Newton`, or `:None`, specifying method used to estimate nuisance parameter in neg bin regression
    # internal IHT variables
    b      :: Vector{T}     # the statistical model for the genotype matrix, most will be 0
    b0     :: Vector{T}     # estimated model for genotype matrix in the previous iteration
    xb     :: Vector{T}     # vector that holds x*b
    xb0    :: Vector{T}     # xb in the previous iteration
    xk     :: Matrix{T}     # the n by k subset of the design matrix x corresponding to non-0 elements of b
    gk     :: Vector{T}     # numerator of step size. gk = df[idx]. 
    xgk    :: Vector{T}     # xk * gk, denominator of step size
    idx    :: BitVector     # idx[i] = 0 if b[i] = 0 and idx[i] = 1 if b[i] is not 0
    idx0   :: BitVector     # previous iterate of idx
    idc    :: BitVector     # idx[i] = 0 if c[i] = 0 and idx[i] = 1 if c[i] is not 0
    idc0   :: BitVector     # previous iterate of idc
    r      :: Vector{T}     # The difference between the observed and predicted response. For linear model this is the residual
    df     :: Vector{T}     # genotype portion of the score
    df2    :: Vector{T}     # non-genetic covariates portion of the score
    c      :: Vector{T}     # estimated model for non-genetic variates (first entry = intercept)
    c0     :: Vector{T}     # estimated model for non-genetic variates in the previous iteration
    zc     :: Vector{T}     # z * c (covariate matrix times c)
    zc0    :: Vector{T}     # z * c (covariate matrix times c) in the previous iterate
    zdf2   :: Vector{T}     # z * df2 needed to calculate non-genetic covariate contribution for denomicator of step size 
    group  :: Vector{Int}   # vector denoting group membership
    weight :: Vector{T}     # weights (typically minor allele freq) that will scale b prior to projection
    μ      :: Vector{T}     # mean of the current model: μ = l^{-1}(xb)
    grad   :: Vector{T}     # storage for full gradient
end

function IHTVariable(x::M, z::AbstractVecOrMat{T}, y::AbstractVector{T},
    J::Int, k::Union{Int, Vector{Int}}, d::UnivariateDistribution, l::Link,
    group::AbstractVector{Int}, weight::AbstractVector{T}, est_r::Symbol
    ) where {T <: Float, M <: AbstractMatrix}

    n = size(x, 1)
    p = size(x, 2)
    m = size(z, 1)
    q = size(z, 2)
    ly = length(y)
    lg = length(group)
    lw = length(weight)

    if !(ly == n == m)
        throw(DimensionMismatch("row dimension of y, x, and z ($ly, $n, $m) are not equal"))
    end

    if lg != (p + q) && lg != 0
        throw(DimensionMismatch("group must have length " * string(p + q) * " but was $lg"))
    end 

    if lw != (p + q) && lw != 0
        throw(DimensionMismatch("weight must have length " * string(p + q) * " but was $lw"))
    end

    if typeof(k) == Int64
        columns = k
        ks = Int[]
    else
        columns = sum(k)
        ks, k = k, 0
    end

    b      = zeros(T, p)
    b0     = zeros(T, p)
    xb     = zeros(T, n)
    xb0    = zeros(T, n)
    xk     = zeros(T, n, J * columns - 1) # subtracting 1 because the intercept will likely be selected in the first iter
    gk     = zeros(T, J * columns - 1)    # subtracting 1 because the intercept will likely be selected in the first iter
    xgk    = zeros(T, n)
    idx    = falses(p)
    idx0   = falses(p)
    idc    = falses(q)
    idc0   = falses(q)
    r      = zeros(T, n)
    df     = zeros(T, p)
    df2    = zeros(T, q)
    c      = zeros(T, q)
    c0     = zeros(T, q)
    zc     = zeros(T, n)
    zc0    = zeros(T, n)
    zdf2   = zeros(T, n)
    μ      = zeros(T, n)
    storage = zeros(T, p + q)

    return IHTVariable{T, M}(
        x, y, z, k, J, ks, d, l, est_r, 
        b, b0, xb, xb0, xk, gk, xgk, idx, idx0, idc, idc0, r, df, df2, c, c0, zc, zc0, zdf2, group, weight, μ, storage)
end

function initialize(x::M, z::AbstractVecOrMat{T}, y::AbstractVecOrMat{T},
    J::Int, k::Union{Int, Vector{Int}}, d::Distribution, l::Link,
    group::AbstractVector{Int}, weight::AbstractVector{T}, est_r::Symbol
    ) where {T <: Float, M <: AbstractMatrix}

    if size(y, 2) > 1
        v = mIHTVariable(x, z, y, k)
    else
        v = IHTVariable(x, z, y, J, k, d, l, group, weight, est_r)
    end

    # initialize non-zero indices
    MendelIHT.init_iht_indices!(v)

    # store relevant components of x for first iteration
    copyto!(v.xk, @view(x[:, v.idx])) 

    return v
end

"""
Multivaraite Gaussian IHT object, containing intermediate variables and temporary arrays
"""
mutable struct mIHTVariable{T <: Float, M <: AbstractMatrix}
    # data and pre-specified parameters for fitting
    x      :: M                      # design matrix (genotypes) of subtype AbstractMatrix{T}
    y      :: Matrix{T}              # response vector (phenotypes)
    z      :: VecOrMat{T}            # other non-genetic covariates 
    k      :: Int                    # sparsity parameter
    # internal IHT variables
    b      :: Matrix{T}     # the statistical model for the genotype matrix, most will be 0
    b0     :: Matrix{T}     # estimated model for genotype matrix in the previous iteration
    xb     :: Matrix{T}     # Matrix that holds x*b
    xb0    :: Matrix{T}     # xb in the previous iteration
    xk     :: Matrix{T}     # the n by k subset of the design matrix x corresponding to non-0 elements of b
    # gk     :: Matrix{T}     # numerator of step size. gk = df[idx]. 
    # xgk    :: Matrix{T}     # xk * gk, denominator of step size
    idx    :: BitVector     # idx[i] = 0 if b[i, j] = 0 for all j and idx[i] = 1 if b[i, j] != 0 for at least one j
    idx0   :: BitVector     # previous iterate of idx
    idc    :: BitVector     # idx[i] = 0 if c[i, j] = 0 for all j and idx[i] = 1 if c[i, j] != 0 for at least one j
    idc0   :: BitVector     # previous iterate of idc
    nzct   :: Vector{Int}   # Vector tracking number of non-zero beta positions for each column of beta
    resid  :: Matrix{T}     # The difference between the observed and predicted response
    df     :: Matrix{T}     # genotype portion of the score = X'(Y - XB)Γ
    df2    :: Matrix{T}     # non-genetic covariates portion of the score
    c      :: Matrix{T}     # estimated model for non-genetic variates (first entry = intercept)
    c0     :: Matrix{T}     # estimated model for non-genetic variates in the previous iteration
    zc     :: Matrix{T}     # z * c (covariate matrix times c)
    zc0    :: Matrix{T}     # z * c (covariate matrix times c) in the previous iterate
    zdf2   :: Matrix{T}     # z * df2 needed to calculate non-genetic covariate contribution for denomicator of step size 
    μ      :: Matrix{T}     # mean of the current model: μ = xb + zc
    grad   :: Matrix{T}     # storage for full gradient
    Σ      :: Matrix{T}     # estimated covariance matrix (TODO: try StaticArrays.jl here)
    Σ0     :: Matrix{T}     # estimated covariance matrix in previous iterate (TODO: try StaticArrays here)
    dΣ     :: Matrix{T}     # gradient of covariance matrix
end

# X  = n × p
# Z  = n × q
# Y  = n × r
# B  = p × r
# Γ  = r × r
# XB = n × r
# df = p × r (gradient)
function mIHTVariable(x::M, z::AbstractVecOrMat{T}, y::AbstractMatrix{T},
    k::Int) where {T <: Float, M <: AbstractMatrix}

    n = size(x, 1) # number of samples 
    p = size(x, 2) # number of SNPs
    q = size(z, 2) # number of non-genetic covariates
    r = size(y, 2) # number of traits

    if !(n == size(y, 1) == size(z, 1))
        throw(DimensionMismatch("number of samples in y, x, and z = $(size(y, 1)), $n, $(size(z, 1)) are not equal"))
    end

    b      = zeros(T, p, r)
    b0     = zeros(T, p, r)
    xb     = zeros(T, n, r)
    xb0    = zeros(T, n, r)
    xk     = zeros(T, n, k - 1) # subtracting 1 because the intercept will likely be selected in the first iter
    # gk     = zeros(T, k - 1, r) # subtracting 1 because the intercept will likely be selected in the first iter
    # xgk    = zeros(T, n, r)
    idx    = falses(p)
    idx0   = falses(p)
    idc    = falses(q)
    idc0   = falses(q)
    nzct   = zeros(Int, r)
    resid  = zeros(T, n, r)
    df     = zeros(T, p, r)
    df2    = zeros(T, q, r)
    c      = zeros(T, q, r)
    c0     = zeros(T, q, r)
    zc     = zeros(T, n, r)
    zc0    = zeros(T, n, r)
    zdf2   = zeros(T, n, r)
    μ      = zeros(T, n, r)
    grad   = zeros(T, p + q, r)
    Σ      = Matrix{T}(I, r, r)
    Σ0     = Matrix{T}(I, r, r)
    dΣ     = zeros(T, r, r)

    return mIHTVariable{T, M}(
        x, y, z, k, 
        b, b0, xb, xb0, xk, idx, idx0, idc, idc0, nzct, resid, df, df2, c, c0,
        zc, zc0, zdf2, μ, grad, Σ, Σ0, dΣ)
end

"""
immutable objects that house results returned from IHT run. 
"""
struct IHTResult{T <: Float}
    time  :: Union{Float64, Float32}    # total compute time
    logl  :: T                          # final loglikelihood
    iter  :: Int64                      # number of iterations until convergence
    beta  :: VecOrMat{T}                # estimated beta for genetic predictors
    c     :: VecOrMat{T}                # estimated beta for nongenetic predictors
    J     :: Int64                      # maximum number of groups (1 for multivariate analysis)
    k     :: Union{Int64, Vector{Int}}  # maximum number of predictors (vector if group IHT have differently sized groups)
    group :: Vector{Int64}              # group membership
    d     :: Distribution               # distribution of phenotype
    Σ     :: Union{Nothing, Matrix{T}}  # estimated covariance matrix for multivariate analysis
end
IHTResult(time, logl, iter, v::IHTVariable) = IHTResult(time, logl, iter,
    v.b, v.c, v.J, v.k, v.group, v.d, nothing)
IHTResult(time, logl, iter, v::mIHTVariable) = IHTResult(time, logl, iter,
    v.b, v.c, 1, v.k, Int[], MvNormal, v.Σ)

"""
functions to display IHTResults object
"""
function Base.show(io::IO, x::IHTResult)
    snp_position = findall(x -> x != 0, x.beta)
    nongenetic_position = findall(x -> x != 0, x.c)

    println(io, "\nIHT estimated ", count(!iszero, x.beta), " nonzero SNP predictors and ", count(!iszero, x.c), " non-genetic predictors.")
    println(io, "\nCompute time (sec):     ", x.time)
    println(io, "Final loglikelihood:    ", x.logl)
    println(io, "Iterations:             ", x.iter)
    println(io, "\nSelected genetic predictors:")
    print(io, DataFrame(Position=snp_position, Estimated_β=x.beta[snp_position]))
    println(io, "\n\nSelected nongenetic predictors:")
    print(io, DataFrame(Position=nongenetic_position, Estimated_β=x.c[nongenetic_position]))
end

"""
verbose printing of cv results
"""
function print_cv_results(io::IO, errors::Vector{T}, 
    path::AbstractVector{<:Integer}, k::Int) where {T <: Float}
    println(io, "\n\nCrossvalidation Results:")
    println(io, "\tk\tMSE")
    for i = 1:length(errors)
        println(io, "\t", path[i], "\t", errors[i])
    end
end
# default IO for print_cv_results is STDOUT
print_cv_results(errors::Vector{T}, path::AbstractVector{<:Integer}, k::Int
    ) where {T <: Float} = print_cv_results(stdout, errors, path, k)

"""
verbose printing of running `iht_run_many_models` with a bunch of models
"""
function print_a_bunch_of_path_results(io::IO, loglikelihoods::AbstractVector{T},
    path::AbstractVector{<:Integer}) where {T <: Float}
    println(io, "\n\nResults of running all the model sizes specified in `path`:")
    println(io, "\tk\tloglikelihoods")
    for i = 1:length(loglikelihoods)
        println(io, "\t", path[i], "\t", loglikelihoods[i])
    end
    println(io, "\nWe recommend running cross validation through `cv_iht_distributed` on " *
    "appropriate model sizes. Roughly speaking, this is when error stopped decreasing significantly.")
end
# default IO for print_a_bunch_of_path_results is STDOUT
print_a_bunch_of_path_results(loglikelihoods::AbstractVector{T}, 
    path::AbstractVector{<:Integer}) where {T <: Float} =
    print_a_bunch_of_path_results(stdout, loglikelihoods, path)
