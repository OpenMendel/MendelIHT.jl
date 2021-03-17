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
    zdf2   :: Vector{T}     # z * df2 needed to calculate non-genetic covariate contribution for denomicator of step size 
    group  :: Vector{Int}   # vector denoting group membership
    weight :: Vector{T}     # weights (typically minor allele freq) that will scale b prior to projection
    μ      :: Vector{T}     # mean of the current model: μ = l^{-1}(xb)
    cv_wts :: Vector{T}     # weights for cross validation. cv_wts[i] = 0 means sample i should not be included in fitting. 
    full_b :: Vector{T}     # storage for full beta and full gradient
end

nsamples(v::IHTVariable) = length(v.y)
nsnps(v::IHTVariable) = size(v.x, 2)
ncovariates(v::IHTVariable) = size(v.z, 2) # number of nongenetic covariates
ntraits(v::IHTVariable) = 1

function IHTVariable(x::M, z::AbstractVecOrMat{T}, y::AbstractVector{T},
    J::Int, k::Union{Int, Vector{Int}}, d::UnivariateDistribution, l::Link,
    group::AbstractVector{Int}, weight::AbstractVector{T}, est_r::Symbol,
    cv_wts::AbstractVector{T},
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
    zdf2   = zeros(T, n)
    μ      = zeros(T, n)
    storage = zeros(T, p + q)

    return IHTVariable{T, M}(
        x, y, z, k, J, ks, d, l, est_r, 
        b, b0, xb, xk, gk, xgk, idx, idx0, idc, idc0, r, df, df2, c, c0, zc, zdf2, group, weight, μ, cv_wts, storage)
end

function initialize(x::M, z::AbstractVecOrMat{T}, y::AbstractVecOrMat{T},
    J::Int, k::Union{Int, Vector{Int}}, d::Distribution, l::Link,
    group::AbstractVector{Int}, weight::AbstractVector{T}, est_r::Symbol,
    cv_wts::AbstractVector{T},
    ) where {T <: Float, M <: AbstractMatrix}

    if is_multivariate(y)
        v = mIHTVariable(x, z, y, k, cv_wts)
    else
        v = IHTVariable(x, z, y, J, k, d, l, group, weight, est_r, cv_wts)
    end

    # initialize non-zero indices
    MendelIHT.init_iht_indices!(v)

    return v
end

"""
Multivaraite Gaussian IHT object, containing intermediate variables and temporary arrays
"""
mutable struct mIHTVariable{T <: Float, M <: AbstractMatrix}
    # data and pre-specified parameters for fitting
    X      :: M             # design matrix (genotypes) of subtype AbstractMatrix{T}
    Y      :: Matrix{T}     # response matrix (phenotypes)
    Z      :: VecOrMat{T}   # other non-genetic covariates 
    k      :: Int           # sparsity parameter
    # internal IHT variables
    B      :: Matrix{T}     # r × p matrix that holds the statistical model for the genotype matrix, most will be 0
    B0     :: Matrix{T}     # estimated model for genotype matrix in the previous iteration
    BX     :: Matrix{T}     # r × n matrix that holds B*X
    Xk     :: Matrix{T}     # the k × n subset of the design matrix x corresponding to non-0 elements of b
    idx    :: BitVector     # length p vector where idx[i] = 0 if i-th column of B is all zero, and idx[i] = 1 otherwise
    idx0   :: BitVector     # previous iterate of idx
    idc    :: BitVector     # length q vector where idc[i] = 0 i-th column of C is all zero, and idc[i] = 1 otherwise
    idc0   :: BitVector     # previous iterate of idc
    resid  :: Matrix{T}     # r × n matrix holding the residuals (Y - BX)
    df     :: Matrix{T}     # r × p genotype portion of the score = Γ(Y - XB)X'
    df2    :: Matrix{T}     # r × q non-genetic portion of the score
    dfidx  :: Matrix{T}     # r × k matrix storing df[:, idx], needed in stepsize calculation
    C      :: Matrix{T}     # r × q mtrix that holds the estimated model for non-genetic variates (first entry = intercept)
    C0     :: Matrix{T}     # estimated model for non-genetic variates in the previous iteration
    CZ     :: Matrix{T}     # r × n matrix holding C * Z (C times nongenetic covariates)
    μ      :: Matrix{T}     # mean of the current model: μ = BX + CZ
    Γ      :: Matrix{T}     # estimated inverse covariance matrix (TODO: try StaticArrays.jl here)
    Γ0     :: Matrix{T}     # Γ in previous iterate (TODO: try StaticArrays here)
    # storage variables
    full_b :: Vector{T}     # storage for vectorized form of full beta [vec(B); vec(Z)]
    r_by_r1 :: Matrix{T}    # an r × r storage (needed in loglikelihood)
    r_by_r2 :: Matrix{T}    # another r × r storage (needed in loglikelihood)
    r_by_n1 :: Matrix{T}    # an r × n storage (needed in score! function)
    r_by_n2 :: Matrix{T}    # an r × n storage (needed in stepsize calculation)
end

function mIHTVariable(x::M, z::AbstractVecOrMat{T}, y::AbstractMatrix{T},
    k::Int) where {T <: Float, M <: AbstractMatrix}

    n = size(x, 2) # number of samples 
    p = size(x, 1) # number of SNPs
    q = size(z, 1) # number of non-genetic covariates
    r = size(y, 1) # number of traits

    if !(n == size(y, 2) == size(z, 2))
        throw(DimensionMismatch("number of samples in y, x, and z = $(size(y, 2)), $n, $(size(z, 2)) are not equal"))
    end

    B      = zeros(T, r, p)
    B0     = zeros(T, r, p)
    BX     = zeros(T, r, n)
    Xk     = zeros(T, k - 1, n) # subtracting 1 because the intercept will likely be selected in the first iter
    idx    = falses(p)
    idx0   = falses(p)
    idc    = falses(q)
    idc0   = falses(q)
    resid  = zeros(T, r, n)
    df     = zeros(T, r, p)
    df2    = zeros(T, r, q)
    dfidx  = zeros(T, r, k - 1)
    C      = zeros(T, r, q)
    C0     = zeros(T, r, q)
    CZ     = zeros(T, r, n)
    μ      = zeros(T, r, n)
    Γ      = Matrix{T}(I, r, r)
    Γ0     = Matrix{T}(I, r, r)
    full_b = zeros(T, r * (p + q))
    r_by_r1 = zeros(T, r, r)
    r_by_r2 = zeros(T, r, r)
    r_by_n1 = zeros(T, r, n)
    r_by_n2 = zeros(T, r, n)

    return mIHTVariable{T, M}(
        x, y, z, k,
        B, B0, BX, Xk, idx, idx0, idc, idc0, resid, df, df2, dfidx, C, C0,
        CZ, μ, Γ, Γ0, full_b, r_by_r1, r_by_r2, r_by_n1, r_by_n2)
end

nsamples(v::mIHTVariable) = size(v.Y, 2)
nsnps(v::mIHTVariable) = size(v.X, 1)
ncovariates(v::mIHTVariable) = size(v.Z, 1) # number of nongenetic covariates
ntraits(v::mIHTVariable) = size(v.Y, 1)

"""
Immutable object that houses results returned from a single-trait IHT run. 
"""
struct IHTResult{T <: Float}
    time  :: Float64                    # total compute time
    logl  :: T                          # final loglikelihood
    iter  :: Int64                      # number of iterations until convergence
    beta  :: Vector{T}                  # estimated beta for genetic predictors
    c     :: Vector{T}                  # estimated beta for nongenetic predictors
    J     :: Int64                      # maximum number of groups (1 for multivariate analysis)
    k     :: Union{Int64, Vector{Int}}  # maximum number of predictors (vector if group IHT have differently sized groups)
    group :: Vector{Int64}              # group membership
    d     :: Distribution               # distribution of phenotype
    σg    :: T                          # Estimated proportion of variance explained in phenotype
end
IHTResult(time, logl, iter, σg, v::IHTVariable) = IHTResult(time, logl, iter,
    v.b, v.c, v.J, v.k, v.group, v.d, σg)

"""
Immutable object that houses results returned from a multivariate Gaussian IHT run. 
"""
struct mIHTResult{T <: Float}
    time   :: Float64                    # total compute time
    logl   :: T                          # final loglikelihood
    iter   :: Int64                      # number of iterations until convergence
    beta   :: VecOrMat{T}                # estimated beta for genetic predictors
    c      :: VecOrMat{T}                # estimated beta for nongenetic predictors
    k      :: Int64                      # maximum number of predictors
    traits :: Int64                      # number of traits analyzed jointly
    Σ      :: Matrix{T}                  # estimated covariance matrix for multivariate analysis
    σg     :: Vector{T}                  # Estimated proportion of variance explained in phenotype
end
IHTResult(time, logl, iter, σg, v::mIHTVariable) = mIHTResult(time, logl, iter,
    v.B, v.C, v.k, ntraits(v), inv(v.Γ), σg)

"""
Displays IHTResults object
"""
function Base.show(io::IO, x::IHTResult)
    snp_position = findall(x -> x != 0, x.beta)
    nongenetic_position = findall(x -> x != 0, x.c)

    println(io, "\nIHT estimated ", count(!iszero, x.beta), " nonzero SNP predictors and ", count(!iszero, x.c), " non-genetic predictors.")
    println(io, "\nCompute time (sec):     ", x.time)
    println(io, "Final loglikelihood:    ", x.logl)
    println(io, "SNP PVE:                ", x.σg)
    println(io, "Iterations:             ", x.iter)
    println(io, "\nSelected genetic predictors:")
    print(io, DataFrame(Position=snp_position, Estimated_β=x.beta[snp_position]))
    println(io, "\n\nSelected nongenetic predictors:")
    print(io, DataFrame(Position=nongenetic_position, Estimated_β=x.c[nongenetic_position]))
end

"""
Displays mIHTResult object
"""
function Base.show(io::IO, x::mIHTResult)
    println(io, "\nCompute time (sec):     ", x.time)
    println(io, "Final loglikelihood:    ", x.logl)
    println(io, "Iterations:             ", x.iter)
    for r in 1:x.traits
        println(io, "Trait $r's SNP PVE:      ", x.σg[r])
    end
    println("")
    for r in 1:x.traits
        β1 = @view(x.beta[r, :])
        C1 = @view(x.c[r, :])
        snp_position = findall(x -> x != 0, β1)
        nongenetic_position = findall(x -> x != 0, C1)    
        println(io, "\nTrait $r: IHT estimated ", count(!iszero, β1),
            " nonzero SNP predictors")
        println(io, DataFrame(Position=snp_position, Estimated_β=β1[snp_position]))
        println(io, "\nTrait $r: IHT estimated ", count(!iszero, C1),
            " non-genetic predictors")
        println(io, DataFrame(Position=nongenetic_position, Estimated_β=C1[nongenetic_position]))
    end
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
    println(io, "\nWe recommend running cross validation through `cv_iht` on " *
    "appropriate model sizes, which is roughly the values of k where the " *
    "loglikelihood stop increasing significantly.")
end
# default IO for print_a_bunch_of_path_results is STDOUT
print_a_bunch_of_path_results(loglikelihoods::AbstractVector{T}, 
    path::AbstractVector{<:Integer}) where {T <: Float} =
    print_a_bunch_of_path_results(stdout, loglikelihoods, path)
