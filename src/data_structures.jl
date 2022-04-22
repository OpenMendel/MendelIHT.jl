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
    best_b :: Vector{T}     # best estimated genotype model in terms of loglikelihood
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
    best_c :: Vector{T}     # best estimated model for non-genetic variates in terms of loglikelihood
    zc     :: Vector{T}     # z * c (covariate matrix times c)
    zdf2   :: Vector{T}     # z * df2 needed to calculate non-genetic covariate contribution for denomicator of step size 
    group  :: Vector{Int}   # vector denoting group membership
    weight :: Vector{T}     # weights (typically minor allele freq) that will scale b prior to projection
    μ      :: Vector{T}     # mean of the current model: μ = l^{-1}(xb)
    cv_wts :: Vector{T}     # weights for cross validation. cv_wts[i] = 0 means sample i should not be included in fitting. 
    zkeep  :: BitVector     # tracks index of non-genetic covariates not subject to projection. zkeep[i] = true means `i` will not be projected. 
    zkeepn :: Int           # Total number of covariates that aren't subject to projection
    full_b :: Vector{T}     # storage for full beta and full gradient
    memory_efficient :: Bool # if true, xk and gk would be length 0 vector
end

nsamples(v::IHTVariable) = count(!iszero, v.cv_wts)
nsnps(v::IHTVariable) = size(v.x, 2)
ncovariates(v::IHTVariable) = size(v.z, 2) # number of nongenetic covariates
ntraits(v::IHTVariable) = 1

function IHTVariable(x::M, z::AbstractVecOrMat{T}, y::AbstractVector{T},
    J::Int, k::Union{Int, Vector{Int}}, d::UnivariateDistribution, l::Link,
    group::AbstractVector{Int}, weight::AbstractVector{T}, est_r::Symbol,
    zkeep::BitVector, memory_efficient::Bool) where {T <: Float, M <: AbstractMatrix}

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

    if lg != p && lg != 0
        throw(DimensionMismatch("group must have length $p but was $lg"))
    end 

    if lw != p && lw != 0
        throw(DimensionMismatch("weight must have length $p but was $lw"))
    end

    if typeof(k) == Int64
        columns = k
        ks = Int[]
    else
        columns = sum(k)
        ks, k = k, 0
    end

    if length(zkeep) != q
        throw(DimensionMismatch("zkeep must have length $q but was $(length(zkeep))"))
    end

    b      = Vector{T}(undef, p)
    b0     = Vector{T}(undef, p)
    best_b = Vector{T}(undef, p)
    xb     = Vector{T}(undef, n)
    xk     = memory_efficient ? Matrix{T}(undef, n, Threads.nthreads()) : Matrix{T}(undef, n, J * columns)
    gk     = memory_efficient ? Vector{T}(undef, 0) : Vector{T}(undef, J * columns)
    xgk    = Vector{T}(undef, n)
    idx    = BitArray(undef, p)
    idx0   = BitArray(undef, p)
    idc    = BitArray(undef, q)
    idc0   = BitArray(undef, q)
    r      = Vector{T}(undef, n)
    df     = Vector{T}(undef, p)
    df2    = Vector{T}(undef, q)
    c      = Vector{T}(undef, q)
    best_c = Vector{T}(undef, q)
    c0     = Vector{T}(undef, q)
    zc     = Vector{T}(undef, n)
    zdf2   = Vector{T}(undef, n)
    μ      = Vector{T}(undef, n)
    cv_wts = Vector{T}(undef, n)
    storage = Vector{T}(undef, p + q)

    return IHTVariable{T, M}(
        x, y, z, k, J, ks, d, l, est_r, 
        b, b0, best_b, xb, xk, gk, xgk, idx, idx0, idc, idc0, r, df, df2,
        c, c0, best_c, zc, zdf2, group, weight, μ, cv_wts, zkeep, sum(zkeep), 
        storage, memory_efficient)
end

function initialize(x::M, z::AbstractVecOrMat{T}, y::AbstractVecOrMat{T},
    J::Int, k::Union{Int, Vector{Int}}, d::Distribution, l::Link,
    group::AbstractVector{Int}, weight::AbstractVector{T}, est_r::Symbol,
    initialize_beta::Bool, zkeep::BitVector;
    cv_train_idx=trues(is_multivariate(y) ? size(x, 2) : size(x, 1)),
    memory_efficient::Bool=false, verbose::Bool=false
    ) where {T <: Float, M <: AbstractMatrix}

    if is_multivariate(y)
        v = mIHTVariable(x, z, y, k, zkeep, memory_efficient)
    else
        v = IHTVariable(x, z, y, J, k, d, l, group, weight, est_r, zkeep, memory_efficient)
    end

    # initialize non-zero indices
    MendelIHT.init_iht_indices!(v, initialize_beta, cv_train_idx, verbose)

    return v
end

"""
Multivaraite Gaussian IHT object, containing intermediate variables and temporary arrays
"""
mutable struct mIHTVariable{T <: Float, M <: AbstractMatrix}
    # data and pre-specified parameters for fitting
    X      :: M             # p × n design matrix (genotypes) of subtype AbstractMatrix{T}
    Y      :: Matrix{T}     # r × n response matrix (phenotypes)
    Z      :: VecOrMat{T}   # q × n other non-genetic covariates (intercept goes here)
    k      :: Int           # sparsity parameter
    # internal IHT variables
    B      :: Matrix{T}     # r × p matrix that holds the statistical model for the genotype matrix, most will be 0
    B0     :: Matrix{T}     # estimated model for genotype matrix in the previous iteration
    best_B :: Matrix{T}     # best statistical model for the genotype matrix in terms of loglikelihood
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
    best_C :: Matrix{T}     # best statistical model for the non-genetic variates in terms of loglikelihood
    CZ     :: Matrix{T}     # r × n matrix holding C * Z (C times nongenetic covariates)
    μ      :: Matrix{T}     # mean of the current model: μ = BX + CZ
    Γ      :: Matrix{T}     # estimated inverse covariance matrix (TODO: try StaticArrays.jl here)
    Γ0     :: Matrix{T}     # Γ in previous iterate (TODO: try StaticArrays here)
    cv_wts :: Vector{T}     # weights for cross validation. cv_wts[i] = 0 means sample i should not be included in fitting. 
    zkeep  :: BitVector     # tracks index of non-genetic covariates not subject to projection. zkeep[i] = true means `i` will not be projected. 
    zkeepn :: Int           # Total number of covariates that aren't subject to projection
    # storage variables
    full_b :: Vector{T}     # storage for vectorized form of full beta [vec(B); vec(Z)]
    r_by_r1 :: Matrix{T}    # an r × r storage (needed in loglikelihood)
    r_by_r2 :: Matrix{T}    # another r × r storage (needed in loglikelihood)
    r_by_n1 :: Matrix{T}    # an r × n storage (needed in score! function)
    r_by_n2 :: Matrix{T}    # an r × n storage (needed in stepsize calculation)
    n_by_r  :: Matrix{T}    # an n × r storage (needed to efficiently compute gradient)
    p_by_r  :: Matrix{T}    # an p × r storage (needed to efficiently compute gradient)
    k_by_r  :: Matrix{T}    # an k × r storage (needed for debiasing)
    k_by_k  :: Matrix{T}    # an k × k storage (needed for debiasing)
    memory_efficient :: Bool # if true, Xk would be 0 by 0 matrix
end

function mIHTVariable(x::M, z::AbstractVecOrMat{T}, y::AbstractMatrix{T},
    k::Int, zkeep::BitVector, memory_efficient::Bool
    ) where {T <: Float, M <: AbstractMatrix}

    n = size(x, 2) # number of samples 
    p = size(x, 1) # number of SNPs
    q = size(z, 1) # number of non-genetic covariates
    r = size(y, 1) # number of traits

    if !(n == size(y, 2) == size(z, 2))
        throw(DimensionMismatch("number of samples in y, x, and z = $(size(y, 2)), $n, $(size(z, 2)) are not equal"))
    end

    if length(zkeep) != q
        throw(DimensionMismatch("zkeep must have length $q but was $(length(zkeep))"))
    end

    B      = Matrix{T}(undef, r, p)
    B0     = Matrix{T}(undef, r, p)
    best_B = Matrix{T}(undef, r, p)
    BX     = Matrix{T}(undef, r, n)
    Xk     = Matrix{T}(undef, k, n)
    idx    = BitArray(undef, p)
    idx0   = BitArray(undef, p)
    idc    = BitArray(undef, q)
    idc0   = BitArray(undef, q)
    resid  = Matrix{T}(undef, r, n)
    df     = Matrix{T}(undef, r, p)
    df2    = Matrix{T}(undef, r, q)
    dfidx  = Matrix{T}(undef, r, k)
    C      = Matrix{T}(undef, r, q)
    C0     = Matrix{T}(undef, r, q)
    best_C = Matrix{T}(undef, r, q)
    CZ     = Matrix{T}(undef, r, n)
    μ      = Matrix{T}(undef, r, n)
    Γ      = Matrix{T}(I, r, r)
    Γ0     = Matrix{T}(I, r, r)
    cv_wts = Vector{T}(undef, n)
    full_b = Vector{T}(undef, r * (p + q))
    r_by_r1 = Matrix{T}(undef, r, r)
    r_by_r2 = Matrix{T}(undef, r, r)
    r_by_n1 = Matrix{T}(undef, r, n)
    r_by_n2 = Matrix{T}(undef, r, n)
    n_by_r = Matrix{T}(undef, n, r)
    p_by_r = Matrix{T}(undef, p, r)
    k_by_r = Matrix{T}(undef, k, r)
    k_by_k = Matrix{T}(undef, k, k)

    return mIHTVariable{T, M}(
        x, y, z, k,
        B, B0, best_B, BX, Xk, idx, idx0, idc, idc0, resid, df, df2, dfidx, C,
        C0, best_C, CZ, μ, Γ, Γ0, cv_wts, zkeep, r*sum(zkeep), full_b, r_by_r1,
        r_by_r2, r_by_n1, r_by_n2, n_by_r, p_by_r, k_by_r, k_by_k, memory_efficient)
end

nsamples(v::mIHTVariable) = count(!iszero, v.cv_wts)
nsnps(v::mIHTVariable) = size(v.X, 1)
ncovariates(v::mIHTVariable) = size(v.Z, 1) # number of nongenetic covariates
ntraits(v::mIHTVariable) = size(v.Y, 1)

"""
Immutable object that houses results returned from a single-trait IHT run. 
"""
struct IHTResult{T <: Float}
    time  :: Float64                    # total compute time
    logl  :: T                          # best loglikelihood achieved
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
    v.best_b, v.best_c, v.J, v.k, v.group, v.d, σg)

"""
Immutable object that houses results returned from a multivariate Gaussian IHT run. 
"""
struct mIHTResult{T <: Float}
    time   :: Float64                    # total compute time
    logl   :: T                          # best loglikelihood achieved
    iter   :: Int64                      # number of iterations until convergence
    beta   :: VecOrMat{T}                # estimated beta for genetic predictors
    c      :: VecOrMat{T}                # estimated beta for nongenetic predictors
    k      :: Int64                      # maximum number of predictors
    traits :: Int64                      # number of traits analyzed jointly
    Σ      :: Matrix{T}                  # estimated covariance matrix for multivariate analysis
    σg     :: Vector{T}                  # Estimated proportion of variance explained in phenotype
end
IHTResult(time, logl, iter, σg, v::mIHTVariable) = mIHTResult(time, logl, iter,
    v.best_B, v.best_C, v.k, ntraits(v), inv(v.Γ), σg)

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
# default IO for show is stdout
show(x::IHTResult) = show(stdout, x)

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
    println(io, "\nEstimated trait covariance:")
    println(io, DataFrame(x.Σ, ["trait$i" for i in 1:x.traits]))

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
    println(io, "\nBest k = $k\n")
end
# default IO for print_cv_results is stdout
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
# default IO for print_a_bunch_of_path_results is stdout
print_a_bunch_of_path_results(loglikelihoods::AbstractVector{T}, 
    path::AbstractVector{<:Integer}) where {T <: Float} =
    print_a_bunch_of_path_results(stdout, loglikelihoods, path)
