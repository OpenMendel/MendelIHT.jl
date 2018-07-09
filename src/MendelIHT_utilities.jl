export IHTVariable, IHTVariables, use_A2_as_minor_allele, standardize_snpmatrix
export use_A2_as_minor_allele, project_k!

"""
Object to contain intermediate variables and temporary arrays. Used for cleaner code in L0_reg
"""
struct IHTVariable
    b    :: Vector{Float64}     # the statistical model, most will be 0
    b0   :: Vector{Float64}     # previous estimated model in the mm step
    xb   :: Vector{Float64}     # vector that holds x*b 
    xb0  :: Vector{Float64}     # previous xb in the mm step
    xk   :: Matrix{Float64}     # the n by k subset of the design matrix x corresponding to non-0 elements of b
    gk   :: Vector{Float64}     # Numerator of step size μ.   gk = df[idx] is a temporary array of length `k` that arises as part of the gradient calculations. I avoid doing full gradient calculations since most of `b` is zero. 
    xgk  :: Vector{Float64}     # Demonimator of step size μ. x * gk also part of the gradient calculation 
    idx  :: BitArray{1}         # BitArray indices of nonzeroes in b for A_mul_B
    idx0 :: BitArray{1}         # previous iterate of idx
    r    :: Vector{Float64}     # n-vector of residuals
    df   :: Vector{Float64}     # the gradient: df = -x' * (y - xb)
end

function IHTVariables(
    x :: SnpData,
    y :: Vector{Float64},
    k :: Int64
)
    n, p = x.people, x.snps + 1 #adding 1 for p because we need an intercept

    #check if k is sensible
    if k > p;  throw(ArgumentError("k cannot exceed the number of SNPs")); end
    if k <= 0; throw(ArgumentError("k must be positive integer")); end

    b    = zeros(Float64, p)
    df   = zeros(Float64, p)
    xb   = zeros(Float64, n)
    r    = zeros(Float64, n)
    b0   = zeros(Float64, p)
    xb0  = zeros(Float64, n)
    xk   = zeros(Float64, n, k)
    xgk  = zeros(Float64, n)
    gk   = zeros(Float64, k)
    idx  = falses(p) 
    idx0 = falses(p)
    return IHTVariable(b, b0, xb, xb0, xk, gk, xgk, idx, idx0, r, df)
end

immutable IHTResult
    time :: Float64
    loss :: Float64
    iter :: Int64
    beta :: Vector{Float64}
end

function IHTResults(
    time :: Float64, 
    loss :: Float64, 
    iter :: Int64,
    beta :: Vector{Float64})
    return IHTResult(time, loss, iter, beta)
end

# """
#     project_k!(x, k)

# This function projects a vector `x` onto the set S_k = { y in R^p : || y ||_0 <= k }.
# It does so by first finding the pivot `a` of the `k` largest components of `x` in magnitude.
# `project_k!` then thresholds `x` by `abs(a)`, sending small components to 0. 

# Arguments:

# - `b` is the vector to project.
# - `k` is the number of components of `b` to preserve.
# """
# function project_k!(
#     x :: Vector{Float64},
#     k :: Int;
# )
#     a = select(x, k, by = abs, rev = true)
#     for i in eachindex(x) 
#         if abs(x[i]) < abs(a) 
#             x[i] = 0.0
#         end
#     end
#     return nothing
# end

"""
Returns ω, a constant we need to bound the step size μ to guarantee convergence. 
"""

function compute_ω!(
    v         :: IHTVariable,
    snpmatrix :: Matrix{Float64}, 
)
    #update v.xb
    A_mul_B!(v.xb, snpmatrix, v.b)

    #calculate ω efficiently (old b0 and xb0 have been copied before calling iht!)
    return sqeuclidean(v.b, v.b0) / sqeuclidean(v.xb, v.xb0)
end

"""
This function is needed for testing purposes only. 

Converts a SnpArray to a matrix of float64 using A2 as the minor allele. We want this function 
because SnpArrays.jl uses the less frequent allele in each SNP as the minor allele, while PLINK.jl 
always uses A2 as the minor allele, and it's nice if we could cross-compare the results. 
"""
function use_A2_as_minor_allele(snpmatrix :: SnpArray)
    n, p = size(snpmatrix)
    matrix = zeros(n, p)
    for i in 1:p
        for j in 1:n
            if snpmatrix[j, i] == (0, 0); matrix[j, i] = 0.0; end
            if snpmatrix[j, i] == (0, 1); matrix[j, i] = 1.0; end
            if snpmatrix[j, i] == (1, 1); matrix[j, i] = 2.0; end
            if snpmatrix[j, i] == (1, 0); matrix[j, i] = missing; end
        end
    end
    return matrix
end

"""
A function for determining whether or not to backtrack. If all conditions are satisfied,
then we DONT do line search, which means _iht_backtrack need to return TRUE.
"""
function _iht_backtrack(
    v :: IHTVariable,
    ω :: Float64,
    μ :: Float64
)
    μ < 0.99*ω 
    #&& sum(v.idx) != 0 &&
    #sum(xor.(v.idx,v.idx0)) != 0 
end

"""
This function computes the gradient step v.b = P_k(β + μ∇f(β)), and updates v.idx. 
Recall calling axpy! implies v.b = v.b + μ*v.df, but v.df stores an extra negative sign.
"""
function _iht_gradstep(
    v  :: IHTVariable,
    μ  :: Float64,
    k  :: Int;
)
    BLAS.axpy!(μ, v.df, v.b) # take the gradient step: v.b = b + μ∇f(b) (which is a plus since df stores X(-1*(Y-Xb)))
    project_k!(v.b, k)       # P_k( β - μ∇f(β) ): preserve top k components of b
    _iht_indices(v, k)       # Update idx. (find indices of new beta that are nonzero)

    # If the k'th largest component is not unique, warn the user. 
    sum(v.idx) <= k || warn("More than k components of b is non-zero! Need: VERY DANGEROUS DARK SIDE HACK!")
end

"""
this function updates finds the non-zero index of b, and set v.idx = 1 for those indices. 
"""
function _iht_indices(
    v :: IHTVariable,
    k :: Int
)
    # set v.idx[i] = 1 if v.b[i] != 0 (i.e. find components of beta that are non-zero)
    v.idx .= v.b .!= 0

    # if idx is the 0 vector, v.idx[i] = 1 if i is one of the k largest components
    # of the gradient (stored in v.df), and set other components of idx to 0. 
    if sum(v.idx) == 0
        a = select(v.df, k, by=abs, rev=true) 
        v.idx[abs.(v.df) .>= abs(a)-2*eps()] .= true
        v.gk .= zeros(sum(v.idx))
    end

    return nothing
end