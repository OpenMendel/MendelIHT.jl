"""
Object to contain intermediate variables and temporary arrays. Used for cleaner code in L0_reg
"""
mutable struct IHTVariable{T <: Float, V <: DenseVector}

   #TODO: Consider changing b and b0 to SparseVector
   #TODO: Add in itc
   # itc  :: T             # the intercept
   # itc0 :: T             # old intercept
   b    :: Vector{T}     # the statistical model, most will be 0
   b0   :: Vector{T}     # previous estimated model in the mm step
   xb   :: Vector{T}     # vector that holds x*b 
   xb0  :: Vector{T}     # previous xb in the mm step
   xk   :: SnpLike{2}    # the n by k subset of the design matrix x corresponding to non-0 elements of b
   gk   :: Vector{T}     # gk = df[idx] is a temporary array of length `k` that arises as part of the gradient calculations. I avoid doing full gradient calculations since most of `b` is zero. 
   xgk  :: Vector{T}     # x * gk also part of the gradient calculation 
   idx  :: BitVector     # BitArray indices of nonzeroes in b for A_mul_B
   idx0 :: BitVector     # previous iterate of idx
   r    :: V             # n-vector of residuals
   df   :: V             # the gradient: df = -x' * (y - xb - intercept)
end

function IHTVariables{T <: Float}(
    x :: SnpLike{2},
    y :: Vector{T},
    k :: Int64
) 
    n, p = size(x) 

    #check if k is sensible
    @assert k <= p "k cannot exceed the number of SNPs"
    @assert k > 0  "k must be positive integer"
    p += 1 # add 1 for the intercept, need to change this to use itc later

    # itc  = zero(T)
    # itc0 = zero(T)
    b    = zeros(T, p)
    b0   = zeros(T, p)
    xb   = zeros(T, n)
    xb0  = zeros(T, n)
    xk   = SnpArray(n, k)
    gk   = zeros(T, k)
    xgk  = zeros(T, n)
    idx  = falses(p) 
    idx0 = falses(p)
    r    = zeros(T, n)
    df   = zeros(T, p)
    return IHTVariable{T, typeof(y)}(b, b0, xb, xb0, xk, gk, xgk, idx, idx0, r, df)
end

#"""
#Returns ω, a constant we need to bound the step size μ to guarantee convergence. 
#"""
#
#function compute_ω!(v::IHTVariable, snpmatrix::Matrix{Float64})
#    #update v.xb
#    A_mul_B!(v.xb, snpmatrix, v.b)
#
#    #calculate ω efficiently (old b0 and xb0 have been copied before calling iht!)
#    return sqeuclidean(v.b, v.b0) / sqeuclidean(v.xb, v.xb0)
#end

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
This function computes the gradient step v.b = P_k(β + μ∇f(β)), and updates v.idx. 
Recall calling axpy! implies v.b = v.b + μ*v.df, but v.df stores an extra negative sign.
"""
function _iht_gradstep{T <: Float}(
   v  :: IHTVariable{T},
   μ  :: Float64,
   k  :: Int;
)
   BLAS.axpy!(μ, v.df, v.b) # take the gradient step: v.b = b + μ∇f(b) (which is an addition since df stores X(-1*(Y-Xb)))
   project_k!(v.b, k)       # P_k( β - μ∇f(β) ): preserve top k components of b
   _iht_indices(v, k)       # Update idx. (find indices of new beta that are nonzero)

   # If the k'th largest component is not unique, warn the user. 
   sum(v.idx) <= k || warn("More than k components of b is non-zero! Need: VERY DANGEROUS DARK SIDE HACK!")
end

"""
this function updates the non-zero index of b, and set v.idx = 1 for those indices. 
"""
function _iht_indices{T <: Float}(
    v :: IHTVariable{T},
    k :: Int;
)
    # which components of beta are nonzero?
    #update_indices!(v.idx, v.b)
    v.idx .= v.b .!= 0

    # if current vector is 0,
    # then take largest elements of d as nonzero components for b
    if sum(v.idx) == 0
        a = select(v.df, k, by=abs, rev=true) :: T
#        threshold!(IDX, g, abs(a), n=p)
        v.idx[abs.(v.df) .>= abs(a)-2*eps()] = true
        v.gk = zeros(T, sum(v.idx))
    end

    return nothing
end
"""
this function calculates the omega (here a / b) used for determining backtracking
"""
function _iht_omega{T <: Float}(
    v :: IHTVariable{T}
)
    a = sqeuclidean(v.b, v.b0::Vector{T}) :: T
    b = sqeuclidean(v.xb, v.xb0::Vector{T}) :: T
    return a, b
end

"""
this function for determining whether or not to backtrack. True = backtrack
"""
function _iht_backtrack{T <: Float}(
    v       :: IHTVariable{T},
    ot      :: T,
    ob      :: T,
    mu      :: T,
    mu_step :: Int,
    nstep   :: Int
)
    mu*ob > 0.99*ot              &&
    sum(v.idx) != 0              &&
    sum(xor.(v.idx,v.idx0)) != 0 &&
    mu_step < nstep
end

"""
Compute the standard deviation of a SnpArray in place
"""
function std_reciprocal(A::SnpArray, mean_vec::Vector{Float64})
    m, n = size(A)
    @assert n == length(mean_vec) "number of columns of snpmatrix doesn't agree with length of mean vector"
    std_vector = zeros(Float64, n) 

    @inbounds for j in 1:n
        @simd for i in 1:m
            (a1, a2) = A[i, j]
            if !isnan(a1, a2) #only add if data not missing
                std_vector[j] += (convert(Float64, a1 + a2) - mean_vec[j])^2
            end 
        end
        std_vector[j] = 1.0 / sqrt(std_vector[j] / (m - 1))
    end
    return std_vector
end


