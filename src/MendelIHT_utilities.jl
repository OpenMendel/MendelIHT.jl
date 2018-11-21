"""
Object to contain intermediate variables and temporary arrays. Used for cleaner code in L0_reg
"""
mutable struct IHTVariable{T <: Float, V <: DenseVector}

    #TODO: Consider changing b and b0 to SparseVector
    b     :: Vector{T}     # the statistical model for the genotype matrix, most will be 0
    b0    :: Vector{T}     # estimated model for genotype matrix in the previous iteration
    xb    :: Vector{T}     # vector that holds x*b
    xb0   :: Vector{T}     # xb in the previous iteration
    xk    :: SnpLike{2}    # the n by k subset of the design matrix x corresponding to non-0 elements of b
    gk    :: Vector{T}     # numerator of step size. gk = df[idx]. 
    xgk   :: Vector{T}     # xk * gk, denominator of step size
    idx   :: BitVector     # idx[i] = 0 if b[i] = 0 and idx[i] = 1 if b[i] is not 0
    idx0  :: BitVector     # previous iterate of idx
    idc   :: BitVector     # idx[i] = 0 if c[i] = 0 and idx[i] = 1 if c[i] is not 0
    idc0  :: BitVector     # previous iterate of idc
    r     :: V             # n-vector of residuals
    df    :: V             # the gradient portion of the genotype part: df = ∇f(β) = snpmatrix * (y - xb - zc)
    df2   :: V             # the gradient portion of the non-genetic covariates: covariate matrix * (y - xb - zc)
    c     :: Vector{T}     # estimated model for non-genetic variates (first entry = intercept)
    c0    :: Vector{T}     # estimated model for non-genetic variates in the previous iteration
    zc    :: Vector{T}     # z * c (covariate matrix times c)
    zc0   :: Vector{T}     # z * c (covariate matrix times c) in the previous iterate
    zdf2  :: Vector{T}     # z * df2. needed to calculate non-genetic covariate contribution for denomicator of step size 
    group :: Vector{Int64} # vector denoting group membership
    p     :: Vector{T}     # vector storing the mean of a glm: p = g^{-1}( Xβ )
end

function IHTVariables{T <: Float}(
    x :: SnpLike{2},
    z :: Matrix{T},
    y :: Vector{T},
    J :: Int64,
    k :: Int64;
)
    n, p  = size(x)
    q     = size(z, 2)

    b     = zeros(T, p)
    b0    = zeros(T, p)
    xb    = zeros(T, n)
    xb0   = zeros(T, n)
    xk    = SnpArray(n, J * k - 1) # subtracting 1 because the intercept will likely be selected in the first iter
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

    return IHTVariable{T, typeof(y)}(b, b0, xb, xb0, xk, gk, xgk, idx, idx0, idc, idc0, r, df, df2, c, c0, zc, zc0, zdf2, group, p)
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
This function computes the gradient step v.b = P_k(β + μ∇f(β)) and updates idx and idc. It is an
addition here because recall that v.df stores an extra negative sign.
"""
function _iht_gradstep{T <: Float}(
    v :: IHTVariable{T},
    μ :: T,
    J :: Int,
    k :: Int,
    temp_vec :: Vector{T}
)
    BLAS.axpy!(μ, v.df, v.b)                  # take gradient step: b = b + μv, v = score
    BLAS.axpy!(μ, v.df2, v.c)                 # take gradient step: b = b + μv, v = score
##
    length_b = length(v.b)
    temp_vec[1:length_b] .= v.b
    temp_vec[length_b+1:end] .= v.c
    project_group_sparse!(temp_vec, v.group, J, k) # project [v.b; v.c] to sparse vector
    v.b .= view(temp_vec, 1:length_b)
    v.c .= view(temp_vec, length_b+1:length(temp_vec))
##
    v.idx .= v.b .!= 0                        # find new indices of new beta that are nonzero
    v.idc .= v.c .!= 0

    # If the k'th largest component is not unique, warn the user.
    sum(v.idx) <= J*k || warn("More than J*k components of b is non-zero! Need: VERY DANGEROUS DARK SIDE HACK!")
end

# """
# This function computes the score step v.b = P_k(β + μ(X^T (Y - P))) and updates idx and idc
# for logistic regression. 
# """
# function _iht_gradstep_logistic{T <: Float}(
#     v        :: IHTVariable{T},
#     μ        :: T,
#     J        :: Int,
#     k        :: Int,
#     temp_vec :: Vector{T},
#     storage  :: Vector{Vector{T}}
# )
#     #take gradient step 
#     BLAS.axpy!(μ, v.df, v.b)                  # take gradient step: b = b + μ(X^T (Y - P))
#     BLAS.axpy!(μ, v.df2, v.c)                 # take gradient step: b = b + μ(X^T (Y - P))
# ##
#     length_b = length(v.b)
#     temp_vec[1:length_b] .= v.b
#     temp_vec[length_b+1:end] .= v.c
#     project_group_sparse!(temp_vec, v.group, J, k) # project [v.b; v.c] to sparse vector
#     v.b .= view(temp_vec, 1:length_b)
#     v.c .= view(temp_vec, length_b+1:length(temp_vec))
# ##
#     v.idx .= v.b .!= 0                        # find new indices of new beta that are nonzero
#     v.idc .= v.c .!= 0

#     # If the k'th largest component is not unique, warn the user.
#     sum(v.idx) <= J*k || warn("More than J*k components of b is non-zero! Need: VERY DANGEROUS DARK SIDE HACK!")
# end

"""
When initializing the IHT algorithm, take largest elements of each group of df as nonzero
components of b. This function set v.idx = 1 for those indices.
"""
function init_iht_indices!{T <: Float}(
    v :: IHTVariable{T},
    J :: Int,
    k :: Int;
    temp_vec :: Vector{T} = zeros(length(v.df) + length(v.df2))
)
##
    length_df = length(v.df)
    temp_vec[1:length_df] .= v.df
    temp_vec[length_df+1:end] .= v.df2
    project_group_sparse!(temp_vec, v.group, J, k)
    v.df = view(temp_vec, 1:length_df)
    v.df2 = view(temp_vec, length_df+1:length(temp_vec))
##
    v.idx .= v.df .!= 0                        # find new indices of new beta that are nonzero
    v.idc .= v.df2 .!= 0

    @assert sum(v.idx) + sum(v.idc) <= J * k "Did not initialize IHT correctly: more non-zero entries in model than J*k"

    return nothing
end

"""
In `_init_iht_indices` and `_iht_gradstep`, if non-genetic cov got 
included/excluded, we must resize xk, gk, store2, and store3. 
"""
function check_covariate_supp!{T <: Float}(
    v       :: IHTVariable{T},
    storage :: Vector{Vector{T}},
)
    if sum(v.idx) != size(v.xk, 2)
        v.xk = SnpArray(size(v.xk, 1), sum(v.idx))
        v.gk = zeros(T, sum(v.idx))
        storage[2] = zeros(T, size(v.xgk)) # length n
        storage[3] = zeros(T, size(v.gk))  # length J * k
    end
end

"""
this function calculates the omega (here a / b) used for determining backtracking
"""
function _iht_omega{T <: Float}(
    v :: IHTVariable{T}
)
    # println("previous model is " * string(v.b0[find(v.b0)]))
    # println("current model is " * string(v.b[find(v.b)]))
    # println("previous intercept is " * string(v.c0))
    # println("current intercept is " * string(v.c))

    a = sqeuclidean(v.b, v.b0::Vector{T}) + sqeuclidean(v.c, v.c0::Vector{T}) :: T
    b = sqeuclidean(v.xb, v.xb0::Vector{T}) + sqeuclidean(v.zc, v.zc0::Vector{T}) :: T
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
this function for determining whether or not to backtrack. True = backtrack
"""
function _iht_logistic_backtrack{T <: Float}(
    logl      :: T, 
    prev_logl :: T,
    mu_step   :: Int,
    nstep     :: Int
)
    prev_logl > logl &&
    mu_step < nstep
end

"""
Compute the standard deviation of a SnpArray in place
"""
function std_reciprocal{T <: Float}(A::SnpArray, mean_vec::Vector{T})
    m, n = size(A)
    @assert n == length(mean_vec) "number of columns of snpmatrix doesn't agree with length of mean vector"
    std_vector = zeros(T, n)

    @inbounds for j in 1:n
        @simd for i in 1:m
            (a1, a2) = A[i, j]
            if !isnan(a1, a2) #only add if data not missing
                std_vector[j] += (convert(T, a1 + a2) - mean_vec[j])^2
            end
        end
        std_vector[j] = 1.0 / sqrt(std_vector[j] / (m - 1))
    end
    return std_vector
end

""" Projects the vector y = [y1; y2] onto the set with at most J active groups and at most
k active predictors per group. The variable group encodes group membership. Currently
assumes there are no unknown or overlaping group membership.

TODO: check if sortperm can be replaced by something that doesn't sort the whole array
"""
function project_group_sparse!{T <: Float}(
    y     :: Vector{T},
    group :: Vector{Int64},
    J     :: Int64,
    k     :: Int64
)
    @assert length(group) == length(y) "group membership vector does not have the same length as the vector to be projected on"

    groups = maximum(group)
    group_count = zeros(Int, groups)         #counts number of predictors in each group
    group_norm = zeros(groups)               #l2 norm of each group
    perm = zeros(Int64, length(y))           #vector holding the permuation vector after sorting
    sortperm!(perm, y, by = abs, rev = true)

    #calculate the magnitude of each group, where only top predictors contribute
    for i in eachindex(y)
        j = perm[i]
        n = group[j]
        if group_count[n] < k
            group_norm[n] = group_norm[n] + y[j]^2
            group_count[n] = group_count[n] + 1
        end
    end

    #go through the top predictors in order. Set predictor to 0 if criteria not met
    group_rank = zeros(Int64, length(group_norm))
    sortperm!(group_rank, group_norm, rev = true)
    group_rank = invperm(group_rank)
    fill!(group_count, 1)
    for i in eachindex(y)
        j = perm[i]
        n = group[j]
        if (group_rank[n] > J) || (group_count[n] > k)
            y[j] = 0.0
        else
            group_count[n] = group_count[n] + 1
        end
    end
end

"""
an object that houses results returned from a group IHT run
"""
immutable gIHTResults{T <: Float, V <: DenseVector}
    time  :: T
    loss  :: T
    iter  :: Int
    beta  :: V
    c     :: V
    J     :: Int64
    k     :: Int64
    group :: Vector{Int64}

    gIHTResults{T,V}(time, loss, iter, beta, c, J, k, group) where {T <: Float, V <: DenseVector{T}} = new{T,V}(time, loss, iter, beta, c, J, k, group)
end

# strongly typed external constructor for gIHTResults
gIHTResults(time::T, loss::T, iter::Int, beta::V, c::V, J::Int, k::Int, group::Vector{Int}) where {T <: Float, V <: DenseVector{T}} = gIHTResults{T, V}(time, loss, iter, beta, c, J, k, group)

"""
a function to display gIHTResults object
"""
function Base.show(io::IO, x::gIHTResults)
    println(io, "IHT results:")
    println(io, "\nCompute time (sec):     ", x.time)
    println(io, "Final loss:             ", x.loss)
    println(io, "Iterations:             ", x.iter)
    println(io, "Max number of groups:   ", x.J)
    println(io, "Max predictors/group:   ", x.k)
    println(io, "IHT estimated ", countnz(x.beta), " nonzero coefficients.")
    non_zero = find(x.beta)
    print(io, DataFrame(Group=x.group[non_zero], Predictor=non_zero, Estimated_β=x.beta[non_zero]))
    println(io, "\n\nIntercept of model = ", x.c[1])

    return nothing
end
"""
Calculates the Prior Weighting for IHT.
Returns a weight array (my_snpweights) (1,10000) and a MAF array (my_snpMAF ) (1,10000).
"""
function calculate_snp_weights(
    x        :: SnpLike{2},
    y        :: Vector{Float64},
    k        :: Int,
    v        :: IHTVariable,
    use_maf  :: Bool,
    maf      :: Array{Float64,1}
)
    # get my_snpMAF from x
    ALLELE_MAX = 2 * size(x,1)
    my_snpMAF = maf' # crashes line 308 npzwrite
    my_snpMAF = convert(Matrix{Float64},my_snpMAF)

    # GORDON - CALCULATE CONSTANT WEIGHTS - another weighting option
    my_snpweights_const = copy(my_snpMAF) # only to allocate my_snpweights_const
    # need to test for bad user input !!!
    for i = 1:size(my_snpweights_const,2)
        my_snpweights_const[1,i] = keyword["pw_algorithm_value"]
    end

    # GORDON - CALCULATE WEIGHTS BASED ON p=MAF, 1/(2√pq) SUGGESTED BY BEN AND HUA ZHOU
    my_snpweights_p = my_snpMAF      # p_hat
    my_snpweights = 2 * sqrt.(my_snpweights_p .* (1 - my_snpweights_p))   # just verifying 2 * sqrtm(p .* q) == 1.0 OK!
    my_snpweights_huazhou = my_snpweights
    my_snpweights = my_snpweights .\ 1      # this works! to get reciprocal of each element
    my_snpweights_huazhou_reciprocal = my_snpweights

    # DECIDE NOW WHICH WEIGHTS TO APPLY !!!
    if true # to ensure an algorithm, do this regardless
        my_snpweights = copy(my_snpweights_const)    # Ben/Kevin this is currently at 1.0 for testing null effect
    end
    if keyword["prior_weights"] == "maf"
        my_snpweights = copy(my_snpweights_huazhou_reciprocal)
    end
    return my_snpMAF, my_snpweights
end

"""
Function that saves `b`, `xb`, `idx`, `idc`, `c`, and `zc` after each iteration. 
"""
function save_prev!{T <: Float}(
    v :: IHTVariable{T}
)
    copy!(v.b0, v.b)     # b0 = b
    copy!(v.xb0, v.xb)   # Xb0 = Xb
    copy!(v.idx0, v.idx) # idx0 = idx
    copy!(v.idc0, v.idc) # idc0 = idc
    copy!(v.c0, v.c)     # c0 = c
    copy!(v.zc0, v.zc)   # Zc0 = Zc
end

"""
This function computes the best step size μ for normal responses. 
"""
function _iht_stepsize{T <: Float}(
    v        :: IHTVariable{T},
    z        :: Matrix{T},
    mean_vec :: Vector{T},
    std_vec  :: Vector{T},
    storage  :: Vector{Vector{T}}
)
    # store relevant components of gradient (gk is numerator of step size). 
    v.gk .= view(v.df, v.idx)

    # compute the denominator of step size using only relevant components 
    A_mul_B!(v.xgk, v.zdf2, v.xk, z, v.gk, view(v.df2, v.idc), view(mean_vec, v.idx), view(std_vec, v.idx), storage)

    # warn if xgk only contains zeros
    all(v.xgk .== zero(T)) && warn("Entire active set has values equal to 0")

    # compute step size. Note intercept is separated from x, so gk & xgk is missing an extra entry equal to 1^T (y-Xβ-intercept) = sum(v.r)
    μ = (((sum(abs2, v.gk) + sum(abs2, view(v.df2, v.idc))) / (sum(abs2, v.xgk) + sum(abs2, v.zdf2)))) :: T

    # check for finite stepsize
    isfinite(μ) || throw(error("Step size is not finite, is active set all zero?"))

    return μ
end

"""
This function computes the best step size μ for bernoulli responses. 
"""
function _logistic_stepsize{T <: Float}(
    v         :: IHTVariable{T},
    x         :: SnpLike{2},
    z         :: Matrix{T},
    mean_vec  :: Vector{T},
    std_vec   :: Vector{T},
)
    #BELOW IS ASSUMING WE IDENTIFIED CORRECT SUPPORT: THUS THE STEP SIZE CALCULATION
    #COMUPTES NUMERATOR AND DENOMINATOR USING ONLY THE SUPPORT SET
    # store relevant components of x
    v.xk .= view(x, :, v.idx)

    # store relevant components of gradient (gk is numerator of step size). 
    v.gk .= view(v.df, v.idx)

    #compute J = X^T * P * X
    X = convert(Matrix{T}, v.xk)
    normalize!(X, view(mean_vec, v.idx), view(std_vec, v.idx))
    full_X = [X view(z, :, v.idc)]
    J = full_X' * (diagm(v.p .* (1.0 .- v.p)) * full_X) #J = (p + q) by (p + q)

    #compute denominator 
    full_v = [view(v.df, v.idx) ; view(v.df2, v.idc)]
    denom = full_v' * (J * full_v)

    # compute step size. Note intercept is separated from x, so gk & xgk is missing an extra entry equal to 1^T (y-Xβ-intercept) = sum(v.r)
    μ = ((sum(abs2, v.gk) + sum(abs2, view(v.df2, v.idc))) / denom) :: T





    #compute J = X^T * P * X
    # X = convert(Matrix{T}, x)
    # normalize!(X, mean_vec, std_vec)
    # full_X = [X z]
    # J = full_X' * (diagm(v.p .* (1.0 .- v.p)) * full_X) #J = (p + q) by (p + q)

    # #compute denominator 
    # full_v = [v.df ; v.df2]
    # denom = full_v' * (J * full_v)

    # # compute step size.
    # μ = (sum(abs2, full_v) / denom) :: T

    # # check for finite stepsize
    # isfinite(μ) || throw(error("Step size is not finite, is active set all zero?"))

    return μ
end

function normalize!{T <: Float}(
    X        :: AbstractMatrix{T},
    mean_vec :: AbstractVector{T},
    std_vec  :: AbstractVector{T}
)

    @assert size(X, 2) == length(mean_vec) "normalize!: X and mean_vec have different size"
    for i in 1:size(X, 2)
        X[:, i] .= (X[:, i] .- mean_vec[i]) .* std_vec[i]
    end
end

"""
    Returns true if condition satisfied. 

For logistic regression, checks whether y[i] == 1 or y[i] == 0. 
For poisson regression, checks whether y is a vector of integer 
"""
function check_y_content{T <: Float}(
    y   :: Vector{T},
    glm :: String
)
    if glm == "poisson"
        @assert typeof(y) == Vector{Int} "Poisson regression requires the response vector y to be integer valued. "
    end
    
    if glm == "logistic"
        for i in 1:length(y)
            if y[i] != 0 && y[i] != 1
                throw(error("Logistic regression requires the response vector y to be 0 or 1"))
            end
        end
    end

    return nothing
end

"""
This is a wrapper linear algebra function that computes [C1 ; C2] = [A1 ; A2] * [B1 ; B2] 
where A1 is a snpmatrix and A2 is a dense Matrix{Float}. Used for cleaner code. 

Here we are separating the computation because A1 is stored in compressed form while A2 is 
uncompressed (float64) matrix. This means that they cannot be stored in the same data 
structure. 
"""
function A_mul_B!{T <: Float}(
    C1       :: AbstractVector{T},
    C2       :: AbstractVector{T},
    A1       :: SnpLike{2},
    A2       :: AbstractMatrix{T},
    B1       :: AbstractVector{T},
    B2       :: AbstractVector{T},
    mean_vec :: AbstractVector{T},
    std_vec  :: AbstractVector{T},
    storage  :: Vector{Vector{T}}
)
    SnpArrays.A_mul_B!(C1, A1, B1, mean_vec, std_vec, storage[2], storage[3])
    BLAS.A_mul_B!(C2, A2, B2)
end

"""
This is a wrapper linear algebra function that computes [C1 ; C2] = [A1 ; A2]^T * [B1 ; B2] 
where A1 is a snpmatrix and A2 is a dense Matrix{Float}. Used for cleaner code. 

Here we are separating the computation because A1 is stored in compressed form while A2 is 
uncompressed (float64) matrix. This means that they cannot be stored in the same data 
structure. 
"""
function At_mul_B!{T <: Float}(
    C1       :: AbstractVector{T},
    C2       :: AbstractVector{T},
    A1       :: SnpLike{2},
    A2       :: AbstractMatrix{T},
    B1       :: AbstractVector{T},
    B2       :: AbstractVector{T},
    mean_vec :: AbstractVector{T},
    std_vec  :: AbstractVector{T},
    storage  :: Vector{Vector{T}}
)
    SnpArrays.At_mul_B!(C1, A1, B1, mean_vec, std_vec, storage[1])
    BLAS.At_mul_B!(C2, A2, B2)
end


"""
This function calculates the score (gradient) direction for different glm models, and stores
the result in v.df and v.df2, where the former stores the gradient associated with the snpmatrix
direction and the latter associates with the intercept + other non-genetic covariates. 

For normal responses, score = ∇f(β) = -X^T (Y - Xβ)
For logistics responses, score = -∇L(β) = -X^T (Y - P) (using logit link)
For Poisson responses, score = -∇L(β) = X^T (Y - Λ)
"""
function update_df!{T <: Float}(
    glm       :: String,
    v         :: IHTVariable{T}, 
    x         :: SnpLike{2},
    z         :: Matrix{T},
    y         :: Vector{T},
    mean_vec  :: AbstractVector{T},
    std_vec   :: AbstractVector{T},
    storage   :: Vector{Vector{T}}
)
    if glm == "normal"
        At_mul_B!(v.df, v.df2, x, z, v.r, v.r, mean_vec, std_vec, storage)
    elseif glm == "logistic"
        # inverse_link!(v)     #first update the P vector
        v.p .= logistic.(v.xb) #first update the P vector
        y_minus_p = y - v.p
        At_mul_B!(v.df, v.df2, x, z, y_minus_p, y_minus_p, mean_vec, std_vec, storage)
    else
        throw(error("unsupport glm method."))
    end
end

"""
This function calculates the inverse link of a glm model: p = g^{-1}( Xβ )

In poisson and logistic, the score (gradient) direction is X'(Y - P) where 
p_i depends on x and β. This function updates this P vector. 
"""
function inverse_link!{T <: Float}(
    v :: IHTVariable{T}
)
    @inbounds @simd for i in eachindex(v.p)
        # xβ = dot(view(x.A1, i, :), b) + dot(view(x.A2, i, :), b) + dot(view(z, i, :), c)
        v.p[i] = 1.0 / (1.0 + e^(-v.xb[i] - v.zc[i]))
    end
end


"""
This function computes the loglikelihood of a model β for a given glm response
"""
function compute_logl{T <: Float}(
    v        :: IHTVariable{T},
    x        :: SnpLike{2},
    z        :: Matrix{T},
    y        :: Vector{T},
    glm      :: String,
    mean_vec :: AbstractVector{T},
    std_vec  :: AbstractVector{T},
    storage  :: Vector{Vector{T}}
) 
    if glm == "logistic"
        # Xβ = zeros(size(x, 1))
        # SnpArrays.A_mul_B!(Xβ, x, v.b, mean_vec, std_vec, storage[2], storage[1])
        # Xβ .+= z * v.c
        # return dot(y, Xβ) - sum(log.(1.0 .+ exp.(Xβ)))
        return dot(y, v.xb + v.zc) - sum(log.(1.0 .+ exp.(v.xb + v.zc))) 
        # return dot(y, v.xb + v.zc) - sum(log.(1.0 .+ exp.(v.xb + v.zc))) - 30.0*(norm(v.b, 2) + norm(v.c, 2)) 
    else 
        error("compute_logl: currently only supports logistic")
    end
end





