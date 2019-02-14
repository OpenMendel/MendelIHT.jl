"""
This function computes the gradient step v.b = P_k(β + μ∇f(β)) and updates idx and idc. It is an
addition here because recall that v.df stores an extra negative sign.
"""
function _iht_gradstep(
    v :: IHTVariable{T},
    μ :: T,
    J :: Int,
    k :: Int,
    temp_vec :: Vector{T}
) where {T <: Float}
    #v.df is dense
    BLAS.axpy!(μ, v.df, v.b)  # take gradient step: b = b + μv, v = score
    BLAS.axpy!(μ, v.df2, v.c) # take gradient step: b = b + μv, v = score

    # copy v.b and v,c to temp_vec and project temp_vec
    length_b = length(v.b)
    temp_vec[1:length_b] .= v.b
    temp_vec[length_b+1:end] .= v.c
    project_group_sparse!(temp_vec, v.group, J, k) # project [v.b; v.c] to sparse vector
    v.b .= view(temp_vec, 1:length_b)
    v.c .= view(temp_vec, length_b+1:length(temp_vec))

    #recompute current support
    v.idx .= v.b .!= 0
    v.idc .= v.c .!= 0
    _choose!(v, J, k) # if more than J*k entries are selected, randomly choose J*k of them
end

"""
When initializing the IHT algorithm, take largest elements of each group of df as nonzero
components of b. This function set v.idx = 1 for those indices.
"""
function init_iht_indices!(
    v :: IHTVariable{T},
    J :: Int,
    k :: Int;
    temp_vec :: Vector{T} = zeros(length(v.df) + length(v.df2))
) where {T <: Float}

    a = sort([v.df; v.df2], rev=true)[k * J]
    v.idx .= v.df .>= a
    v.idc .= v.df2 .>= a
    _choose!(v, J, k) # if more than J*k entries are selected, randomly choose J*k of them
end

"""
if more than k entries are selected after projection, randomly select top k entries.
This can happen if entries of b are equal to each other.
"""
function _choose!(
    v :: IHTVariable{T},
    J :: Int,
    k :: Int;
) where {T <: Float}
    while sum(v.idx) + sum(v.idc) > J * k
        num = sum(v.idx) + sum(v.idc) - J * k
        idx_length = length(v.idx)
        temp = [v.idx; v.idc]

        nonzero_idx = findall(x -> x == true, temp)
        pos = nonzero_idx[rand(1:length(nonzero_idx))] #randomly choose 1 to set to 0
        pos > idx_length ? v.idc[pos - idx_length] = 0 : v.idx[pos] = 0
    end
end

"""
In `_init_iht_indices` and `_iht_gradstep`, if non-genetic cov got 
included/excluded, we must resize xk, gk, store2, and store3. 
"""
function check_covariate_supp!(v :: IHTVariable{T}) where {T <: Float}
    if sum(v.idx) != size(v.xk, 2)
        v.xk = zeros(T, size(v.xk, 1), sum(v.idx))
        v.gk = zeros(T, sum(v.idx))
    end
end

"""
this function calculates the omega (here a / b) used for determining backtracking
"""
function _iht_omega(v :: IHTVariable{T}) where {T <: Float}
    a = sqeuclidean(v.b, v.b0::Vector{T}) + sqeuclidean(v.c, v.c0::Vector{T}) :: T
    b = sqeuclidean(v.xb, v.xb0::Vector{T}) + sqeuclidean(v.zc, v.zc0::Vector{T}) :: T
    return a, b
end

"""
this function for determining whether or not to backtrack for normal least squares. True = backtrack
"""
function _normal_backtrack(
    v       :: IHTVariable{T},
    ot      :: T,
    ob      :: T,
    mu      :: T,
    mu_step :: Int,
    nstep   :: Int
) where {T <: Float}
    mu*ob > 0.99*ot              &&
    sum(v.idx) != 0              &&
    sum(xor.(v.idx,v.idx0)) != 0 &&
    mu_step < nstep
end

"""
this function for determining whether or not to backtrack for logistic regression. True = backtrack
"""
function _logistic_backtrack(
    logl      :: T, 
    prev_logl :: T,
    mu_step   :: Int,
    nstep     :: Int
) where {T <: Float}
    prev_logl > logl && mu_step < nstep 
end

"""
this function for determining whether or not to backtrack for poisson regression. True = backtrack

Note we require the model coefficients to be "small"  (that is, max entry not greater than 10) to 
prevent loglikelihood blowing up in first few iteration.
"""
function _poisson_backtrack(
    v         :: IHTVariable{T},
    logl      :: T, 
    prev_logl :: T,
    mu_step   :: Int,
    nstep     :: Int
) where {T <: Float}
    mu_step >= nstep  && return false
    prev_logl > logl && return true
end

"""
Compute the standard deviation of a SnpArray in place. Note this function assumes all SNPs are not missing.
Otherwise, the inner loop should only add if data not missing.
"""
function std_reciprocal(
    x        :: SnpBitMatrix, 
    mean_vec :: Vector{T}
) where {T <: Float}
    m, n = size(x)
    @assert n == length(mean_vec) "number of columns of snpmatrix doesn't agree with length of mean vector"
    std_vector = zeros(T, n)

    @inbounds for j in 1:n
        @simd for i in 1:m
            a1 = x.B1[i, j]
            a2 = x.B2[i, j]
            std_vector[j] += (convert(T, a1 + a2) - mean_vec[j])^2
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
function project_group_sparse!(
    y     :: Vector{T},
    group :: Vector{Int64},
    J     :: Int64,
    k     :: Int64
) where {T <: Float}
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
Calculates the Prior Weighting for IHT.
Returns a weight array (my_snpweights) (1,10000) and a MAF array (my_snpMAF ) (1,10000).
"""
function calculate_snp_weights(
    x        :: SnpArray,
    y        :: Vector{T},
    k        :: Int,
    v        :: IHTVariable,
    use_maf  :: Bool,
    maf      :: Array{T,1}
) where {T <: Float}
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
function save_prev!(
    v :: IHTVariable{T}
) where {T <: Float}
    copyto!(v.b0, v.b)     # b0 = b
    copyto!(v.xb0, v.xb)   # Xb0 = Xb
    copyto!(v.idx0, v.idx) # idx0 = idx
    copyto!(v.idc0, v.idc) # idc0 = idc
    copyto!(v.c0, v.c)     # c0 = c
    copyto!(v.zc0, v.zc)   # Zc0 = Zc
end

"""
This function computes the best step size μ for normal responses. 
"""
function _iht_stepsize(
    v        :: IHTVariable{T},
    z        :: AbstractMatrix{T},
) where {T <: Float}
    # store relevant components of gradient (gk is numerator of step size). 
    v.gk .= view(v.df, v.idx)

    # compute the denominator of step size using only relevant components
    A_mul_B!(v.xgk, v.zdf2, v.xk, view(z, :, v.idc), v.gk, view(v.df2, v.idc))

    # warn if xgk only contains zeros
    all(v.xgk .== zero(T)) && warn("Entire active set has values equal to 0")

    # compute step size. Note intercept is separated from x, so gk & xgk is missing an extra entry equal to 1^T (y-Xβ-intercept) = sum(v.r)
    μ = (((sum(abs2, v.gk) + sum(abs2, view(v.df2, v.idc))) / (sum(abs2, v.xgk) + sum(abs2, v.zdf2)))) :: T

    # check for finite stepsize
    isfinite(μ) || throw(error("Step size is not finite, check if active set is all zero or if your SnpArray have missing values."))

    return μ
end

"""
This function computes the best step size μ for bernoulli responses. Here the computation
is done on the n by k support set of the SNP matrix. 
"""
function _logistic_stepsize(
    v :: IHTVariable{T},
    z :: AbstractMatrix{T},
) where {T <: Float}

    # store relevant components of gradient 
    v.gk .= view(v.df, v.idx)
    A_mul_B!(v.xgk, v.zdf2, v.xk, view(z, :, v.idc), v.gk, view(v.df2, v.idc))

    #compute denominator of step size
    denom = (v.xgk + v.zdf2)' * ((v.p .* (1 .- v.p)) .* (v.xgk + v.zdf2))

    # compute step size. Note non-genetic covariates are separated from x
    μ = ((sum(abs2, v.gk) + sum(abs2, view(v.df2, v.idc))) / denom) :: T

    return μ
end

function _poisson_stepsize(
    v :: IHTVariable{T},
    z :: AbstractMatrix{T},
) where {T <: Float}

    # store relevant components of gradient
    v.gk .= view(v.df, v.idx)
    A_mul_B!(v.xgk, v.zdf2, v.xk, view(z, :, v.idc), v.gk, view(v.df2, v.idc))

    #compute denominator and numerator of step size
    denom = (v.xgk + v.zdf2)' * (v.p .* (v.xgk + v.zdf2))
    numer = (sum(abs2, v.gk) + sum(abs2, view(v.df2, v.idc)))

    # compute step size. Note non-genetic covariates are separated from x
    μ = (numer / denom) :: T
end

# function normalize!(
#     X        :: AbstractMatrix{T},
#     mean_vec :: AbstractVector{T},
#     std_vec  :: AbstractVector{T}
# ) where {T <: Float}

#     @assert size(X, 2) == length(mean_vec) "normalize!: X and mean_vec have different size"
#     for i in 1:size(X, 2)
#         X[:, i] .= (X[:, i] .- mean_vec[i]) .* std_vec[i]
#     end
# end

"""
    Returns true if condition satisfied. 

For logistic regression, checks whether y[i] == 1 or y[i] == 0. 
For poisson regression, checks whether y is a vector of integer 
"""
function check_y_content(
    y   :: Vector{T},
    glm :: String
) where {T <: Float}
    if glm == "poisson"
        try
            convert(Vector{Int64}, y)
        catch e
            warn("cannot convert response vector y to be integer valued. Please check if y is count data.")
        end
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
function A_mul_B!(
    C1       :: AbstractVector{T},
    C2       :: AbstractVector{T},
    A1       :: SnpBitMatrix{T},
    A2       :: AbstractMatrix{T},
    B1       :: AbstractVector{T},
    B2       :: AbstractVector{T},
) where {T <: Float}
    SnpArrays.mul!(C1, A1, B1)
    LinearAlgebra.mul!(C2, A2, B2)
end

function A_mul_B!(
    C1       :: AbstractVector{T},
    C2       :: AbstractVector{T},
    A1       :: AbstractMatrix{T},
    A2       :: AbstractMatrix{T},
    B1       :: AbstractVector{T},
    B2       :: AbstractVector{T},
) where {T <: Float}
    LinearAlgebra.mul!(C1, A1, B1)
    LinearAlgebra.mul!(C2, A2, B2)
end

"""
This is a wrapper linear algebra function that computes [C1 ; C2] = [A1 ; A2]^T * [B1 ; B2] 
where A1 is a snpmatrix and A2 is a dense Matrix{Float}. Used for cleaner code. 

Here we are separating the computation because A1 is stored in compressed form while A2 is 
uncompressed (float64) matrix. This means that they cannot be stored in the same data 
structure. 
"""
function At_mul_B!(
    C1 :: AbstractVector{T},
    C2 :: AbstractVector{T},
    A1 :: SnpBitMatrix{T},
    A2 :: AbstractMatrix{T},
    B1 :: AbstractVector{T},
    B2 :: AbstractVector{T},
) where {T <: Float}
    SnpArrays.mul!(C1, A1', B1)
    LinearAlgebra.mul!(C2, A2', B2)
end

function At_mul_B!(
    C1 :: AbstractVector{T},
    C2 :: AbstractVector{T},
    A1 :: AbstractMatrix{T},
    A2 :: AbstractMatrix{T},
    B1 :: AbstractVector{T},
    B2 :: AbstractVector{T},
) where {T <: Float}
    LinearAlgebra.mul!(C1, A1', B1)
    LinearAlgebra.mul!(C2, A2', B2)
end

"""
This function calculates the score (gradient) for different glm models, and stores the
result in v.df and v.df2. The former stores the gradient associated with the snpmatrix
direction and the latter associates with the intercept + other non-genetic covariates. 

The following summarizes the score direction for different responses.
    Normal = ∇f(β) = -X^T (Y - Xβ)
    Binary = ∇L(β) = -X^T (Y - P) (using logit link)
    Count  = ∇L(β) = X^T (Y - Λ)
"""
function update_df!(
    glm    :: String,
    v      :: IHTVariable{T}, 
    x      :: SnpBitMatrix{T},
    z      :: AbstractMatrix{T},
    y      :: AbstractVector{T};
) where {T <: Float}
    if glm == "normal"
        @. v.r = y - v.xb - v.zc
        At_mul_B!(v.df, v.df2, x, z, v.r, v.r)
    elseif glm == "logistic"
        @. v.p = logistic(v.xb + v.zc)
        @. v.ymp = y - v.p
        At_mul_B!(v.df, v.df2, x, z, v.ymp, v.ymp)
    elseif glm == "poisson"
        @. v.p = exp(v.xb + v.zc)
        @. v.ymp = y - v.p
        At_mul_B!(v.df, v.df2, x, z, v.ymp, v.ymp)
    else
        throw(error("computing gradient for an unsupport glm method: " * glm))
    end
end

"""
`compute_logl` computes the loglikelihood of a model β for a given glm response
"""
function compute_logl(
    v      :: IHTVariable{T},
    y      :: AbstractVector{T},
    glm    :: String;
) where {T <: Float}
    if glm == "logistic"
        return _logistic_logl(y, v.xb + v.zc)
    elseif glm == "poisson"
        return _poisson_logl(y, v.xb + v.zc)
    else 
        error("compute_logl: currently only supports logistic and poisson")
    end
end

function _logistic_logl(
    y      :: Vector{T}, 
    xb     :: Vector{T};
) where {T <: Float64}
    logl = 0.0
    @inbounds for i in eachindex(y)
        logl += y[i]*xb[i] - log(1.0 + exp(xb[i]))
    end
    return logl
end

function _poisson_logl(
    y      :: Vector{T}, 
    xb     :: Vector{T};
) where {T <: Float64}
    logl = 0.0
    @inbounds for i in eachindex(y)
        logl += y[i]*xb[i] - exp(xb[i])
    end
    return logl
end

"""
Simple function for simulating a random SnpArray without missing value.
This is for testing purposes only. 
"""
function simulate_random_snparray(
    n :: Int64,
    p :: Int64,
)
    #first simulate a random {0, 1, 2} matrix with each SNP drawn from Binomial(2, r[i])
    x_tmp = zeros(UInt8, n, p)
    snps = zeros(UInt8, n)
    mafs = zeros(Float64, p)
    for j in 1:p
        mafs[j] = _generate_binomials!(snps)
        x_tmp[:, j] .= snps
    end

    #fill the SnpArray with the corresponding x_tmp entry
    return _make_snparray(x_tmp), mafs
end

"""
For each sample, generate a minor allele count of {0, 1, 2} with maf ∈ (0, 0.5).
If 5 or less minor allele is present in whole sample, regenerate with a different maf. 
"""
function _generate_binomials!(snps :: Vector{UInt8})
    n = length(snps)
    minor_alleles = 0
    maf = 0
    while minor_alleles <= 5
        maf = 0.5rand()
        for i in 1:n
            snps[i] = convert(UInt8, rand(Binomial(2, maf)))
        end
        minor_alleles = sum(snps)
    end
    return maf
end

"""
Make a random SnpArray based on given Matrix{Float64} of 0~2.
This is for testing purposes only. 
"""
function _make_snparray(x_temp :: Matrix{Float64})
    n, p = size(x_temp)
    x = SnpArray(undef, n, p)
    for i in 1:(n*p)
        if x_temp[i] == 0
            x[i] = 0x00
        elseif x_temp[i] == 1
            x[i] = 0x02
        elseif x_temp[i] == 2
            x[i] = 0x03
        else 
            throw(error("matrix shouldn't have missing values!"))
        end
    end
    return x
end

"""
Make a SnpArray from a matrix of UInt8. This is for testing purposes only. 
"""
function _make_snparray(x_temp :: AbstractMatrix{UInt8})
    n, p = size(x_temp)
    x = SnpArray(undef, n, p)
    for i in 1:(n*p)
        if x_temp[i] == 0x00
            x[i] = 0x00
        elseif x_temp[i] == 0x01
            x[i] = 0x02
        elseif x_temp[i] == 0x02
            x[i] = 0x03
        else
            throw(error("matrix shouldn't have missing values!"))
        end
    end
    return x
end

"""
Performs generalized linear regression. X is the design matrix, y is 
the response vector. This function is used as the debiasing step. 
"""
function regress(
    X     :: AbstractMatrix{T}, 
    y     :: AbstractVector{T}, 
    model :: AbstractString
) where {T <: Float}

  if model != "normal" && model != "logistic" && model != "poisson"
    throw(ArgumentError(
      "The only model choices are linear, logistic, and poisson.\n \n"))
  end
  #
  # Create the score vector, information matrix, estimate, a work
  # vector z, and the loglikelihood.
  #
  (n, p) = size(X)
  @assert n == length(y)
  score = zeros(p)
  information = zeros(p, p)
  estimate = zeros(p)
  z = zeros(n)
  #
  # Handle linear regression separately.
  #
  if model == "normal"
    BLAS.gemv!('N', 1.0, X, estimate, 0.0, z) # z = X * estimate
    BLAS.axpy!(-1.0, y, z) # z = z - y
    score = BLAS.gemv('T', -1.0, X, z) # score = - X' * (z - y)
    information = BLAS.gemm('T', 'N', X, X) # information = X' * X
    estimate = information \ score
    BLAS.gemv!('N', 1.0, X, estimate, 0.0, z) # z = X * estimate
    obj = - 0.5 * n * log(sum(abs2, y - z) / n) - 0.5 * n
    return (estimate, obj)
  end
          # #
          # # Prepare for logistic and Poisson regression by estimating the 
          # # intercept.
          # #
          # if model == "logistic"
          #   estimate[1] = log(mean(y) / (1.0 - mean(y)))
          # elseif model == "poisson"
          #   estimate[1] = log(mean(y))
          # else
          #   throw(ArgumentError(
          #     "The only model choices are linear, logistic, and Poisson.\n \n"))
          # end
  #
  # Initialize the loglikelihood and the convergence criterion.
  #
  v = zeros(p)
  obj = 0.0
  old_obj = 0.0
  epsilon = 1e-6
  # 
  #  Estimate parameters by the scoring algorithm.
  #
  for iteration = 1:10
    #
    # Initialize the score and information.
    #
    fill!(score, 0.0)
    fill!(information, 0.0)
    #
    # Compute the score, information, and loglikelihood (obj).
    #
    BLAS.gemv!('N', 1.0, X, estimate, 0.0, z) # z = X * estimate
    clamp!(z, -20.0, 20.0) 
    if model == "logistic"
      z = exp.(-z)
      z = 1.0 ./ (1.0 .+ z)
      w = z .* (1.0 .- z)
      BLAS.axpy!(-1.0, y, z) # z = z - y
      score = BLAS.gemv('T', -1.0, X, z) # score = - X' * (z - y)
      w = sqrt.(w)
      lmul!(Diagonal(w), X) # diag(w) * X
      information = BLAS.gemm('T', 'N', X, X) # information = X' * W * X
      w = 1.0 ./ w
      lmul!(Diagonal(w), X)
    elseif model == "poisson"
      z = exp.(z)
      w = copy(z)
      BLAS.axpy!(-1.0, y, z) # z = z - y
      score = BLAS.gemv('T', -1.0, X, z) # score = - X' * (z - y)
      w = sqrt.(w)
      lmul!(Diagonal(w), X) # diag(w) * X
      information = BLAS.gemm('T', 'N', X, X) # information = X' * W * X
      w = 1.0 ./ w
      lmul!(Diagonal(w), X)
    end
    #
    # Compute the scoring increment.
    #
    increment = information \ score
    #
    # Step halve to produce an increase in the loglikelihood.
    #
    steps = -1
    for step_halve = 0:3
      steps = steps + 1
      obj = 0.0
      estimate = estimate + increment
      BLAS.gemv!('N', 1.0, X, estimate, 0.0, z) # z = X * estimate
      clamp!(z, -20.0, 20.0)
      #
      # Compute the loglikelihood under the appropriate model.
      #
      if model == "logistic"
        z = exp.(-z)
        z = 1.0 ./ (1.0 .+ z)
        for i = 1:n
          if y[i] > 0.0
            obj = obj + log(z[i])
          else
            obj = obj + log(1.0 - z[i])
          end
        end
      elseif model == "poisson"
        for i = 1:n
          q = exp(z[i])
          obj = obj + y[i] * z[i] - q
        end  
      end
      #
      # Check for an increase in the loglikelihood.
      #
      if old_obj < obj
        break
      else
        estimate = estimate - increment # revert back to old estimate
        increment = 0.5 * increment
      end
    end
    #
    # Check for convergence.
    # 
    if iteration > 1 && abs(obj - old_obj) < epsilon * (abs(old_obj) + 1.0)
      return (estimate, obj)
    else
      old_obj = obj
    end
  end
  return (estimate, obj)
end # function regress

"""
When initilizing the model β, we fit a bivariate regression with the given covariate and 
the intercept. Fitting is done using scoring (newton) algorithm. The average of the 
intercept is used as the its own initial guess. 
"""
function initialize_beta!(
    v :: IHTVariable{T},
    y :: AbstractVector{T},
    x :: SnpArray,
    glm :: String,
) where {T <: Float}
    n, p = size(x)
    temp_matrix = ones(n, 2) #2 by p matrix of the intercept + the covariate
    intercept = 0.0
    for i in 1:p
        copyto!(@view(temp_matrix[:, 2]), view(x, :, i), center=true, scale=true)
        # all(temp_matrix[:, 2] .== 0) && continue
        (estimate, obj) = regress(temp_matrix, y, glm)
        intercept += estimate[1]
        v.b[i] = estimate[2]
    end
    v.c[1] = intercept / p
end
