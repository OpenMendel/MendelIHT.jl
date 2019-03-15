"""
    loglikelihood(y::AbstractVector, xb::AbstractVector, d::UnivariateDistribution)

This function calculates the loglikelihood of observing `y` given `μ` = g^{-1}(xb). 
"""
function loglikelihood(d::UnivariateDistribution, y::AbstractVector{T}, 
                       μ::AbstractVector{T}) where {T <: Float}
    logl = zero(T)
    ϕ = MendelIHT.deviance(d, y, μ) / length(y)
    @inbounds for i in eachindex(y)
        logl += loglik_obs(d, y[i], μ[i], 1, ϕ) #currently IHT don't support weights
    end
    return logl
end
# loglikelihood(::Normal, y::AbstractVector, xb::AbstractVector) = -0.5 * sum(abs2, y .- xb)
# loglikelihood(::Bernoulli, y::AbstractVector, xb::AbstractVector) = sum(y .* xb .- log.(1.0 .+ exp.(xb)))
# loglikelihood(::Binomial, y::AbstractVector, xb::AbstractVector, n::AbstractVector) = sum(y .* xb .- n .* log.(1.0 .+ exp.(xb)))
# loglikelihood(::Poisson, y::AbstractVector, xb::AbstractVector) = sum(y .* xb .- exp.(xb) .- lfactorial.(Int.(y)))
# loglikelihood(::Gamma, y::AbstractVector, xb::AbstractVector, ν::AbstractVector) = sum((y .* xb .+ log.(xb)) .* ν) + (ν .- 1) .* log.(y) .- ν .* (log.(1 ./ ν)) .- log.(SpecialFunctions.gamma.(ν))

function update_μ!(μ::AbstractVector{T}, xb::AbstractVector{T}, l::Link) where {T <: Float}
    @inbounds for i in eachindex(μ)
        μ[i] = linkinv(l, xb[i])
    end
end

"""
This function update the linear predictors `xb` with the new proposed b. We clamp the max
value of each entry to (-30, 30) because certain distributions (e.g. Poisson) have exponential
link functions, which causes overflow.
"""
function update_xb!(v::IHTVariable{T}, x::SnpArray, z::AbstractMatrix{T}) where {T <: Float}
    copyto!(v.xk, @view(x[:, v.idx]), center=true, scale=true)
    A_mul_B!(v.xb, v.zc, v.xk, z, view(v.b, v.idx), v.c)
    clamp!(v.xb, -30, 30)
    clamp!(v.zc, -30, 30)
end

"""
The deviance of a GLM can be evaluated as the sum of the squared deviance residuals. Calculation
of sqared deviance residuals is accomplished by `devresid` which is implemented in GLM.jl
"""
function deviance(d::UnivariateDistribution, y::AbstractVector{T}, μ::AbstractVector{T}) where {T <: Float}
    dev = zero(T)
    @inbounds for i in eachindex(y)
        dev += devresid(d, y[i], μ[i])
    end
    return dev
end

"""
    score = X^T * (y - g^{-1}(xb)) = [X^T * (y - g^{-1}(xb)) ; Z^T (y - g^{-1}(xb)))

This function calculates the score (gradient) for different glm models. X stores the snpmatrix
and Z stores intercept + other non-genetic covariates. The resulting score is stored in
v.df and v.df2, respectively. 

"""
function score!(v::IHTVariable{T}, x::SnpBitMatrix{T}, z::AbstractMatrix{T},
    y :: AbstractVector{T}) where {T <: Float}
    @inbounds for i in eachindex(y)
        v.r[i] = y[i] - v.μ[i]
    end
    At_mul_B!(v.df, v.df2, x, z, v.r, v.r)
end

"""
This function is taken from GLM.jl from : 

https://github.com/JuliaStats/GLM.jl/blob/956a64e7df79e80405867238781f24567bd40c78/src/glmtools.jl#L445

Putting it here because it was not exported.
"""
function loglik_obs end

loglik_obs(::Bernoulli, y, μ, wt, ϕ) = wt*logpdf(Bernoulli(μ), y)
loglik_obs(::Binomial, y, μ, wt, ϕ) = logpdf(Binomial(Int(wt), μ), Int(y*wt))
loglik_obs(::Gamma, y, μ, wt, ϕ) = wt*logpdf(Gamma(inv(ϕ), μ*ϕ), y)
loglik_obs(::InverseGaussian, y, μ, wt, ϕ) = wt*logpdf(InverseGaussian(μ, inv(ϕ)), y)
loglik_obs(::Normal, y, μ, wt, ϕ) = wt*logpdf(Normal(μ, sqrt(ϕ)), y)
loglik_obs(::Poisson, y, μ, wt, ϕ) = wt*logpdf(Poisson(μ), y)
# We use the following parameterization for the Negative Binomial distribution:
#    (Γ(θ+y) / (Γ(θ) * y!)) * μ^y * θ^θ / (μ+θ)^{θ+y}
# The parameterization of NegativeBinomial(r=θ, p) in Distributions.jl is
#    Γ(θ+y) / (y! * Γ(θ)) * p^θ(1-p)^y
# Hence, p = θ/(μ+θ)
loglik_obs(d::NegativeBinomial, y, μ, wt, ϕ) = wt*logpdf(NegativeBinomial(d.r, d.r/(μ+d.r)), y)

"""
This function computes the gradient step v.b = P_k(β + η∇f(β)) and updates idx and idc. 
"""
function _iht_gradstep(v::IHTVariable{T}, η::T, J::Int, k::Int, 
                       full_grad::Vector{T}) where {T <: Float}
    lb = length(v.b)
    lw = length(v.weight)
    lg = length(v.group)

    # take gradient step: b = b + μv, v = score
    BLAS.axpy!(η, v.df, v.b)  
    BLAS.axpy!(η, v.df2, v.c)

    # scale snp model if weight is supplied 
    lw == 0 ? full_grad[1:lb] .= v.b : full_grad[1:lb] .= v.b .* v.weight
    full_grad[lb+1:end] .= v.c

    # project to sparsity
    lg == 0 ? project_k!(full_grad, k) : project_group_sparse!(full_grad, v.group, J, k)

    #recompute current support
    v.b .= view(full_grad, 1:lb)
    v.c .= view(full_grad, lb+1:length(full_grad))
    v.idx .= v.b .!= 0
    v.idc .= v.c .!= 0
    
    # if more than J*k entries are selected, randomly choose J*k of them
    _choose!(v, J, k) 

    # make necessary resizing since grad step might include/exclude non-genetic covariates
    check_covariate_supp!(v) 
end

"""
When initializing the IHT algorithm, take largest elements of each group of df as nonzero
components of b. This function set v.idx = 1 for those indices. 

`J` is the maximum number of active groups, and `k` is the maximum number of predictors per
group. `full_grad` is some preallocated vector for efficiency. 
"""
function init_iht_indices!(v::IHTVariable{T}, xbm::SnpBitMatrix, z::AbstractMatrix{T},
                           y::AbstractVector{T}, d::UnivariateDistribution,l::Link, J::Int, 
                           k::Int, full_grad::AbstractVector{T}) where {T <: Float}

    # # update mean vector and use them to compute score (gradient)
    update_μ!(v.μ, v.xb + v.zc, l)
    score!(v, xbm, z, y)

    # find J*k largest entries and set everything else to 0. Choose randomly if more are selected
    a = sort([v.df; v.df2], rev=true)[k * J]
    v.idx .= v.df .>= a
    v.idc .= v.df2 .>= a
    _choose!(v, J, k) 

    # make necessary resizing when necessary
    check_covariate_supp!(v)
end

"""
if more than k entries are selected after projection, randomly select top k entries.
This can happen if entries of b are equal to each other.
"""
function _choose!(v::IHTVariable{T}, J::Int, k::Int) where {T <: Float}
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
included/excluded, we must resize xk and gk
"""
function check_covariate_supp!(v::IHTVariable{T}) where {T <: Float}
    if sum(v.idx) != size(v.xk, 2)
        v.xk = zeros(T, size(v.xk, 1), sum(v.idx))
        v.gk = zeros(T, sum(v.idx))
    end
end

# """
# this function calculates the omega (here a / b) used for determining backtracking
# """
# function _iht_omega(v::IHTVariable{T}) where {T <: Float}
#     a = sqeuclidean(v.b, v.b0::Vector{T}) + sqeuclidean(v.c, v.c0::Vector{T}) :: T
#     b = sqeuclidean(v.xb, v.xb0::Vector{T}) + sqeuclidean(v.zc, v.zc0::Vector{T}) :: T
#     return a, b
# end

# """
# this function for determining whether or not to backtrack for normal least squares. True = backtrack
# """
# function _normal_backtrack(v::IHTVariable{T}, ot::T, ob::T, η::T, η_step::Int, nstep::Int
# ) where {T <: Float}
#     η*ob > 0.99*ot               &&
#     sum(v.idx) != 0              &&
#     sum(xor.(v.idx,v.idx0)) != 0 &&
#     η_step < nstep
# end

"""
This function returns true if backtracking condition is met. Currently, backtracking condition
includes either one of the following:
    1. New loglikelihood is smaller than the old one
    2. Current backtrack exceeds maximum allowed backtracking (default = 3)

Note, for Posison, we require the model coefficients to be "small" to prevent 
loglikelihood blowing up in first few iteration. This is accomplished by clamping
xb values to be in (-30, 30)
"""
function _iht_backtrack_(logl::T, prev_logl::T, η_step::Int64, nstep::Int64) where {T <: Float}
    prev_logl > logl && η_step < nstep 
end

"""
Compute the standard deviation of a SnpArray in place. Note this function assumes all SNPs 
are not missing. Otherwise, the inner loop should only add if data not missing.
"""
function std_reciprocal(x::SnpBitMatrix, mean_vec::Vector{T}) where {T <: Float}
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

"""
    project_k!(x, k)
This function projects a vector `x` onto the set S_k = { y in R^p : || y ||_0 <= k }.
It does so by first finding the pivot `a` of the `k` largest components of `x` in magnitude.
`project_k!` then applies `x[abs.(x) < a] = 0 
Arguments:
- `b` is the vector to project.
- `k` is the number of components of `b` to preserve.
"""
function project_k!(x::AbstractVector{T}, k::Int64) where {T <: Float}
    a = abs(partialsort(x, k, by=abs, rev=true))
    for i in eachindex(x)
        abs(x[i]) < a && (x[i] = zero(T))
    end
end

""" 
Projects the vector y = [y1; y2] onto the set with at most J active groups and at most
k active predictors per group. The variable group encodes group membership. Currently
assumes there are no unknown or overlaping group membership.

TODO: check if sortperm can be replaced by something that doesn't sort the whole array
"""
function project_group_sparse!(y::AbstractVector{T}, group::AbstractVector{Int64},
    J::Int64, k::Int64) where {T <: Float}
    groups = maximum(group)          # number of groups
    group_count = zeros(Int, groups) # counts number of predictors in each group
    group_norm = zeros(groups)       # l2 norm of each group
    perm = zeros(Int64, length(y))   # vector holding the permuation vector after sorting
    sortperm!(perm, y, by = abs, rev = true)

    #calculate the magnitude of each group, where only top predictors contribute
    @inbounds for i in eachindex(y)
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
    @inbounds for i in eachindex(y)
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
Calculates the Prior Weighting for IHT. Returns a weight array `my_snpweights`.
"""
function calculate_snp_weights(x::SnpArray)
    p = maf(x)
    p .= 1 ./ (2 .* sqrt.(p .* (1 .- p)))
    clamp!(p, 1.0, 2.0)
    return p
end

"""
Function that saves `b`, `xb`, `idx`, `idc`, `c`, and `zc` after each iteration. 
"""
function save_prev!(v::IHTVariable{T}) where {T <: Float}
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
function iht_stepsize(v::IHTVariable{T}, z::AbstractMatrix{T}, 
                      d::UnivariateDistribution) where {T <: Float}
    
    # first store relevant components of gradient
    v.gk .= view(v.df, v.idx)
    A_mul_B!(v.xgk, v.zdf2, v.xk, view(z, :, v.idc), v.gk, view(v.df2, v.idc))
    
    # now compute and return step size. Note non-genetic covariates are separated from x
    denom = Transpose(v.xgk + v.zdf2) * Diagonal(glmvar.(d, v.μ)) * (v.xgk + v.zdf2)
    numer = sum(abs2, v.gk) + sum(abs2, @view(v.df2[v.idc]))
    return (numer / denom) :: T
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
This is a wrapper linear algebra function that computes [C1 ; C2] = [A1 ; A2] * [B1 ; B2] 
where A1 is a snpmatrix and A2 is a dense Matrix{Float}. Used for cleaner code. 

Here we are separating the computation because A1 is stored in compressed form while A2 is 
uncompressed (float64) matrix. This means that they cannot be stored in the same data 
structure. 
"""
function A_mul_B!(C1::AbstractVector{T}, C2::AbstractVector{T}, A1::SnpBitMatrix{T},
        A2::AbstractMatrix{T}, B1::AbstractVector{T}, B2::AbstractVector{T}) where {T <: Float}
    SnpArrays.mul!(C1, A1, B1)
    LinearAlgebra.mul!(C2, A2, B2)
end

function A_mul_B!(C1::AbstractVector{T}, C2::AbstractVector{T}, A1::AbstractMatrix{T},
        A2::AbstractMatrix{T}, B1::AbstractVector{T}, B2::AbstractVector{T}) where {T <: Float}
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
function At_mul_B!(C1::AbstractVector{T}, C2::AbstractVector{T}, A1::SnpBitMatrix{T},
        A2::AbstractMatrix{T}, B1::AbstractVector{T}, B2::AbstractVector{T}) where {T <: Float}
    SnpArrays.mul!(C1, Transpose(A1), B1)
    LinearAlgebra.mul!(C2, Transpose(A2), B2)
end

function At_mul_B!(C1::AbstractVector{T}, C2::AbstractVector{T}, A1::AbstractMatrix{T},
        A2::AbstractMatrix{T}, B1::AbstractVector{T}, B2::AbstractVector{T}) where {T <: Float}
    LinearAlgebra.mul!(C1, A1', B1)
    LinearAlgebra.mul!(C2, A2', B2)
end

"""
This function will create a random SnpArray in the current directory without missing value, 
where each SNP has at least 5 minor alleles. 

n = number of samples
p = number of SNPs
s = name of the simulated SnpArray. 
"""
function simulate_random_snparray(n::Int64, p::Int64, s::String)
    #first simulate a random {0, 1, 2} matrix with each SNP drawn from Binomial(2, r[i])
    A1 = BitArray(undef, n, p) 
    A2 = BitArray(undef, n, p) 
    mafs = zeros(Float64, p)
    for j in 1:p
        minor_alleles = 0
        maf = 0
        while minor_alleles <= 5
            maf = 0.5rand()
            for i in 1:n
                A1[i, j] = rand(Bernoulli(maf))
                A2[i, j] = rand(Bernoulli(maf))
            end
            minor_alleles = sum(view(A1, :, j)) + sum(view(A2, :, j))
        end
        mafs[j] = maf
    end

    #fill the SnpArray with the corresponding x_tmp entry
    return _make_snparray(A1, A2, s), mafs
end

"""
Make a SnpArray from 2 BitArrays.
"""
function _make_snparray(A1::BitArray, A2::BitArray, s::String)
    n, p = size(A1)
    x = SnpArray(s, n, p)
    for i in 1:(n*p)
        c = A1[i] + A2[i]
        if c == 0
            x[i] = 0x00
        elseif c == 1
            x[i] = 0x02
        elseif c == 2
            x[i] = 0x03
        else
            throw(error("matrix shouldn't have missing values!"))
        end
    end
    return x
end

"""
This function simulates a random response vector `y` based on provided x, β, distirbution,
and link function. `k` denotes the true number of predictors. When the distribution is from 
Poisson, Gamma, or Negative Binomial, then the true model is drawn from N(0, 0.3) to roughly 
ensure the mean of response `y` doesn't become too large. 

Note: for negative binomial and gamma, the link function must be LogLink. For Bernoulli,
the probit link seems to work better than logitlink.

`nn` is an optional input argument needed for gamma and negative binomial simulations. 
For Negative binomial, it is the number of success until stopping. For gamma distribution, 
it is the shape parameter. 
"""
function simulate_random_response(x::SnpArray, xbm::SnpBitMatrix, β::AbstractVector{T}, 
                                  k::Int, d::UnivariateDistribution, l::Link; nn = 10
                                  ) where {T <: Float}
    
    n, p = size(x)
    if (typeof(d) <: NegativeBinomial) || (typeof(d) <: Gamma)
        l == LogLink() || throw(ArgumentError("Distribution $d must use LogLink!"))
    end

    #simulate a random model β from a normal distribution
    true_b = zeros(p)
    if d == (Poisson || Gamma || NegativeBinomial)
        true_b[1:k] = rand(Normal(0, 0.3), k)
    else
        true_b[1:k] = randn(k)
    end
    shuffle!(true_b)
    correct_position = findall(x -> x != 0, true_b)

    #simulate phenotypes (e.g. vector y)
    if (typeof(d) <: Normal) || (typeof(d) <: Poisson) || (typeof(d) <: Bernoulli) || (typeof(d) <: Poisson)
        prob = linkinv.(l, xbm * true_b)
        y = [rand(d(i)) for i in prob]
    elseif (typeof(d) <: NegativeBinomial)
        μ = linkinv.(l, xbm * true_b)
        prob = 1 ./ (1 .+ μ ./ nn)
        y = [rand(d(nn, i)) for i in prob] #number of failtures before nn success occurs
    elseif (typeof(d) <: Gamma)
        μ = linkinv.(l, xbm * true_b)
        β = 1 ./ μ # here β is the rate parameter for gamma distribution
        y = [rand(d(nn, i)) for i in β] #nn is used as the shape parameter for gamma
    end
    y = Float64.(y)

    return y, true_b, correct_position
end

"""
When initilizing the model β, for each covariate we fit a bivariate regression with 
itself and the intercept. Fitting is done using scoring (newton) algorithm in GLM.jl. 
The average of the intercept over all fits is used as the its initial guess. 

This function is quite slow and not memory efficient currently. 
"""
function initialize_beta!(v::IHTVariable{T}, y::AbstractVector{T}, x::SnpArray,
                          d::UnivariateDistribution, l::Link) where {T <: Float}
    n, p = size(x)
    temp_matrix = ones(n, 2)           # n by 2 matrix of the intercept and 1 single covariate
    temp_glm = initialize_glm_object() # preallocating in a dumb ways

    intercept = 0.0
    for i in 1:p
        copyto!(@view(temp_matrix[:, 2]), @view(x[:, i]), center=true, scale=true)
        temp_glm = fit(GeneralizedLinearModel, temp_matrix, y, d, l)
        intercept += temp_glm.pp.beta0[1]
        v.b[i] = temp_glm.pp.beta0[2]
    end
    v.c[1] = intercept / p
end

"""
This function initializes 1 instance of a GeneralizedLinearModel(G<:GlmResp, L<:LinPred, Bool). 
"""
function initialize_glm_object()
    d = Bernoulli
    l = canonicallink(d())
    x = rand(100, 2)
    y = rand(0:1, 100)
    return fit(GeneralizedLinearModel, x, y, d(), l)
end
