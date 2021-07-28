"""
    loglikelihood(v::IHTVariable{T, M})

Calculates the loglikelihood of observing `y` given mean `μ = E(y) = g^{-1}(xβ)`
and some univariate distribution `d`. 

Note that loglikelihood is the sum of the logpdfs for each observation. 
"""
function loglikelihood(v::IHTVariable{T, M}) where {T <: Float, M}
    d = v.d 
    y = v.y
    μ = v.μ
    wts = v.cv_wts
    logl = zero(T)
    ϕ = MendelIHT.deviance(v) / length(y) # variance in the case of normal
    @inbounds for i in eachindex(y)
        logl += loglik_obs(d, y[i], μ[i], wts[i], ϕ)
    end
    return logl
end

"""
This function is taken from GLM.jl from: 
https://github.com/JuliaStats/GLM.jl/blob/956a64e7df79e80405867238781f24567bd40c78/src/glmtools.jl#L445

`wt`: in GLM.jl, this is prior frequency (a.k.a. case) weights for observations.
We use this for cross validation weighting: if sample `i` is being fitted,
wt[i] = 1. Otherwise wt[i] = 0.
"""
function loglik_obs end

loglik_obs(::Bernoulli, y, μ, wt, ϕ) = wt*logpdf(Bernoulli(μ), y)
loglik_obs(::Binomial, y, μ, wt, ϕ) = logpdf(Binomial(Int(wt), μ), Int(y*wt))
loglik_obs(::Gamma, y, μ, wt, ϕ) = wt*logpdf(Gamma(inv(ϕ), μ*ϕ), y)
loglik_obs(::InverseGaussian, y, μ, wt, ϕ) = wt*logpdf(InverseGaussian(μ, inv(ϕ)), y)
loglik_obs(::Normal, y, μ, wt, ϕ) = wt*logpdf(Normal(μ, sqrt(ϕ)), y)
loglik_obs(::Poisson, y, μ, wt, ϕ) = wt*logpdf(Poisson(μ), y)
# We use the following parameterization for the Negative Binomial distribution:
#    (Γ(r+y) / (Γ(r) * y!)) * μ^y * r^r / (μ+r)^{r+y}
# The parameterization of NegativeBinomial(r=r, p) in Distributions.jl is
#    Γ(r+y) / (y! * Γ(r)) * p^r(1-p)^y
# Hence, p = r/(μ+r)
loglik_obs(d::NegativeBinomial, y, μ, wt, ϕ) = wt*logpdf(NegativeBinomial(d.r, d.r/(μ+d.r)), y)

"""
    deviance(d, y, μ, wts)

Calculates the sum of the squared deviance residuals (e.g. (y - μ)^2 for Gaussian case) 
Each individual sqared deviance residual is evaluated using `devresid`
which is implemented in GLM.jl
"""
function deviance(d::UnivariateDistribution, y::AbstractVector{T},
    μ::AbstractVector{T}, wts::AbstractVector{T}) where {T <: Float}
    dev = zero(T)
    @inbounds for i in eachindex(y)
        dev += wts[i] * devresid(d, y[i], μ[i])
    end
    return dev
end
deviance(v::IHTVariable{T, M}) where {T <: Float, M} = 
    MendelIHT.deviance(v.d, v.y, v.μ, v.cv_wts)

"""
    update_μ!(μ, xb, l)

Update the mean (μ) using the linear predictor `xb` with link `l`.
"""
function update_μ!(μ::AbstractVecOrMat{T}, xb::AbstractVecOrMat{T}, l::Link) where {T <: Float}
    @inbounds for i in eachindex(μ)
        μ[i] = linkinv(l, xb[i])
    end
end

function update_μ!(v::IHTVariable{T, M}) where {T <: Float, M}
    μ = v.μ
    xb = v.xb
    zc = v.zc
    l = v.l
    @inbounds for i in eachindex(μ)
        μ[i] = linkinv(l, xb[i] + zc[i]) #genetic + nongenetic contributions
    end
end

"""
    update_xb!(v::IHTVariable{T, M})

Updates the linear predictors `xb` and `zc` with the new proposed `b` and `c`.
`b` is sparse but `c` (beta for non-genetic covariates) is dense.

We clamp the max value of each entry to (-20, 20) because certain distributions
(e.g. Poisson) have exponential link functions, which causes overflow.
"""
function update_xb!(v::IHTVariable{T, M}) where {T <: Float, M}
    copyto!(v.xk, @view(v.x[:, v.idx]))
    mul!(v.xb, v.xk, view(v.b, v.idx))
    mul!(v.zc, v.z, v.c)
    if !(typeof(v.d) <: Normal)
        clamp!(v.xb, -20, 20)
        clamp!(v.zc, -20, 20)
    end
end

"""
    score!(v::IHTVariable{T})

Calculates the score (gradient) `X^T * W * (y - g(x^T b))` for different GLMs. 
W is a diagonal matrix where `w[i, i] = dμ/dη / var(μ)` (see documentation)
"""
function score!(v::IHTVariable{T, M}) where {T <: Float, M}
    d, l, x, z, y, cv_wts = v.d, v.l, v.x, v.z, v.y, v.cv_wts
    @inbounds for i in eachindex(y)
        η = v.xb[i] + v.zc[i]
        w = mueta(l, η) / glmvar(d, v.μ[i])
        v.r[i] = w * (y[i] - v.μ[i]) * cv_wts[i] # cv_wts handles sample masking for cross validation
    end
    mul!(v.df, Transpose(x), v.r)
    mul!(v.df2, Transpose(z), v.r)
end

"""
Wrapper function to decide whether to use Newton or MM algorithm for estimating 
the nuisance paramter of negative binomial regression. 
"""
function mle_for_r(v::IHTVariable{T, M}) where {T <: Float, M}
    method = v.est_r
    if method == :MM
        return update_r_MM(v)
    elseif method == :Newton
        return update_r_newton(v)
    else
        throw(ArgumentError("Only support method is Newton or MM, but got $method"))
    end

    return nothing
end

"""
Performs maximum loglikelihood estimation of the nuisance paramter for negative 
binomial model using MM's algorithm. 
"""
function update_r_MM(v::IHTVariable{T, M}) where {T <: Float, M}
    y = v.y
    μ = v.μ
    r = v.d.r # estimated r in previous iteration
    num = zero(T)
    den = zero(T)
    for i in eachindex(y)
        for j = 0:y[i] - 1
            num = num + (r /(r + j))  # numerator for r
        end
        p = r / (r + μ[i])
        den = den + log(p)  # denominator for r
    end

    return NegativeBinomial(-num / den, T(0.5))
end

"""
Performs maximum loglikelihood estimation of the nuisance paramter for negative 
binomial model using Newton's algorithm. Will run a maximum of `maxIter` and
convergence is defaulted to `convTol`.
"""
function update_r_newton(v::IHTVariable{T, M};
    maxIter=100, convTol=T(1.e-6)) where {T <: Float, M}
    y = v.y
    μ = v.μ
    r = v.d.r # estimated r in previous iteration

    function first_derivative(r::T)
        tmp(yi, μi) = -(yi+r)/(μi+r) - log(μi+r) + 1 + log(r) + digamma(r+yi) - digamma(r)
        return sum(tmp(yi, μi) for (yi, μi) in zip(y, μ))
    end

    function second_derivative(r::T)
        tmp(yi, μi) = (yi+r)/(μi+r)^2 - 2/(μi+r) + 1/r + trigamma(r+yi) - trigamma(r)
        return sum(tmp(yi, μi) for (yi, μi) in zip(y, μ))
    end

    function negbin_loglikelihood(r::T)
        v.d = NegativeBinomial(r, T(0.5))
        return MendelIHT.loglikelihood(v)
    end

    function newton_increment(r::T)
        # use gradient descent if hessian not positive definite
        dx  = first_derivative(r)
        dx2 = second_derivative(r)
        if dx2 < 0
            increment = first_derivative(r) / second_derivative(r)
        else 
            increment = first_derivative(r)
        end
        return increment
    end

    new_r    = one(T)
    stepsize = one(T)
    for i in 1:maxIter

        # run 1 iteration of Newton's algorithm
        increment = newton_increment(r)
        new_r = r - stepsize * increment

        # linesearch
        old_logl = negbin_loglikelihood(r)
        for j in 1:20
            if new_r <= 0
                stepsize = stepsize / 2
                new_r = r - stepsize * increment
            else 
                new_logl = negbin_loglikelihood(new_r)
                if old_logl >= new_logl
                    stepsize = stepsize / 2
                    new_r = r - stepsize * increment
                else
                    break
                end
            end
        end

        #check convergence
        if abs(r - new_r) <= convTol
            return NegativeBinomial(new_r, T(0.5))
        else
            r = new_r
        end
    end

    return NegativeBinomial(r, T(0.5))
end

"""
This function computes the gradient step v.b = P_k(β + η∇f(β)) and updates idx and idc. 
"""
function _iht_gradstep!(v::IHTVariable{T, M}, η::T) where {T <: Float, M}
    lg = length(v.group)
    J = v.J
    k = length(v.ks) > 0 ? v.ks : v.k

    # take gradient step: b = b + ηv, v = score
    BLAS.axpy!(η, v.df, v.b)  
    BLAS.axpy!(η, v.df2, v.c)

    # project to sparsity, scaling model by weight vector, if supplied
    if lg == 0
        vectorize!(v.full_b, v.b, v.c, v.weight, v.zkeep)
        project_k!(v.full_b, k + v.zkeepn)
        unvectorize!(v.full_b, v.b, v.c, v.weight, v.zkeep)
    else
        # TODO: enable model selection for non-genetic covariates on group projection
        project_group_sparse!(v.b, v.group, J, k)
    end

    #recombute support
    v.idx .= v.b .!= 0
    v.idc .= v.c .!= 0

    # if more than J*k entries are selected, randomly choose J*k of them
    typeof(k) == Int && _choose!(v) 

    # make necessary resizing since grad step might include/exclude non-genetic covariates
    check_covariate_supp!(v) 
end

"""
    vectorize!(a::AbstractVector, b::AbstractVector, c::AbstractVector, ckeep::BitVector)

Without allocations, copies `b` into `a` and the copies `c` into
remaining parts of `a`. 

`ckeep` tracks entries of `c` that will not be projected. That is, entries of `a`
corresponding to entries in `c[i]` will be filled with `Inf` if `ckeep[i] = true`.
"""
function vectorize!(a::AbstractVector, b::AbstractVector, c::AbstractVector,
    weight::AbstractVector, ckeep::BitVector)
    lb = length(b)
    lw = length(weight)
    la = length(a)

    # scale model by weight vector, if supplied 
    if lw == 0
        copyto!(@view(a[1:lb]), b)
        copyto!(@view(a[lb+1:la]), c)
    else
        @inbounds for i in 1:lb
            a[i] = b[i] * weight[i]
        end
        ii = 1
        @inbounds for i in lb+1:la
            a[i] = c[ii] * weight[i]
            ii += 1
        end
    end

    # don't project certain non-genetic covariates
    a_view = @view(a[lb+1:la])
    @view(a_view[ckeep]) .= typemax(eltype(a))
end

"""
    unvectorize!(a::AbstractVector, B::AbstractMatrix, C::AbstractMatrix, Ckeep::BitVector)

Without allocations, copies the first `length(b)` part of `a` into `b` 
and the remaining parts of `a` into `c`.

`ckeep` tracks entries of `c` that will not be projected. That is, entries in `c[i]`
will not be touched if `ckeep[i] = true`.
"""
function unvectorize!(a::AbstractVector, b::AbstractVector, c::AbstractVector,
    weight::AbstractVector, ckeep::BitVector)
    lb = length(b)
    lw = length(weight)
    la = length(a)

    # scale model by weight vector, if supplied 
    if lw == 0
        copyto!(b, @view(a[1:lb]))
        ii = 1
        @inbounds for i in lb+1:la
            if !ckeep[ii]
                c[ii] = a[i] 
            end
            ii += 1
        end
    else
        @inbounds for i in 1:lb
            b[i] = a[i] / weight[i]
        end
        ii = 1
        @inbounds for i in lb+1:la
            if !ckeep[ii]
                c[ii] = a[i] / weight[i]
            end
            ii += 1
        end
    end
end

"""
When initializing the IHT algorithm, take largest elements in magnitude of each
group of the score as nonzero components of b. This function set v.idx = 1 for
those indices. If `init_beta=true`, then beta values will be initialized to
their univariate values (see [`initialize_beta`](@ref)), in which case we will simply
choose top `k` entries

`J` is the maximum number of active groups, and `k` is the maximum number of
predictors per group. 
"""
function init_iht_indices!(v::IHTVariable, init_beta::Bool, cv_idx::BitVector)
    fill!(v.b, 0)
    fill!(v.b0, 0)
    fill!(v.best_b, 0)
    fill!(v.xb, 0)
    fill!(v.xk, 0)
    fill!(v.gk, 0)
    fill!(v.xgk, 0)
    fill!(v.idx, false)
    fill!(v.idx0, false)
    copyto!(v.idc, v.zkeep)
    copyto!(v.idc0, v.zkeep)
    fill!(v.r, 0)
    fill!(v.df, 0)
    fill!(v.df2, 0)
    fill!(v.c, 0)
    fill!(v.best_c, 0)
    fill!(v.c0, 0)
    fill!(v.zc, 0)
    fill!(v.zdf2, 0)
    fill!(v.μ, 0)
    fill!(v.cv_wts, 0)
    fill!(v.full_b, 0)
    v.cv_wts[cv_idx] .= 1

    init_beta && !(typeof(v.d) <: Normal) && 
        throw(ArgumentError("Intializing beta values only work for Gaussian phenotypes! Sorry!"))

    # find the intercept by Newton's method
    ybar = zero(eltype(v.y))
    @inbounds @simd for i in eachindex(v.y)
        ybar += v.y[i] * v.cv_wts[i]
    end
    ybar /= count(!iszero, v.cv_wts)
    for iteration = 1:20 
        g1 = linkinv(v.l, v.c[1])
        g2 = mueta(v.l, v.c[1])
        v.c[1] = v.c[1] - clamp((g1 - ybar) / g2, -1.0, 1.0)
        abs(g1 - ybar) < 1e-10 && break
    end
    mul!(v.zc, v.z, v.c)

    # update mean vector and use them to compute score (gradient)
    update_μ!(v)
    score!(v)

    if init_beta
        initialize_beta!(v, cv_idx)
        project_k!(v)
    else
        # first `k` non-zero entries are chosen based on largest gradient
        vectorize!(v.full_b, v.df, v.df2, v.weight, v.zkeep)
        if length(v.ks) == 0 # no group projection
            project_k!(v.full_b, v.k + v.zkeepn) # project k + number of nongentic covariates to keep
            unvectorize!(v.full_b, v.df, v.df2, v.weight, v.zkeep)
            v.idx .= v.df .!= 0
            v.idc .= v.zkeep

            # Choose randomly if more are selected
            _choose!(v) 
        else 
            project_group_sparse!(v.df, v.group, v.J, v.ks)
            v.idx .= v.b .!= 0
            fill!(v.idc, true)
        end
    end

    # make necessary resizing when necessary
    check_covariate_supp!(v)

    # store relevant components of x for first iteration
    copyto!(v.xk, @view(v.x[:, v.idx])) 
end

"""
if more than J*k entries are selected after projection, randomly select top J*k entries.
This can happen if entries of b are equal to each other.
"""
function _choose!(v::IHTVariable{T}) where {T <: Float}
    sparsity = v.k + v.zkeepn
    groups = (v.J == 0 ? 1 : v.J)

    nonzero = sum(v.idx) + sum(v.idc) - v.zkeepn
    if nonzero > groups * sparsity
        z = zero(eltype(v.b))
        non_zero_idx = findall(!iszero, v.idx)
        excess = nonzero - groups * sparsity
        for pos in sample(non_zero_idx, excess, replace=false)
            v.b[pos]   = z
            v.idx[pos] = false
        end
    end
end

"""
In `_init_iht_indices` and `_iht_gradstep!`, if non-genetic cov got 
included/excluded, we must resize `xk` and `gk`.

TODO: Use ElasticArrays.jl
"""
function check_covariate_supp!(v::IHTVariable{T}) where {T <: Float}
    nzidx = sum(v.idx)
    if nzidx != size(v.xk, 2)
        v.xk = zeros(T, size(v.xk, 1), nzidx)
        v.gk = zeros(T, nzidx)
    end
    @inbounds for i in eachindex(v.zkeep)
        v.zkeep[i] && !v.idc[i] && error("A non-genetic covariate was accidentally set to 0! Shouldn't happen!")
    end
end

"""
    _iht_backtrack_(logl::T, prev_logl::T, η_step::Int64, nstep::Int64)

Returns true if one of the following conditions is met:
1. New loglikelihood is smaller than the old one
2. Current backtrack (`η_step`) exceeds maximum allowed backtracking (`nstep`, default = 3)
"""
function _iht_backtrack_(logl::T, prev_logl::T, η_step::Int64, nstep::Int64) where {T <: Float}
    (prev_logl > logl) && (η_step < nstep)
end

"""
    standardize!(z::AbstractVecOrMat)

Standardizes each column of `z` to mean 0 and variance 1. Make sure you 
do not standardize the intercept. 
"""
@inline function standardize!(z::AbstractVecOrMat)
    n, q = size(z)
    μ = _mean(z)
    σ = _std(z, μ)

    @inbounds for j in 1:q
        @simd for i in 1:n
            z[i, j] = (z[i, j] - μ[j]) * σ[j]
        end
    end
end

@inline function _mean(z)
    n, q = size(z)
    μ = zeros(q)
    @inbounds for j in 1:q
        tmp = 0.0
        @simd for i in 1:n
            tmp += z[i, j]
        end
        μ[j] = tmp / n
    end
    return μ
end

function _std(z, μ)
    n, q = size(z)
    σ = zeros(q)

    @inbounds for j in 1:q
        @simd for i in 1:n
            σ[j] += (z[i, j] - μ[j])^2
        end
        σ[j] = 1.0 / sqrt(σ[j] / (n - 1))
    end
    return σ
end

"""
    project_k!(x::AbstractVector, k::Integer)

Sets all but the largest `k` entries of `x` to 0. 

# Examples:
```julia-repl
using MendelIHT
x = [1.0; 2.0; 3.0]
project_k!(x, 2) # keep 2 largest entry
julia> x
3-element Array{Float64,1}:
 0.0
 2.0
 3.0
```

# Arguments:
- `x`: the vector to project.
- `k`: the number of components of `x` to preserve.
"""
function project_k!(x::AbstractVector{T}, k::Int64) where {T <: Float}
    k < 0 && throw(DomainError("Attempted to project to sparsity level $k"))
    a = abs(partialsort(x, k, by=abs, rev=true))
    @inbounds for i in eachindex(x)
        abs(x[i]) < a && (x[i] = zero(T))
    end
end

function project_k!(v::IHTVariable)
    v.k < 0 && throw(DomainError("Attempted to project to sparsity level $(v.k)"))

    # copy genetic and non-genetic effects to full_grad, project, and copy back
    vectorize!(v.full_b, v.b, v.c, v.weight, v.zkeep)
    project_k!(v.full_b, v.k + v.zkeepn)
    unvectorize!(v.full_b, v.b, v.c, v.weight, v.zkeep)

    # update support
    v.idx .= v.b .!= 0
    v.idc .= v.c .!= 0
    check_covariate_supp!(v)
end

""" 
    project_group_sparse!(y::AbstractVector, group::AbstractVector, J::Integer, k<:Real)

When `k` is an integer, projects the vector `y` onto the set with at most `J` active groups 
and at most `k` active predictors per group. To have variable group sparsity level, input `k`
as a vector of integers. We will preserve `k[1]` elements for group 1, `k[2]` predictors for 
group 2...etc. This function assumes there are no unknown or overlaping group membership.

Note: In the `group` vector, the first group must be 1, and the second group must be 2...etc. 

# Examples
```julia-repl
using MendelIHT
J, k, n = 2, 3, 20
y = collect(1.0:20.0)
y_copy = copy(y)
group = rand(1:5, n)
project_group_sparse!(y, group, J, k)
for i = 1:length(y)
    println(i,"  ",group[i],"  ",y[i],"  ",y_copy[i])
end

J, k, n = 2, 0.9, 20
y = collect(1.0:20.0)
y_copy = copy(y)
group = rand(1:5, n)
project_group_sparse!(y, group, J, k)
for i = 1:length(y)
    println(i,"  ",group[i],"  ",y[i],"  ",y_copy[i])
end
```

# Arguments 
- `y`: The vector to project
- `group`: Vector encoding group membership
- `J`: Max number of non-zero group
- `k`: Maximum predictors per group. Can be a positive integer or a vector of integers. 
"""
function project_group_sparse!(y::AbstractVector{T}, group::AbstractVector{Int64},
    J::Int64, k::Int64) where {T <: Float}
    groups = maximum(group)          # number of groups
    group_count = zeros(Int, groups) # counts number of predictors in each group
    group_norm = zeros(groups)       # l2 norm of each group
    perm = zeros(Int64, length(y))   # vector holding the permuation vector after sorting
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

function project_group_sparse!(y::AbstractVector{T}, group::AbstractVector{Int64},
    J::Int64, k::Vector{Int}) where {T <: Float}
    groups = maximum(group)          # number of groups
    group_count = zeros(Int, groups) # counts number of predictors in each group
    group_norm = zeros(groups)       # l2 norm of each group
    perm = zeros(Int64, length(y))   # vector holding the permuation vector after sorting
    sortperm!(perm, y, by = abs, rev = true)

    #calculate the magnitude of each group, where only top predictors contribute
    for i in eachindex(y)
        j = perm[i]
        n = group[j]
        if group_count[n] < k[n]
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
        if (group_rank[n] > J) || (group_count[n] > k[n])
            y[j] = 0.0
        else
            group_count[n] = group_count[n] + 1
        end
    end
end

"""
    maf_weights(x::SnpArray; max_weight::T = Inf)

Calculates the prior weight based on minor allele frequencies. 

Returns an array of weights where `w[i] = 1 / (2 * sqrt(p[i] (1 - p[i]))) ∈ (1, ∞).`
Here `p` is the minor allele frequency computed by `maf()` in SnpArrays. 

- `x`: A SnpArray 
- `max_weight`: Maximum weight for any predictor. Defaults to `Inf`. 
"""
function maf_weights(x::SnpArray; max_weight::T = Inf) where {T <: Float}
    p = maf(x)
    p .= 1 ./ (2 .* sqrt.(p .* (1 .- p)))
    clamp!(p, 1.0, max_weight)
    return p
end

"""
Function that saves `b`, `xb`, `idx`, `idc`, `c`, and `zc` after each iteration. 
"""
function save_prev!(v::IHTVariable{T}, cur_logl::T, best_logl::T) where {T <: Float}
    copyto!(v.b0, v.b)     # b0 = b
    copyto!(v.idx0, v.idx) # idx0 = idx
    copyto!(v.idc0, v.idc) # idc0 = idc
    copyto!(v.c0, v.c)     # c0 = c
    if cur_logl > best_logl
        copyto!(v.best_b, v.b) # best_b = b
        copyto!(v.best_c, v.c) # best_c = c
    end
    return max(cur_logl, best_logl)
end

"""
Computes the best step size η = v'v / v'Jv = v'v / v'X'WXv

Here v is the score and J is the expected information matrix, which is 
computed by J = g'(xb) / var(μ), assuming dispersion is 1. Note 

Note: cross validation weights are needed in W.
"""
function iht_stepsize!(v::IHTVariable{T, M}) where {T <: Float, M}
    z = v.z # non genetic covariates
    d = v.d # distribution
    l = v.l # link function

    # first compute Xv using relevant components of gradient
    copyto!(v.gk, view(v.df, v.idx)) 
    mul!(v.xgk, v.xk, v.gk) 
    mul!(v.zdf2, view(z, :, v.idc), view(v.df2, v.idc))
    v.xgk .+= v.zdf2 # xgk = Xv

    # Compute sqrt(W); use zdf2 as storage
    @inbounds @simd for i in eachindex(v.zdf2)
        v.zdf2[i] = sqrt(abs2(mueta(l, v.xb[i] + v.zc[i])) / glmvar(d, v.μ[i])) * v.cv_wts[i]
    end
    v.xgk .*= v.zdf2 # xgk = sqrt(W)Xv

    # now compute and return step size. Note non-genetic covariates are separated from x
    numer = sum(abs2, v.gk) + sum(abs2, @view(v.df2[v.idc]))
    denom = dot(v.xgk, v.xgk)
    η = numer / denom

    # for bad boundary cases (sometimes, k = 1 in cross validation generates weird η)
    isinf(η) && (η = T(1e-8))
    isnan(η) && (η = T(1e-8))

    return η :: T
end

"""
    initialize_beta!(v, cv_wts)

Initialze beta to univariate regression values. That is, `β[i]` is set to the estimated
beta with `y` as response, and `x[:, i]` with an intercept term as covariate.

TODO: this function assumes quantitative (Gaussian) phenotypes. Make it work for other distributions
"""
function initialize_beta!(
    v::IHTVariable,
    cv_wts::BitVector # cross validation weights; 1 = sample is present, 0 = not present
    )
    y, x, z, β, c, T = v.y, v.x, v.z, v.b, v.c, eltype(v.b)
    xtx_store = [zeros(T, 2, 2) for _ in 1:Threads.nthreads()]
    xty_store = [zeros(T, 2) for _ in 1:Threads.nthreads()]
    xstore = [zeros(T, sum(cv_wts)) for _ in 1:Threads.nthreads()]
    ystore = y[cv_wts]
    # genetic covariates
    Threads.@threads for i in 1:nsnps(v)
        id = Threads.threadid()
        copyto!(xstore[id], @view(x[cv_wts, i]))
        linreg!(xstore[id], ystore, xtx_store[id], xty_store[id])
        β[i] = xty_store[id][2]
    end
    # non-genetic covariates
    Threads.@threads for i in 1:ncovariates(v)
        id = Threads.threadid()
        copyto!(xstore[id], @view(z[cv_wts, i]))
        linreg!(xstore[id], ystore, xtx_store[id], xty_store[id])
        c[i] = xty_store[id][2]
    end
    clamp!(v.b, -2, 2)
    clamp!(v.c, -2, 2)
    copyto!(v.b0, v.b)
    copyto!(v.c0, v.c)
end

"""
    linreg!(x::Vector, y::Vector)

Performs linear regression with `y` as response, `x` and a vector of 1 as
covariate. `β̂` will be stored in `xty_store`. 

Code inspired from Doug Bates on Discourse:
https://discourse.julialang.org/t/efficient-way-of-doing-linear-regression/31232/28
"""
function linreg!(
    x::AbstractVector{T},
    y::AbstractVector{T},
    xtx_store::AbstractMatrix{T} = zeros(T, 2, 2),
    xty_store::AbstractVector{T} = zeros(T, 2)
    ) where {T<:AbstractFloat}
    N = length(x)
    N == length(y) || throw(DimensionMismatch())
    xtx_store[1, 1] = N
    xtx_store[1, 2] = sum(x)
    xtx_store[2, 2] = sum(abs2, x)
    xty_store[1] = sum(y)
    xty_store[2] = dot(x, y)
    try # rare SNPs may have 0s everywhere, causing cholesky to fail
        ldiv!(cholesky!(Symmetric(xtx_store, :U)), xty_store)
    catch
        return xty_store
    end
    return xty_store
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

"""
    naive_impute(x, destination)

Imputes missing entries of a SnpArray using the mode of each SNP, and
saves the result in a new file called destination in current directory. 
Non-missing entries are the same. 
"""
function naive_impute(x::SnpArray, destination::String)
    n, p = size(x)
    y = SnpArray(destination, n, p)

    @inbounds for j in 1:p

        #identify mode
        entry0, entry1, entry2 = 0, 0, 0
        for i in 1:n
            if x[i, j] == 0x00 
                y[i, j] = 0x00
                entry0 += 1
            elseif x[i, j] == 0x02 
                y[i, j] = 0x02
                entry1 += 1
            elseif x[i, j] == 0x03 
                y[i, j] = 0x03
                entry2 += 1
            end
        end
        most_often = max(entry0, entry1, entry2)
        missing_entry = 0x00
        if most_often == entry1
            missing_entry = 0x02
        elseif most_often == entry2
            missing_entry = 0x03
        end

        # impute 
        for i in 1:n
            if x[i, j] == 0x01 
                y[i, j] = missing_entry
            end
        end
    end

    return nothing
end

# small function to check sparsity parameter `k` is reasonable. 
function check_group(k, group)
    if typeof(k) <: Vector 
        @assert length(group) > 1 "Doubly sparse projection specified (since k" * 
            " is a vector) but there are no group information."
        for i in 1:length(k)
            group_member = count(x -> x == i, group)
            group_member > k[i] || throw(DomainError("Maximum predictors for group " * 
                "$i was $(k[i]) but there are only $group_member predictors is this " * 
                "group. Please choose a smaller number."))
        end
    else
        @assert k >= 0 "Value of k (max predictors per group) must be nonnegative!\n"
    end
end

# helper function from https://discourse.julialang.org/t/how-to-find-out-the-version-of-a-package-from-its-module/37755
pkgversion(m::Module) = Pkg.TOML.parsefile(joinpath(dirname(string(first(methods(m.eval)).file)), "..", "Project.toml"))["version"]

function print_iht_signature(io::IO)
    v = pkgversion(MendelIHT)
    println(io, "****                   MendelIHT Version $v                  ****")
    println(io, "****     Benjamin Chu, Kevin Keys, Chris German, Hua Zhou       ****")
    println(io, "****   Jin Zhou, Eric Sobel, Janet Sinsheimer, Kenneth Lange    ****")
    println(io, "****                                                            ****")
    println(io, "****                 Please cite our paper!                     ****")
    println(io, "****         https://doi.org/10.1093/gigascience/giaa044        ****")
    println(io, "")
    io != stdout && print_iht_signature(stdout)
end
print_iht_signature() = print_iht_signature(stdout)

function print_parameters(io::IO, k, d, l, use_maf, group, debias, tol, max_iter, min_iter)
    regression = typeof(d) <: Normal ? "linear" : typeof(d) <: Bernoulli ? 
        "logistic" : typeof(d) <: Poisson ? "Poisson" : 
        typeof(d) <: NegativeBinomial ? "NegativeBinomial" : 
        typeof(d) <: MvNormal ? "Multivariate Gaussian" : "unknown"
    println(io, "Running sparse $regression regression")
    println(io, "Number of threads = ", Threads.nthreads())
    println(io, "Link functin = $l")
    typeof(k) <: Int && println(io, "Sparsity parameter (k) = $k")
    typeof(k) <: Vector{Int} && println(io, "Sparsity parameter (k) = using group membership specified in k")
    println(io, "Prior weight scaling = ", use_maf ? "on" : "off")
    println(io, "Doubly sparse projection = ", length(group) > 0 ? "on" : "off")
    println(io, "Debiasing after $debias iterations")
    println(io, "Max IHT iterations = $max_iter")
    println(io, "Converging when tol < $tol and iteration ≥ $min_iter:\n")
    io != stdout && print_parameters(stdout, k, d, l, use_maf, group, debias, tol, max_iter, min_iter)
end
print_parameters(k, d, l, use_maf, group, debias, tol, max_iter, min_iter) = 
    print_parameters(stdout, k, d, l, use_maf, group, debias, tol, max_iter, min_iter)

function check_convergence(v::IHTVariable)
    the_norm = max(chebyshev(v.b, v.b0), chebyshev(v.c, v.c0)) #max(abs(x - y))
    scaled_norm = the_norm / (max(norm(v.b0, Inf), norm(v.c0, Inf)) + 1.0)
    return scaled_norm
end

function backtrack!(v::IHTVariable, η::Float)
    # recompute gradient step
    copyto!(v.b, v.b0)
    copyto!(v.c, v.c0)
    _iht_gradstep!(v, η)

    # recompute η = xb, μ = g(η), and loglikelihood to see if we're now increasing
    update_xb!(v)
    update_μ!(v)
    if v.est_r != :None
        v.d = mle_for_r(v)
    end
    
    return loglikelihood(v)
end

function check_data_dim(y::AbstractVecOrMat, x::AbstractMatrix, z::AbstractVecOrMat)
    if is_multivariate(y)
        r, n1 = size(y)
        p, n2 = size(x)
        q, n3 = size(z)
        n1 == n2 == n3 || throw(DimensionMismatch("Detected multivariate analysis" *
            " but size(y, 2) = $n1, size(x, 2) = $n2, size(z, 2) = $n3 which don't " * 
            "match. Recall each column of `y`, `x`, `z` should be sample " * 
            "phenotypes/genotypes/covariates."))
    else
        n1 = length(y)
        n2, p = size(x)
        n3, = size(z)
        n1 == n2 == n3 || throw(DimensionMismatch("Detected univariate analysis " *
        "but length(y) = $n1, size(x, 1) = $n2, size(z, 1) = $n3 which don't match. " * 
        "Recall each `y` should be a vector of phenotypes, and each row of `x`" * 
        " and `z` should be sample genotypes/covariates."))
    end
end

function save_best_model!(v::IHTVariable)
    # compute η = xb with the best estimated model
    copyto!(v.b, v.best_b)
    copyto!(v.c, v.best_c)
    v.idx .= v.b .!= 0
    v.idc .= v.c .!= 0
    check_covariate_supp!(v)
    update_xb!(v)

    # update estimated mean μ with genotype predictors
    update_μ!(v.μ, v.xb, v.l) 
end

"""
    debias!(v::IHTVariable)

After each IHT iteration, `β` is sparse. This function solves for the exact
solution `β̂` on the non-zero indices of `β`, a process known as debiasing.
"""
function debias!(v::IHTVariable)
    if sum(v.idx) == size(v.xk, 2)
        temp_glm = fit(GeneralizedLinearModel, v.xk, v.y, v.d, v.l)
        view(v.b, v.idx) .= temp_glm.pp.beta0
    end
end
