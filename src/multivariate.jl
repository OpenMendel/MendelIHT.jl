"""
    loglikelihood(v::mIHTVariable)

Calculates the loglikelihood of observing `Y` given mean `μ` and precision matrix
`Γ` (inverse covariance matrix) under a multivariate Gaussian.

# TODO
Fix logdet allocation by computing cholesky factor of Σ. 
logdet(Γ) = logdet(inv(Σ)) = logdet(inv(LL')) = log(det(inv(L)')det(inv(L))) = 2logdet(L) (which shouldnt allocate since L is upper triangular)
https://ucla-biostat-257-2020spring.github.io/slides/12-chol/chol.html#Multivariate-normal-density
"""
function loglikelihood(v::mIHTVariable)
    Y = v.Y
    Γ = v.Γ
    δ = v.resid
    mul!(v.r_by_r1, δ, Transpose(δ)) # r_by_r = (Y - BX)(Y - BX)'
    mul!(v.r_by_r2, Γ, v.r_by_r1) # r_by_r2 = Γ(Y - BX)(Y - BX)'
    return nsamples(v) / 2 * logdet(Γ) + tr(v.r_by_r2) # logdet allocates! 
end

"""
    update_xb!(v::mIHTVariable)

Update the linear predictors `BX` with the new proposed `B`. `B` is sparse but 
`C` (beta for non-genetic covariates) is dense.
"""
function update_xb!(v::mIHTVariable)
    copyto!(v.Xk, @view(v.X[v.idx, :]))
    A_mul_B!(v.BX, v.CZ, view(v.B, :, v.idx), v.C, v.Xk, v.Z)
end

"""
    update_μ!(v::mIHTVariable)

Update the mean `μ` with the linear predictors `BX` and `CZ`. Here `BX` is the 
genetic contribution and `CZ` is the non-genetic contribution of all covariates
"""
function update_μ!(v::mIHTVariable)
    μ = v.μ
    BX = v.BX
    CZ = v.CZ
    @inbounds @simd for i in eachindex(μ)
        μ[i] = BX[i] + CZ[i]
    end
end

"""
    score!(v::mIHTVariable)

Calculates the gradient `Γ(Y - XB)X'` for multivariate Gaussian model.
"""
function score!(v::mIHTVariable)
    y = v.Y
    μ = v.μ
    r = v.resid # r × n
    Γ = v.Γ # r × r
    @inbounds for i in eachindex(y)
        r[i] = y[i] - μ[i]
    end
    mul!(v.r_by_n1, Γ, r) # r_by_n1 = Γ(Y - BX)
    mul!(v.df, v.r_by_n1, Transpose(v.X)) # v.df = Γ(Y - BX)X'
    mul!(v.df2, v.r_by_n1, Transpose(v.Z)) # v.df2 = Γ(Y - BX)Z'
    v.fullIHT && score_Γ!(v)
end

# TODO: efficieny
function score_Γ!(v::mIHTVariable)
    mul!(v.r_by_r1, v.resid, v.resid') # r_by_r1 = (Y - BX)(Y - BX)'
    v.dΓ = nsamples(v) / 2 * inv(v.Γ) - 0.5v.r_by_r1
end

"""
    _iht_gradstep(v::mIHTVariable, η::Float)

Computes the gradient step v.b = P_k(β + η∇f(β)) and updates idx and idc. 
"""
function _iht_gradstep(v::mIHTVariable, η::Float)
    full_b = v.full_b # use full_b as storage for complete beta = [v.b v.c]
    p = nsnps(v)

    # take gradient step: b = b + η ∇f
    BLAS.axpy!(η, v.df, v.B)
    BLAS.axpy!(η, v.df2, v.C)
    BLAS.axpy!(η, v.dΓ, v.Γ)

    # store complete beta [v.b v.c] in full_b 
    full_b[:, 1:p] .= v.B
    full_b[:, p+1:end] .= v.C

    # project beta to sparsity. Project Γ to nearest pd matrix or solve for Σ exactly
    project_k!(full_b, v.k)
    v.fullIHT ? project_Σ!(v) : solve_Σ!(v)

    # save model after projection
    copyto!(v.B, @view(full_b[:, 1:p]))
    copyto!(v.C, @view(full_b[:, 1:p+1:end]))

    #recombute support
    update_support!(v.idx, v.B)
    update_support!(v.idc, v.C)

    # if more than k entries are selected per column, randomly choose k of them
    _choose!(v)

    # make necessary resizing since grad step might include/exclude non-genetic covariates
    check_covariate_supp!(v) 
end

"""
    update_support!(idx::BitVector, b::AbstractMatrix)

Updates `idx` so that `idx[i] = true` if `i`th column of `v.B` contains non-0
values. 
"""
function update_support!(idx::BitVector, b::AbstractMatrix{T}) where T
    r, p = size(b)
    fill!(idx, false)
    @inbounds for j in 1:p, i in 1:r
        if b[i, j] != zero(T)
            idx[j] = true
        end
    end
    return nothing
end

"""
    iht_stepsize(v)

Computes the best step size `η = ||∇f||^2_F / tr(X'∇f'Γ∇fX)` where
∇f = Γ(Y - BX)(Y - BX)'. Note the denominator can be rewritten as
||v||^2_F where v = sqrt(Γ)∇fX.

TODO: sqrt(Γ) is allocating
"""
function iht_stepsize(v::mIHTVariable{T, M}) where {T <: Float, M}
    # store part of X corresponding to non-zero component of B
    copyto!(v.dfidx, @view(v.df[:, v.idx]))

    # compute numerator of step size
    numer = zero(T)
    @inbounds for i in eachindex(v.dfidx)
        numer += abs2(v.dfidx[i])
    end
    if v.fullIHT
        for i in v.dΓ
            numer += abs2(i)
        end
    end

    # compute denominator of step size
    # println(v.Γ)
    # println(inv(v.Γ))
    denom = zero(T)
    if v.fullIHT
        # TODO efficiency
        denom += tr(v.Xk' * v.dfidx' * v.Γ * v.dfidx * v.Xk)
        denom += tr((inv(v.Γ) * v.dΓ)^2)
    else
        mul!(v.r_by_n1, v.dfidx, v.Xk) # r_by_n1 = ∇f*X
        mul!(v.r_by_n2, sqrt(v.Γ), v.r_by_n1) # r_by_n2 = sqrt(Γ)*∇f*X
        @inbounds for i in eachindex(v.r_by_n2)
            denom += abs2(v.r_by_n2[i])
        end
    end

    return numer / denom :: T
end

"""
    project_Σ!(A::AbstractMatrix)

Projects square matrix `A` to the nearest covariance (symmetric + pos def) matrix.

# TODO: efficiency
"""
function project_Σ!(v::mIHTVariable, tol = 0.01)
    λs, U = eigen(v.Γ)
    for (i, λ) in enumerate(λs)
        λ < 0 && (λs[i] = tol)
    end
    v.Γ = U * Diagonal(λs) * U'
    # println("issymmetric = ", issymmetric(v.Γ))
    # println("isposdef = ", isposdef(v.Γ))
    # println("condition number = ", cond(v.Γ))
end

"""
    solve_Σ!(v::mIHTVariable)

Solve for `Σ = 1/n(Y-BX)(Y-BX)'` exactly rather than projecting
"""
function solve_Σ!(v::mIHTVariable)
    mul!(v.r_by_r1, v.resid, Transpose(v.resid)) # r_by_r1 = (Y-BX)(Y-BX)'
    v.r_by_r1 ./= nsamples(v)
    LinearAlgebra.inv!(cholesky!(v.r_by_r1)) # r_by_r1 = (1/n(Y-BX)(Y-BX)')^{-1}
    copyto!(v.Γ, v.r_by_r1)
end

"""
    project_k!(x::AbstractMatrix{T}, k::Int64)

Project `x` so that each row contains `k` or fewer non-zero entries. 

TODO: How would this function work for shared predictors?
"""
function project_k!(x::AbstractMatrix{T}, k::Int64) where {T <: Float}
    for xi in eachrow(x)
        project_k!(xi, k)
    end
end

"""
    check_covariate_supp!(v::mIHTVariable)

Possibly rescales `v.Xk` and `v.dfidx`, which needs to happen when non-genetic
covariates get included/excluded between different iterations
"""
function check_covariate_supp!(v::mIHTVariable{T, M}) where {T <: Float, M}
    n, r = nsamples(v), ntraits(v)
    nzidx = sum(v.idx)
    if nzidx != size(v.Xk, 1)
        v.Xk = zeros(T, nzidx, n)
        v.dfidx = zeros(T, r, nzidx) # TODO ElasticArrays.jl
    end
end

"""
    _choose!(v::mIHTVariable)

Strictly force each row of `B` to have `k` or fewer non-zero entries. When `B`
has `≥k` non-zero entries in a row, randomly choose `k` among them. Due to
numerical error, it is possible to have `≥k` non-zero entries per row after
projection.
"""
function _choose!(v::mIHTVariable)
    n, p = nsamples(v), nsnps(v)
    sparsity = v.k
    nz_idx = Int[]
    # loop through rows of full beta
    @inbounds for i in 1:ntraits(v)
        # find position of non-zero beta
        nz = 0
        for j in 1:p
            if v.B[i, j] != 0
                nz += 1
                push!(nz_idx, j)
            end
        end
        excess = nz - sparsity
        # if more non-zero beta than expected, randomly set a few to zero
        if excess > 0
            shuffle!(nz_idx)
            for i in 1:excess
                v.B[i, nz_idx[i]] = 0
            end
        end
        empty!(nz_idx)
    end
end

"""
Function that saves variables that need to be updated each iteration
"""
function save_prev!(v::mIHTVariable)
    copyto!(v.B0, v.B)     # B0 = B
    copyto!(v.BX0, v.BX)   # BX0 = BX
    copyto!(v.idx0, v.idx) # idx0 = idx
    copyto!(v.idc0, v.idc) # idc0 = idc
    copyto!(v.C0, v.C)     # C0 = C
    copyto!(v.CZ0, v.CZ)   # CZ0 = CZ
    copyto!(v.Γ0, v.Γ)     # Γ0 = Γ
end

checky(y::AbstractMatrix, d::MvNormal) = nothing

"""
When initializing the IHT algorithm, take `k` largest elements in magnitude of 
the score as nonzero components of b. This function set v.idx = 1 for
those indices. 
"""
function init_iht_indices!(v::mIHTVariable)
    z = v.Z
    y = v.Y
    k = v.k
    c = v.C

    # TODO: find intercept by Newton's method
    # ybar = mean(y)
    # for iteration = 1:20 
    #     g1 = g2 = c[1]
    #     c[1] = c[1] - clamp((g1 - ybar) / g2, -1.0, 1.0)
    #     abs(g1 - ybar) < 1e-10 && break
    # end
    c[:, 1] .= 1.0
    mul!(v.CZ, c, v.Z)

    # update mean vector and use them to compute score (gradient)
    update_μ!(v)
    score!(v)

    # first `k` non-zero entries are chosen based on largest gradient
    p, q = nsnps(v), ncovariates(v)
    v.full_b[:, 1:p] .= v.df
    v.full_b[:, p+1:end] .= v.df2
    @inbounds for r in 1:ntraits(v)
        row = @view(v.full_b[r, :])
        a = partialsort(row, k, by=abs, rev=true)
        for i in 1:p
            abs(v.df[r, i]) ≥ abs(a) && (v.idx[i] = true)
        end
        for i in 1:q
            abs(v.df2[r, i]) ≥ abs(a) && (v.idc[i] = true)
        end
    end

    # Choose randomly if more are selected
    _choose!(v)

    # make necessary resizing when necessary
    check_covariate_supp!(v)

    # store relevant components of x for first iteration
    copyto!(v.Xk, @view(v.X[v.idx, :])) 
end

function check_convergence(v::mIHTVariable)
    the_norm = max(chebyshev(v.B, v.B0), chebyshev(v.C, v.C0)) #max(abs(x - y))
    scaled_norm = the_norm / (max(norm(v.B0, Inf), norm(v.C0, Inf)) + 1.0)
    return scaled_norm
end

function backtrack!(v::mIHTVariable, η::Float)
    # recompute gradient step
    copyto!(v.B, v.B0)
    copyto!(v.C, v.C0)
    _iht_gradstep(v, η)

    # recompute η = xb, μ = g(η), and loglikelihood to see if we're now increasing
    update_xb!(v)
    update_μ!(v)
    
    return loglikelihood(v)
end

"""
    is_multivariate(y::AbstractVecOrMat)

Returns true if response `y` can be modeled by a multivariate Gaussian
distributions. Currently simply checks whether there is >1 trait
"""
function is_multivariate(y::AbstractVecOrMat)
    size(y, 1) > 1 && size(y, 2) > 1
end
