"""
    loglikelihood(v::mIHTVariable)

Calculates the loglikelihood of observing `Y` given mean `μ` and covariance `Σ`
under a multivariate Gaussian.
"""
function loglikelihood(v::mIHTVariable)
    Y = v.y # n × r
    Σ = v.Σ # r × r
    # TODO fix naive implementation below
    Γ = inv(Σ)
    δ = v.resid
    return size(Y, 1) / 2 * logdet(Γ) + tr(Γ * (δ' * δ))
end

"""
    update_xb!(v::mIHTVariable)

Update the linear predictors `xb` with the new proposed `b`. `b` is sparse but 
`c` (beta for non-genetic covariates) is dense.
"""
function update_xb!(v::mIHTVariable)
    copyto!(v.xk, @view(v.x[:, v.idx]))
    A_mul_B!(v.xb, v.zc, v.xk, v.z, view(v.b, v.idx, :), v.c)
end

"""
    update_μ!(v::mIHTVariable)

Update the mean `μ` with the linear predictors `xb` and `zc`. Here `xb` is the 
genetic contribution and `zc` is the non-genetic contribution of all covariates
"""
function update_μ!(v::mIHTVariable)
    μ = v.μ
    xb = v.xb
    zc = v.zc
    @inbounds @simd for i in eachindex(μ)
        μ[i] = xb[i] + zc[i]
    end
end

"""
    score!(v::mIHTVariable)

Calculates the score (gradient) `X'(Y - μ)Γ` for multivariate Gaussian model.
"""
function score!(v::mIHTVariable)
    y = v.y
    μ = v.μ
    r = v.resid
    @inbounds for i in eachindex(y)
        r[i] = y[i] - μ[i]
    end
    # TODO fix naive implementation below
    Γ = inv(v.Σ)
    v.df = Transpose(v.x) * r * Γ
    v.df2 = Transpose(v.z) * r * Γ
end

"""
Computes the gradient step v.b = P_k(β + η∇f(β)) and updates idx and idc. 
"""
function _iht_gradstep(v::mIHTVariable, η::Float)
    full_b = v.full_b # use full_b as storage for complete beta = [v.b ; v.c]
    lb = size(v.b, 1)
    lf = size(full_b, 1)

    # take gradient step: b = b + η ∇f
    BLAS.axpy!(η, v.df, v.b)
    BLAS.axpy!(η, v.df2, v.c)
    BLAS.axpy!(η, v.dΣ, v.Σ)

    # store complete beta [v.b; v.c] in full_b 
    full_b[1:lb, :] .= v.b
    full_b[lb+1:lf, :] .= v.c

    # project beta to sparsity and Σ to nearest pd matrix
    project_k!(full_b, v.k)
    solve_Σ!(v)
    
    # save model after projection
    copyto!(v.b, @view(full_b[1:lb, :]))
    copyto!(v.c, @view(full_b[lb+1:lf, :]))

    #recombute support
    update_support!(v.idx, v.b)
    update_support!(v.idc, v.c)
    
    # if more than k entries are selected per column, randomly choose k of them
    _choose!(v)

    # make necessary resizing since grad step might include/exclude non-genetic covariates
    check_covariate_supp!(v) 
end

"""
    update_support!(idx::BitVector, b::AbstractMatrix)

Updates `idx` so that `idx[i] = true` if `b[i, j] ≠ 0` for some `j`, otherwise
`idx[i] = false`. 
"""
function update_support!(idx::BitVector, b::AbstractMatrix{T}) where T
    p, r = size(b)
    fill!(idx, false)
    @inbounds for j in 1:r
        nz = 0
        for i in 1:p
            if b[i, j] != 0.0
                idx[i] = true
                nz += 1
            end
        end

    end
    return nothing
end

"""
Computes the best step size 
"""
function iht_stepsize(v::mIHTVariable{T, M}) where {T <: Float, M}
    return one(T) # TODO
end

"""
    project_Σ!(A)

Projects square matrix `A` to the nearest covariance (symmetric + pos def) matrix.
"""
function project_Σ!(A::AbstractMatrix)
    return A # TODO
end

"""
    solve_Σ!(Σ)

Solve for `Σ` exactly rather than projecting. 
"""
function solve_Σ!(v::mIHTVariable)
    n = size(v.y, 1)
    v.Σ = n * v.resid' * v.resid
end

# TODO: How would this function work for shared predictors?
function project_k!(x::AbstractMatrix{T}, k::Int64) where {T <: Float}
    for xi in eachcol(x)
        project_k!(xi, k)
    end
end

# TODO: How would this function work for non-shared predictors?
# TODO: Use ElasticArrays.jl to resize
function check_covariate_supp!(v::mIHTVariable{T, M}) where {T <: Float, M}
    n, r = size(v.y)
    nzidx = sum(v.idx)
    if nzidx != size(v.xk, 2)
        v.xk = zeros(T, n, nzidx)
    end
end

function _choose!(v::mIHTVariable)
    n, p = size(v.b)
    sparsity = v.k
    nz_idx = Int[]
    # loop through columns of full beta
    for j in 1:p
        # find position of non-zero beta in column j
        nz = 0
        for i in 1:n
            if v.b[i, j] != 0
                nz += 1
                push!(nz_idx, i)
            end
        end
        excess = nz - sparsity
        # if more non-zero beta than expected, randomly set a few to zero
        if excess > 0
            shuffle!(nz_idx)
            for i in 1:excess
                v.b[nz_idx[i], j] = 0
            end
        end
        empty!(nz_idx)
    end
end

"""
Function that saves variables that need to be updated each iteration
"""
function save_prev!(v::mIHTVariable)
    copyto!(v.b0, v.b)     # b0 = b
    copyto!(v.xb0, v.xb)   # Xb0 = Xb
    copyto!(v.idx0, v.idx) # idx0 = idx
    copyto!(v.idc0, v.idc) # idc0 = idc
    copyto!(v.c0, v.c)     # c0 = c
    copyto!(v.zc0, v.zc)   # Zc0 = Zc
    copyto!(v.Σ0, v.Σ)   # Zc0 = Zc
end

function checky(y::AbstractMatrix, d::MvNormal)
    return nothing
end

"""
When initializing the IHT algorithm, take `k` largest elements in magnitude of 
the score as nonzero components of b. This function set v.idx = 1 for
those indices. 
"""
function init_iht_indices!(v::mIHTVariable)
    z = v.z
    y = v.y
    k = v.k
    c = v.c

    # TODO: find intercept by Newton's method
    # ybar = mean(y)
    # for iteration = 1:20 
    #     g1 = g2 = c[1]
    #     c[1] = c[1] - clamp((g1 - ybar) / g2, -1.0, 1.0)
    #     abs(g1 - ybar) < 1e-10 && break
    # end
    # mul!(v.zc, z, v.c)

    # update mean vector and use them to compute score (gradient)
    update_μ!(v)
    score!(v)

    # first `k` non-zero entries are chosen based on largest gradient
    ldf = size(v.df, 1)
    v.full_b[1:ldf, :] .= v.df
    v.full_b[ldf+1:end, :] .= v.df2
    @inbounds for c in 1:size(v.full_b, 2)
        col = @view(v.full_b[:, c])
        a = partialsort(col, k, by=abs, rev=true)
        for i in 1:size(v.df, 1)
            abs(v.df[i, c]) > abs(a) && (v.idx[i] = true)
        end
        for i in 1:size(v.df2, 1)
            abs(v.df2[i, c]) > abs(a) && (v.idc[i] = true)
        end
    end

    # Choose randomly if more are selected
    _choose!(v) 

    # make necessary resizing when necessary
    check_covariate_supp!(v)
end
