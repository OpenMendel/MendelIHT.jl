"""
    loglikelihood(v::mIHTVariable)

Calculates the loglikelihood of observing `Y` given mean `μ` and covariance `Σ`
under a multivariate Gaussian.
"""
function loglikelihood(v::mIHTVariable)
    Y = v.y # n × r
    μ = v.μ # n × r
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
    A_mul_B!(v.xb, v.zc, v.xk, v.z, view(v.b, v.idx), v.c)
end

"""
    score!(v::mIHTVariable)

Calculates the score (gradient) `X'(Y - μ)Γ` for multivariate Gaussian model.
W is a diagonal matrix where w[i, i] = dμ/dη / var(μ). 
"""
function score!(v::mIHTVariable{T, M}) where {T <: Float, M <: AbstractMatrix}
    @inbounds for i in eachindex(y)
        v.resid[i] = w * (y[i] - v.μ[i])
    end
    # TODO fix naive implementation below
    Γ = inv(v.Σ)
    v.df = Transpose(v.x) * v.resid * Γ
    v.df2 = Transpose(v.z) * v.resid * Γ
end

"""
Computes the gradient step v.b = P_k(β + η∇f(β)) and updates idx and idc. 
"""
function _iht_gradstep(v::mIHTVariable{T, M}, η::T) where {T <: Float}
    lb = size(v.b, 1)
    lf = size(full_grad, 1)

    # take gradient step: b = b + η ∇f
    BLAS.axpy!(η, v.df, v.b)
    BLAS.axpy!(η, v.df2, v.c)

    # store full gradient
    full_grad[1:lb, :] .= v.b
    full_grad[lb+1:lf, :] .= v.c

    # project to sparsity
    project_k!(full_grad, k)
    
    # save model after projection
    copyto!(v.b, @view(full_grad[1:lb]))
    copyto!(v.c, @view(full_grad[lb+1:lf]))

    #recombute support
    v.idx .= v.b .!= 0
    v.idc .= v.c .!= 0
    
    # if more than k entries are selected per column, randomly choose k of them
    _choose!(v)

    # make necessary resizing since grad step might include/exclude non-genetic covariates
    check_covariate_supp!(v) 
end

"""
Computes the best step size 
"""
function iht_stepsize(v::mIHTVariable{T, M}) where {T <: Float}
    return one(T)
end
