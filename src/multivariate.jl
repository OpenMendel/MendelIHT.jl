"""
    loglikelihood(v::mIHTVariables{T, M})

Calculates the loglikelihood of observing `Y` given mean `μ` and covariance `Σ`
under a multivariate Gaussian.
"""
function loglikelihood(v::mIHTVariables{T, M}) where {T <: Float, M <: AbstractMatrix}
    Y = v.y # n × r
    μ = v.μ # n × r
    Σ = v.Σ # r × r
    # TODO fix naive implementation below
    Γ = inv(Σ)
    δ = y - μ
    return size(Y, 1) / 2 * logdet(Γ) + tr(Γ * (δ' * δ))
end

"""
Update the linear predictors `xb` with the new proposed `b`. `b` is sparse but 
`c` (beta for non-genetic covariates) is dense.
"""
function update_xb!(v::mIHTVariables{T, M}) where {T <: Float, M <: AbstractMatrix}
    copyto!(v.xk, @view(v.x[:, v.idx]))
    A_mul_B!(v.xb, v.zc, v.xk, v.z, view(v.b, v.idx), v.c)
end

"""
    score!(v::mIHTVariable{T})

Calculates the score (gradient) `X'(Y - μ)Γ` for multivariate Gaussian model.

W is a diagonal matrix where w[i, i] = dμ/dη / var(μ). 
"""
function score!(v::mIHTVariables{T, M}) where {T <: Float, M <: AbstractMatrix}
    @inbounds for i in eachindex(y)
        v.resid[i] = w * (y[i] - v.μ[i])
    end
    # TODO fix naive implementation below
    Γ = inv(v.Σ)
    v.df = v.x * Tranpose(v.resid) * Γ # not correct
    At_mul_B!(v.df, v.df2, x, z, v.r, v.r)
end
