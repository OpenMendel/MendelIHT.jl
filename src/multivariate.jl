"""
    loglikelihood(Y, μ, Σ)

Calculates the loglikelihood of observing `Y` given mean `μ` and covariance `Σ`
under a multivariate Gaussian.
"""
function loglikelihood(
    Y::AbstractMatrix{T}, # n × r
    μ::AbstractMatrix{T}, # n × r
    Σ::AbstractMatrix{T}  # r × r
    ) where {T <: Float}
    n = size(Y, 1)
    Γ = inv(Σ)
    δ = y - μ
    return n/2 * logdet(Γ) + tr(Γ * (δ' * δ))
end
