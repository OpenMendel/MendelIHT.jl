"""
    pve(y, X, β; l = IdentityLink())

Estimates phenotype's Proportion of Variance Explained (PVE) by typed genotypes 
(i.e. chip heritability or SNP heritability).

# Model
We compute `Var(ŷ) / Var(y)` where `y` is the raw phenotypes, `X` contains 
all the genotypes, and `ŷ = Xβ` is the predicted (average) phenotype
values from the statistical model β. Intercept is NOT included.
"""
function pve(
    y::AbstractVecOrMat,
    X::AbstractVecOrMat,
    β::AbstractVecOrMat;
    l::Link = IdentityLink()
    )
    μ = linkinv.(l, X * β) # mean
    return _pve(y, μ)
end

function _pve(y::AbstractVector, μ::AbstractVector)
    return var(μ) / var(y)
end

# each column is a trait
function _pve(Y::AbstractMatrix, μ::AbstractMatrix)
    size(Y, 2) == size(μ, 2) || error("Number of columns not equal in Y and μ!")
    return [_pve(@view(Y[i, :]), @view(μ[i, :])) for i in 1:size(Y, 2)]
end

function pve(v::IHTVariable)
    return _pve(v.y, v.μ)
end

function pve(v::mIHTVariable)
    return [_pve(@view(v.Y[i, :]), @view(v.μ[i, :])) for i in 1:ntraits(v)]
end
