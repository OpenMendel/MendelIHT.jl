"""
    pve(y, X, β; l = IdentityLink())

Estimates phenotype's Proportion of Variance Explained (PVE) by typed genotypes 
(i.e. chip heritability or SNP heritability).

# Model
We compute `Var(μ) / Var(y)` where `y` is the raw phenotypes, `X` contains 
all the genotypes, and `μ = g^{-1}(Xβ)` is the predicted (average) phenotype
values from the statistical model β. 
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

function _pve(y::AbstractVecOrMat, μ::AbstractVecOrMat)
    return var(μ) / var(y)
end

function pve(v::IHTVariable)
    update_μ!(v.μ, v.xb, v.l) # update estimated mean μ with genotype predictors
    return _pve(v.y, v.μ)
end

function pve(v::mIHTVariable)
    update_μ!(v.μ, v.BX, IdentityLink()) # update estimated mean μ with genotype predictors
    r = ntraits(v)
    return [_pve(@view(v.Y[i, :]), @view(v.μ[i, :])) for i in 1:r]
end
