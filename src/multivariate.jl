"""
    loglikelihood(v::mIHTVariable)

Calculates the loglikelihood of observing `Y` given mean `μ` and precision matrix
`Γ` (inverse covariance matrix) under a multivariate Gaussian.

Caution: This function mutates storage variables `v.r_by_r1` and `v.r_by_r2`
"""
function loglikelihood(v::mIHTVariable)
    mul!(v.r_by_r1, v.resid, Transpose(v.resid)) # r_by_r = (Y - BX)(Y - BX)'
    mul!(v.r_by_r2, v.Γ, v.r_by_r1) # r_by_r2 = Γ(Y - BX)(Y - BX)'
    return nsamples(v) / 2 * logdet(v.Γ) - 0.5 * tr(v.r_by_r2) # logdet allocates! 
end

"""
    update_xb!(v::mIHTVariable)

Using only support of B, update `v.BX` with the new proposed `B` (effect size
for genetic covariates) and `C` (non-genetic covariates).
"""
function update_xb!(v::mIHTVariable)
    copyto!(v.Xk, @view(v.X[v.idx, :]))
    mul!(v.BX, view(v.B, :, v.idx), v.Xk)
    mul!(v.CZ, v.C, v.Z)
end

"""
    update_μ!(v::mIHTVariable)

Update the mean `μ` with the linear predictors `BX` and `CZ`. Here `BX` is the 
genetic contribution and `CZ` is the non-genetic contribution
"""
function update_μ!(v::mIHTVariable)
    @inbounds @simd for i in eachindex(v.μ)
        v.μ[i] = v.BX[i] + v.CZ[i]
    end
end

"""
    update_resid!(v)

Update residuals `resid = (Y - BX)` where `BX = μ` has already been computed.
"""
function update_resid!(v::mIHTVariable)
    y = v.Y # r × n
    μ = v.μ # r × n
    r = v.resid # r × n
    cv_wts = v.cv_wts # n × 1 cross validation weights
    @inbounds for j in 1:size(y, 2), i in 1:size(y, 1)
        r[i, j] = (y[i, j] - μ[i, j]) * cv_wts[j]
    end
end

"""
    score!(v::mIHTVariable)

Calculates the gradient `Γ(Y - XB)X'` for multivariate Gaussian model. Residuals
were updated in solve_Σ!. 
"""
function score!(v::mIHTVariable)
    mul!(v.r_by_n1, v.Γ, v.resid) # r_by_n1 = Γ(Y - BX)
    update_df!(v) # v.df = Γ(Y - BX)X' (genetic gradient)
    mul!(v.df2, v.r_by_n1, Transpose(v.Z)) # v.df2 = Γ(Y - BX)Z' (nongenetic gradient)
end

"""
    update_df!(v::mIHTVariable)

Compute `v.df = Γ(Y - BX)X'` efficiently where `v.r_by_n1 = Γ(Y - BX)`. Here
`X` is `p × n` where `p` is number of SNPs and `n` is number of samples. 

Note if `X` is a `SnpLinAlg`, currently only `X * v` for vector `v` is
efficiently implemented in `SnpArrays.jl`. For `X * V` with matrix `V`, we need
to compute `X * vi` for each volumn of `V`. Thus, we compute:

`v.n_by_r = Transpose(r_by_n1)`
`v.p_by_r[:, i] = v.X * v.n1_by_r[:, i]` # compute matrix-vector using SnpArrays's efficient routine
`v.df = Transpose(v.p_by_r)`
"""
function update_df!(v::mIHTVariable)
    T = eltype(v.X)
    if typeof(v.X) <: Union{Transpose{T, SnpLinAlg{T}}, Adjoint{T, SnpLinAlg{T}}}
        v.n_by_r .= Transpose(v.r_by_n1)
        SnpArrays.mul!(v.p_by_r, v.X, v.n_by_r) # note v.X is Transpose(SnpLinAlg)
        v.df .= Transpose(v.p_by_r)
    elseif typeof(v.X) <: Union{Transpose{T, AbstractMatrix{T}}, Adjoint{T, AbstractMatrix{T}}} # v.X is transposed numeric matrix
        mul!(v.df, v.r_by_n1, v.X.parent)
    else
        mul!(v.df, v.r_by_n1, Transpose(v.X)) # v.X is numeric matrix
    end
end

"""
    _iht_gradstep!(v::mIHTVariable, η::Float)

Computes the gradient step v.b = P_k(β + η∇f(β)) and updates idx and idc. 
"""
function _iht_gradstep!(v::mIHTVariable, η::Float)
    # take gradient step: b = b + η ∇f
    BLAS.axpy!(η, v.df, v.B)
    BLAS.axpy!(η, v.df2, v.C)

    # project beta to sparsity
    project_k!(v)
end

function project_k!(v::mIHTVariable)
    # store complete beta [v.b v.c] in full_b, potentially replacing v.C with Inf 
    vectorize!(v.full_b, v.B, v.C, v.zkeep)

    # project to sparsity: keeping k + number of nongentic covariates
    project_k!(v.full_b, v.k + v.zkeepn)

    # save model after projection
    unvectorize!(v.full_b, v.B, v.C, v.zkeep)

    # if more than k entries are selected per column, randomly choose k of them
    _choose!(v)

    #recombute support
    update_support!(v.idx, v.B)
    update_support!(v.idc, v.C)

    # make necessary resizing since grad step might include/exclude non-genetic covariates
    check_covariate_supp!(v) 
end

"""
    vectorize!(a::AbstractVector, B::AbstractMatrix, C::AbstractMatrix, Ckeep::BitVector)

Without allocations, copies `vec(B)` into `a` and the copies `vec(C)` into
remaining parts of `a`. 

`Ckeep` tracks entries of `C` that will not be projected. That is, entries of `a`
corresponding to entries in `C[:, i]` will be filled with `Inf` if `Ckeep[i] = true`.
"""
function vectorize!(a::AbstractVector, B::AbstractMatrix, C::AbstractMatrix, Ckeep::BitVector)
    size(C, 2) == length(Ckeep) ||
        error("size(C, 2) = $(size(C, 2)) and length(Ckeep) = $(length(Ckeep)) are different!")
    i = 1
    @inbounds @simd for j in eachindex(B)
        a[i] = B[j]
        i += 1
    end
    m = typemax(eltype(a))
    for j in 1:size(C, 2)
        if Ckeep[j]
            @inbounds @simd for l in 1:size(C, 1)
                a[i] = m
                i += 1
            end
        else
            @inbounds @simd for l in 1:size(C, 1)
                a[i] = C[l, j]
                i += 1
            end
        end
    end
    return nothing
end

"""
    unvectorize!(a::AbstractVector, B::AbstractMatrix, C::AbstractMatrix, Ckeep::BitVector)

Without allocations, copies the first `length(vec(B))` part of `a` into `B` 
and the remaining parts of `a` into `C`.

`Ckeep` tracks entries of `C` that will not be projected. That is, entries in `C[i, :]`
will not be touched if `Ckeep[i] = true`.
"""
function unvectorize!(a::AbstractVector, B::AbstractMatrix, C::AbstractMatrix, Ckeep::BitVector)
    size(C, 2) == length(Ckeep) ||
        error("size(C, 2) = $(size(C, 2)) and length(Ckeep) = $(length(Ckeep)) are different!")
    i = 1
    @inbounds @simd for j in eachindex(B)
        B[j] = a[i]
        i += 1
    end
    @inbounds @simd for j in 1:size(C, 2)
        if !Ckeep[j]
            for l in 1:size(C, 1)
                C[l, j] = a[i]
                i += 1
            end
        end
    end
    return nothing
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
`tr(X'∇f'LU∇fX) = ||v||^2_F` where `v = U∇fX`, `L = U'` is cholesky factor
of `Γ`.

Must be careful for cross validation weights in ∇f * X
"""
function iht_stepsize!(v::mIHTVariable{T, M}) where {T <: Float, M}
    # store part of X corresponding to non-zero component of B
    copyto!(v.dfidx, @view(v.df[:, v.idx]))

    # compute numerator of step size
    numer = zero(T)
    @inbounds for i in v.dfidx
        numer += abs2(i)
    end

    # denominator of step size
    denom = zero(T)
    mul!(v.r_by_n1, v.dfidx, v.Xk) # r_by_n1 = ∇f*X
    for j in 1:size(v.Y, 2), i in 1:ntraits(v)
        v.r_by_n1[i, j] *= v.cv_wts[j] # cross validation masking happens here
    end
    cholesky!(Symmetric(v.Γ, :U), Val(true)) # overwrite upper triangular of Γ with U, where LU = Γ, U = L'
    triu!(v.Γ) # set entries below diagonal to 0, so Γ = L'
    mul!(v.r_by_n2, v.Γ, v.r_by_n1) # r_by_n2 = L'*∇f*X
    @inbounds for i in eachindex(v.r_by_n2)
        denom += abs2(v.r_by_n2[i]) 
    end

    # for bad boundary cases (sometimes, k = 1 in cross validation generates weird η)
    η = numer / denom
    isinf(η) && (η = T(1e-8))
    isnan(η) && (η = T(1e-8))

    return η :: T
end

"""
    project_Σ!(A::AbstractMatrix)

Projects symmetric matrix `A` to the nearest positive semidefinite matrix.

# TODO: efficiency
"""
function project_Σ!(v::mIHTVariable, minλ = 0.01)
    λs, U = eigen(v.Γ)
    for (i, λ) in enumerate(λs)
        λ < minλ && (λs[i] = minλ)
    end
    v.Γ = U * Diagonal(λs) * U'
end

"""
    solve_Σ!(v::mIHTVariable)

Solve for `Σ = 1/n(Y-BX)(Y-BX)'` exactly rather than projecting
"""
function solve_Σ!(v::mIHTVariable)
    update_resid!(v) # v.resid = Y - BX
    mul!(v.r_by_r1, v.resid, Transpose(v.resid)) # r_by_r1 = (Y-BX)(Y-BX)'
    v.r_by_r1 ./= nsamples(v)
    LinearAlgebra.inv!(cholesky!(Symmetric(v.r_by_r1, :U))) # r_by_r1 = (1/n(Y-BX)(Y-BX)')^{-1}
    copyto!(v.Γ, v.r_by_r1)
end

"""
    check_covariate_supp!(v::mIHTVariable)

Possibly rescales `v.Xk` and `v.dfidx`, which needs to happen when non-genetic
covariates get included/excluded between different iterations
"""
function check_covariate_supp!(v::mIHTVariable{T, M}) where {T <: Float, M}
    n, r = size(v.X, 2), ntraits(v)
    nzidx = sum(v.idx)
    if nzidx != size(v.Xk, 1)
        v.Xk = zeros(T, nzidx, n)
        v.dfidx = zeros(T, r, nzidx) # TODO ElasticArrays.jl
    end
    @inbounds for i in eachindex(v.zkeep)
        v.zkeep[i] && !v.idc[i] && error("A non-genetic covariate was accidentally set to 0! Shouldn't happen!")
    end
end

"""
    _choose!(v::mIHTVariable)

When `B` has `≥k` non-zero entries in a row, randomly choose `k` among them. 

Note: It is possible to have `≥k` non-zero entries after projection due to
numerical errors.
"""
function _choose!(v::mIHTVariable)
    sparsity = v.k + v.zkeepn
    B_nz = 0
    C_nz = 0
    B_nz_idx = Int[]
    C_nz_idx = Tuple{Int, Int}[]
    # find position of non-zero beta
    for i in eachindex(v.B)
        if v.B[i] != 0
            B_nz += 1
            push!(B_nz_idx, i)
        end
    end
    for j in 1:size(v.C, 2)
        v.zkeep[j] && continue
        for i in 1:size(v.C, 1)
            if v.C[i, j] != 0
                C_nz += 1
                push!(C_nz_idx, (i, j))
            end
        end
    end

    # if more non-zero than expected, randomly set entries (starting in B) to 0
    excess = B_nz + C_nz - sparsity
    if excess > 0
        shuffle!(B_nz_idx)
        shuffle!(C_nz_idx)
        for i in 1:excess
            if B_nz > 0
                v.B[B_nz_idx[i]] = 0
                B_nz -= 1
            else
                (ii, jj) = C_nz_idx[i]
                v.C[ii, jj] = 0
                C_nz -= 1
            end
        end
    end
    empty!(B_nz_idx)
    empty!(C_nz_idx)
end

"""
Function that saves variables that need to be updated each iteration
"""
function save_prev!(v::mIHTVariable, cur_logl, best_logl)
    copyto!(v.B0, v.B)     # B0 = B
    copyto!(v.idx0, v.idx) # idx0 = idx
    copyto!(v.idc0, v.idc) # idc0 = idc
    copyto!(v.C0, v.C)     # C0 = C
    copyto!(v.Γ0, v.Γ)     # Γ0 = Γ
    if cur_logl > best_logl
        copyto!(v.best_B, v.B) # best_B = B
        copyto!(v.best_C, v.C) # best_C = C
    end
    return max(cur_logl, best_logl)
end

checky(y::AbstractMatrix, d::MvNormal) = nothing

"""
When initializing the IHT algorithm, take `k` largest elements in magnitude of 
the score as nonzero components of `v.B`. This function set v.idx = 1 for
those indices. `v.μ`, `v.df`, `v.df2`, and possibly `v.BX` are also initialized. 
"""
function init_iht_indices!(v::mIHTVariable, init_beta::Bool, cv_idx::BitVector)
    fill!(v.B, 0)
    fill!(v.B0, 0)
    fill!(v.best_B, 0)
    fill!(v.BX, 0)
    fill!(v.Xk, 0)
    fill!(v.idx, false)
    fill!(v.idx0, false)
    copyto!(v.idc, v.zkeep)
    copyto!(v.idc0, v.zkeep)
    fill!(v.resid, 0)
    fill!(v.df, 0)
    fill!(v.df2, 0)
    fill!(v.dfidx, 0)
    fill!(v.C, 0)
    fill!(v.C0, 0)
    fill!(v.best_C, 0)
    fill!(v.CZ, 0)
    fill!(v.μ, 0)
    fill!(v.Γ, 0)
    fill!(v.Γ0, 0)
    for i in 1:ntraits(v)
        v.Γ[i, i] = 1
        v.Γ0[i, i] = 1
    end
    fill!(v.cv_wts, 0)
    v.cv_wts[cv_idx] .= 1
    fill!(v.full_b, 0)
    fill!(v.r_by_r1, 0)
    fill!(v.r_by_r2, 0)
    fill!(v.r_by_n1, 0)
    fill!(v.r_by_n2, 0)
    fill!(v.n_by_r, 0)
    fill!(v.p_by_r, 0)
    fill!(v.k_by_r, 0)
    fill!(v.k_by_k, 0)

    # initialize intercept to mean of each trait
    nz_samples = count(!iszero, v.cv_wts) # for cross validation masking
    for i in 1:ntraits(v)
        ybar = zero(eltype(v.C))
        @inbounds for j in 1:size(v.Y, 2)
            ybar += v.Y[i, j] * v.cv_wts[j]
        end
        v.C[i, 1] = ybar / nz_samples
    end
    mul!(v.CZ, v.C, v.Z)

    if init_beta
        initialize_beta!(v, cv_idx)
        project_k!(v)
        update_xb!(v)
    end

    # update mean and compute score (gradient)
    update_μ!(v)
    update_resid!(v)
    score!(v)

    if !init_beta
        # first `k` non-zero entries in each β are chosen based on largest gradient
        vectorize!(v.full_b, v.df, v.df2, v.zkeep)
        project_k!(v.full_b, v.k + v.zkeepn) # project k + number of nongentic covariates to keep
        unvectorize!(v.full_b, v.df, v.df2, v.zkeep)

        # compute support based on largest gradient
        update_support!(v.idx, v.df)
        update_support!(v.idc, v.df2) # should we initialize v.idc to all trues?
    end

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
    # recompute gradient step with the new (halved) step size η 
    copyto!(v.B, v.B0)
    copyto!(v.C, v.C0)
    copyto!(v.Γ, v.Γ0)
    _iht_gradstep!(v, η)

    # recompute BX and ZC, μ (includes genetic + nongenetic component), and Γ
    update_xb!(v)
    update_μ!(v)
    solve_Σ!(v)

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

function save_best_model!(v::mIHTVariable)
    # compute η = BX with the best estimated model
    copyto!(v.B, v.best_B)
    copyto!(v.C, v.best_C)
    update_support!(v.idx, v.B)
    update_support!(v.idc, v.C)
    check_covariate_supp!(v)
    update_xb!(v)

    # update estimated mean μ with genotype predictors
    update_μ!(v) 
end

function save_last_model!(v::mIHTVariable)
    # compute η = BX with the estimated model from last IHT iteration
    update_support!(v.idx, v.B)
    update_support!(v.idc, v.C)
    check_covariate_supp!(v)
    update_xb!(v)

    # update estimated mean μ with genotype predictors
    update_μ!(v)
end

"""
    initialize_beta!(v, cv_wts)

Initialze beta to univariate regression values. That is, `β[i, j]` is set to the estimated
beta with `y[j, :]` as response, and `x[:, i]` with an intercept term as covariate.

Note: this function assumes quantitative (Gaussian) phenotypes. 
"""
function initialize_beta!(
    v::mIHTVariable,
    cv_wts::BitVector) # cross validation weights; 1 = sample is present, 0 = not present
    y, x, z, B, C, T = v.Y, v.X, v.Z, v.B, v.C, eltype(v.B)
    xtx_store = [zeros(T, 2, 2) for _ in 1:Threads.nthreads()]
    xty_store = [zeros(T, 2) for _ in 1:Threads.nthreads()]
    xstore = [zeros(T, sum(cv_wts)) for _ in 1:Threads.nthreads()]
    ystore = zeros(T, sum(cv_wts))
    @inbounds for j in 1:ntraits(v) # loop over each y
        copyto!(ystore, @view(y[j, cv_wts]))
        # genetic covariates
        Threads.@threads for i in 1:nsnps(v)
            id = Threads.threadid()
            copyto!(xstore[id], @view(x[i, cv_wts]))
            linreg!(xstore[id], ystore, xtx_store[id], xty_store[id])
            B[j, i] = xty_store[id][2]
        end
        # non genetic covariates
        Threads.@threads for i in 1:ncovariates(v)
            id = Threads.threadid()
            copyto!(xstore[id], @view(z[i, cv_wts]))
            linreg!(xstore[id], ystore, xtx_store[id], xty_store[id])
            C[j, i] = xty_store[id][2]
        end
    end
    clamp!(C, -2, 2)
    clamp!(B, -2, 2)
    copyto!(v.B0, v.B)
    copyto!(v.C0, v.C)
end

"""
    debias!(v::mIHTVariable)

Solves the multivariate linear regression `Y = BX + E` by `B̂ = inv(X'X) X'Y` on the
support set of `B`. Since `B` is sparse, this is a low dimensional problem, and 
the solution is unique.

Note: since `X` and `Y` are transposed in memory, we actually have `B̂ = inv(XX')XY'`
"""
function debias!(v::mIHTVariable{T, M}) where {T <: Float, M}
    # first rescale matrix dimension if needed
    supp_size = sum(v.idx)
    if supp_size != size(v.k_by_r, 1)
        v.k_by_r = Matrix{T}(undef, supp_size, size(v.k_by_r, 2))
        v.k_by_k = Matrix{T}(undef, supp_size, supp_size)
    end

    # try debiasing: B̂ = inv(XX')XY'. Do nothing if it fails
    mul!(v.k_by_r, v.Xk, Transpose(v.Y))
    mul!(v.k_by_k, v.Xk, Transpose(v.Xk))
    try
        ldiv!(cholesky!(Symmetric(v.k_by_k, :U)), v.k_by_r)
    catch
        return nothing
    end
    v.B[:, v.idx] .= v.k_by_r'

    # ensure B is k-sparse
    project_k!(v)
end
