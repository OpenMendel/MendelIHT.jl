"""
    fit(y, x, z; kwargs...)

Fits a model on design matrix (genotype data) `x`, response (phenotype) `y`, 
and non-genetic covariates `z` on a specific sparsity parameter `k`. If `k` is 
a constant, then each group will have the same sparsity level. To run doubly 
sparse IHT, construct `k` to be a vector where `k[i]` indicates the max number
of predictors for group `i`. 

# Arguments:
+ `y`: Response vector (phenotypes)
+ `x`: Genotype matrix (an `Array{T, 2}`, `SnpBitMatrix`, or `SnpLinAlg` (recommended))
+ `z`: Matrix of non-genetic covariates. The first column should be the intercept (i.e. column of 1). 

# Optional Arguments:
+ `k`: Number of non-zero predictors in each group. Can be a constant or a vector. 
+ `J`: The number of maximum groups (set as 1 if no group infomation available)
+ `d`: Distribution of your phenotype (e.g. Normal, Bernoulli)
+ `l`: A link function (e.g. IdentityLink, LogitLink, ProbitLink)
+ `group`: vector storing group membership
+ `weight`: vector storing vector of weights containing prior knowledge on each SNP
+ `est_r`: Symbol (`:MM` or `:Newton`) to estimate nuisance parameters for negative binomial regression
+ `use_maf`: boolean indicating whether we want to scale projection with minor allele frequencies (see paper)
+ `debias`: boolean indicating whether we debias at each iteration (see paper)
+ `verbose`: boolean indicating whether we want to print results if model does not converge.
+ `init`: boolean indicating whether we want to initialize `β` to sensible values through fitting. This is not efficient yet. 
+ `tol`: used to track convergence
+ `max_iter`: is the maximum IHT iteration for a model to converge. Defaults to 200, or 100 for cross validation
+ `max_step`: is the maximum number of backtracking. Since l0 norm is not convex, we have no ascent guarantee
"""
function fit(
    y         :: AbstractVector{T},
    x         :: AbstractMatrix{T},
    z         :: AbstractVecOrMat{T};
    k         :: Union{Int, Vector{Int}} = 10,
    J         :: Int = 1,
    d         :: UnivariateDistribution = Normal(),
    l         :: Link = IdentityLink,
    group     :: AbstractVector{Int} = Int[],
    weight    :: AbstractVector{T} = T[],
    est_r     :: Union{Symbol, Nothing} = nothing,
    use_maf   :: Bool = false, 
    debias    :: Bool = false,
    verbose   :: Bool = true,          # print informative things
    init      :: Bool = false,         # not efficient. whether to initialize β to sensible values
    tol       :: T = convert(T, 1e-4), # tolerance for tracking convergence
    max_iter  :: Int = 100,            # maximum IHT iterations
    max_step  :: Int = 5,              # maximum backtracking for each iteration
    ) where T <: Float

    #start timer
    start_time = time()

    # first handle errors
    @assert J ≥ 0        "Value of J (max number of groups) must be nonnegative!\n"
    @assert max_iter ≥ 0 "Value of max_iter must be nonnegative!\n"
    @assert max_step ≥ 0 "Value of max_step must be nonnegative!\n"
    @assert tol > eps(T)  "Value of global tol must exceed machine precision!\n"
    checky(y, d) # make sure response data y is in the form compatible with specified GLM
    check_group(k, group) # make sure sparsity parameter `k` is reasonable. 
    isnothing(est_r) || typeof(d) <: NegativeBinomial || 
        "Only negative binomial regression currently supports nuisance parameter estimation"

    # initialize constants
    mm_iter     = 0                 # number of iterations 
    tot_time    = 0.0               # compute time *within* L0_reg
    next_logl   = oftype(tol,-Inf)  # loglikelihood
    the_norm    = 0.0               # norm(b - b0)
    scaled_norm = 0.0               # the_norm / (norm(b0) + 1)
    η_step      = 0                 # counts number of backtracking steps for η
    converged   = false             # scaled_norm < tol?

    # print information 
    if verbose
        print_iht_signature()
        print_parameters(k, d, l, use_maf, group, debias, tol)
    end

    # Initialize variables. 
    v = IHTVariables(x, z, y, J, k, group, weight)             # Placeholder variable for cleaner code
    full_grad = zeros(T, size(x, 2) + size(z, 2))              # Preallocated vector for efficiency
    init_iht_indices!(v, x, z, y, d, l, J, k, group)           # initialize non-zero indices
    copyto!(v.xk, @view(x[:, v.idx]))                          # store relevant components of x for first iteration
    debias && (temp_glm = initialize_glm_object())             # Preallocated GLM variable for debiasing

    # If requested, compute initial guess for model b
    if init
        initialize_beta!(v, y, x, d, l)
        A_mul_B!(v.xb, v.zc, x, z, v.b, v.c)
    end

    # Begin 'iterative' hard thresholding algorithm
    for iter in 1:max_iter

        # notify and return current model if maximum iteration exceeded
        if iter > max_iter
            mm_iter  = iter
            tot_time = time() - start_time
            verbose && printstyled("Did not converge after $max_iter " * 
                "iterations! The run time for IHT was " * string(tot_time) *
                " seconds\n", color=:red)
            break
        end

        # save values from previous iterate and update loglikelihood
        save_prev!(v)
        logl = next_logl

        # take one IHT step in positive score direction
        (η, η_step, next_logl) = iht_one_step!(v, x, z, y, J, k, d, l, 
            logl, full_grad, max_step, est_r)

        # perform debiasing if requested
        if debias && sum(v.idx) == size(v.xk, 2)
            temp_glm = fit(GeneralizedLinearModel, v.xk, y, d, l)
            view(v.b, v.idx) .= temp_glm.pp.beta0
        end

        # track convergence
        the_norm    = max(chebyshev(v.b, v.b0), chebyshev(v.c, v.c0)) #max(abs(x - y))
        scaled_norm = the_norm / (max(norm(v.b0, Inf), norm(v.c0, Inf)) + 1.0)
        converged   = scaled_norm < tol
        verbose && println("Iteration $iter: tol = $scaled_norm")
        if converged
        # if iter > 1 && abs(next_logl - logl) < tol * (abs(logl) + 1.0)
            tot_time = time() - start_time
            mm_iter  = iter
            break
        end
    end

    return ggIHTResults(tot_time, next_logl, mm_iter, v.b, v.c, J, k, v.group, d)
end

fit(y::AbstractVector{T}, x::AbstractMatrix{T}; kwargs...) where T = 
    fit(y, x, ones(length(y)); kwargs...)

"""
Performs 1 iteration of the IHT algorithm, backtracking a maximum of 5 times.
We allow loglikelihood to potentially decrease to avoid bad boundary cases.
"""
function iht_one_step!(v::IHTVariable{T}, x::AbstractMatrix,
    z::AbstractVecOrMat{T}, y::AbstractVector{T}, J::Int, k::Union{Int, 
    Vector{Int}}, d::UnivariateDistribution, l::Link, old_logl::T, 
    full_grad::AbstractVector{T}, nstep::Int, est_r::Union{Symbol, Nothing},
    ) where {T <: Float}

    # first calculate step size 
    η = iht_stepsize(v, z, d, l)

    # update b and c by taking gradient step v.b = P_k(β + ηv) where v is the score direction
    _iht_gradstep(v, η, J, k, full_grad)

    # update the linear predictors `xb` with the new proposed b, and use that to compute the mean
    update_xb!(v, x, z)
    update_μ!(v.μ, v.xb + v.zc, l)

    # update r
    if !isnothing(est_r)
        old_r = d.r
        d = mle_for_r(y, v.μ, d.r, est_r)
    end

    # calculate current loglikelihood with the new computed xb and zc
    new_logl = loglikelihood(d, y, v.μ)

    η_step = 0
    while _iht_backtrack_(new_logl, old_logl, η_step, nstep)

        # stephalving
        η /= 2

        # recompute gradient step
        copyto!(v.b, v.b0)
        copyto!(v.c, v.c0)
        _iht_gradstep(v, η, J, k, full_grad)

        # recompute η = xb, μ = g(η), and loglikelihood to see if we're now increasing
        update_xb!(v, x, z)
        update_μ!(v.μ, v.xb + v.zc, l)
        isnothing(est_r) || (d = mle_for_r(y, v.μ, d.r, est_r))
        new_logl = loglikelihood(d, y, v.μ)

        # increment the counter
        η_step += 1
    end

    # compute score with the new mean μ
    score!(d, l, v, x, z, y)

    # check for finiteness before moving to the next iteration
    isnan(new_logl) && throw(error("Loglikelihood function is NaN, aborting..."))
    isinf(new_logl) && throw(error("Loglikelihood function is Inf, aborting..."))
    isinf(η) && throw(error("step size not finite! it is $η and max gradient is " * string(maximum(v.gk)) * "!!\n"))

    return η::T, η_step::Int, new_logl::T, d::UnivariateDistribution
end
