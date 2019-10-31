"""
    L0_reg(x, xbm, z, y, J, k, d, l, est_r)

Estimates the nuisance parameter in addition to estimating the mean using IHT. 

We alternate between estimating the mean using IHT and estimating the nuisance parameter
using maximum loglikelihood. Currently, only negative binomial regression supports estimating 
nuisance paramter, and the method of choice includes Newton or MM. This can be specified with
`est_r = :MM` or `est_r = :Newton`.

Note: Convergence criteria does not track the `r` paramter. 
"""
function L0_reg(
    x         :: SnpArray,
    xbm       :: SnpBitMatrix,
    z         :: AbstractMatrix{T},
    y         :: AbstractVector{T},
    J         :: Int,
    k         :: Union{Int, Vector{Int}},
    d         :: UnivariateDistribution,
    l         :: Link,
    est_r     :: Symbol;
    group     :: AbstractVector{Int} = Int[],
    weight    :: AbstractVector{T} = T[],
    use_maf   :: Bool = false, 
    debias    :: Bool = false,
    verbose   :: Bool = true,          # print things when model didn't converge
    init      :: Bool = false,         # not efficient. whether to initialize β to sensible values
    tol       :: T = convert(T, 1e-4), # tolerance for tracking convergence
    max_iter  :: Int = 100,            # maximum IHT iterations
    max_step  :: Int = 5               # maximum backtracking for each iteration
) where {T <: Float}

    #start timer
    start_time = time()

    # first handle errors
    @assert J >= 0        "Value of J (max number of groups) must be nonnegative!\n"
    @assert max_iter >= 0 "Value of max_iter must be nonnegative!\n"
    @assert max_step >= 0 "Value of max_step must be nonnegative!\n"
    @assert tol > eps(T)  "Value of global tol must exceed machine precision!\n"
    checky(y, d) # make sure response data y is in the form compatible with specified GLM
    check_group(k, group) # make sure sparsity parameter `k` is reasonable. 
    @assert typeof(d) <: NegativeBinomial "Only negative binomial regression currently supports nuisance parameter estimation"

    # initialize constants
    mm_iter     = 0                 # number of iterations 
    tot_time    = 0.0               # compute time *within* L0_reg
    next_logl   = oftype(tol,-Inf)  # loglikelihood
    the_norm    = 0.0               # norm(b - b0)
    scaled_norm = 0.0               # the_norm / (norm(b0) + 1)
    η_step      = 0                 # counts number of backtracking steps for η
    converged   = false             # scaled_norm < tol?

    # Initialize variables. 
    v = IHTVariables(x, z, y, J, k, group, weight)             # Placeholder variable for cleaner code
    full_grad = zeros(T, size(x, 2) + size(z, 2))              # Preallocated vector for efficiency
    init_iht_indices!(v, xbm, z, y, d, l, J, k, group)         # initialize non-zero indices
    copyto!(v.xk, @view(x[:, v.idx]), center=true, scale=true) # store relevant components of x for first iteration
    debias && (temp_glm = initialize_glm_object())             # Preallocated GLM variable for debiasing

    # If requested, compute initial guess for model b
    if init
        initialize_beta!(v, y, x, d, l)
        A_mul_B!(v.xb, v.zc, xbm, z, v.b, v.c)
    end

    # Begin 'iterative' hard thresholding algorithm
    for iter = 1:max_iter

        # notify and return current model if maximum iteration exceeded
        if iter >= max_iter
            mm_iter  = iter
            tot_time = time() - start_time
            verbose && printstyled("Did not converge after $max_iter iterations! The run time for IHT was " * string(tot_time) * " seconds\n", color=:red)
            break
        end

        # save values from previous iterate and update loglikelihood
        save_prev!(v)
        logl = next_logl

        # take one IHT step in positive score direction
        (η, η_step, next_logl, d) = iht_one_step!(v, x, xbm, z, y, J, k, d, l, logl, full_grad, iter, max_step, use_maf, est_r)

        # perform debiasing if requested
        if debias && sum(v.idx) == size(v.xk, 2)
            temp_glm = fit(GeneralizedLinearModel, v.xk, y, d, l)
            view(v.b, v.idx) .= temp_glm.pp.beta0
        end

        # track convergence
        the_norm    = max(chebyshev(v.b, v.b0), chebyshev(v.c, v.c0)) #max(abs(x - y))
        scaled_norm = the_norm / (max(norm(v.b0, Inf), norm(v.c0, Inf)) + 1.0)
        converged   = scaled_norm < tol
        if converged
        # if iter > 1 && abs(next_logl - logl) < tol * (abs(logl) + 1.0)
            tot_time = time() - start_time
            mm_iter  = iter
            break
        end
    end

    return ggIHTResults(tot_time, next_logl, mm_iter, v.b, v.c, J, k, v.group, d)
end #function L0_reg

"""
    L0_reg(x, z, y, J, k, d, l, est_r)

Estimates the nuisance parameter in addition to estimating the mean using IHT. `x` is a 
matrix of Float32 or Float64.

We alternate between estimating the mean using IHT and estimating the nuisance parameter
using maximum loglikelihood. Currently, only negative binomial regression supports estimating 
nuisance paramter, and the method of choice includes Newton or MM. This can be specified with
`est_r = :MM` or `est_r = :Newton`.

Note: Convergence criteria does not track the `r` paramter. 
"""
function L0_reg(
    x         :: AbstractMatrix{T},
    z         :: AbstractMatrix{T},
    y         :: AbstractVector{T},
    J         :: Int,
    k         :: Union{Int, Vector{Int}},
    d         :: UnivariateDistribution,
    l         :: Link,
    est_r     :: Symbol;
    group     :: AbstractVector{Int} = Int[],
    weight    :: AbstractVector{T} = T[],
    use_maf   :: Bool = false, 
    debias    :: Bool = false,
    verbose   :: Bool = true,            # print things when model didn't converge
    init      :: Bool = false,           # not efficient. initializes β to sensible values
    tol       :: T    = convert(T, 1e-4),# tolerance for tracking convergence
    max_iter  :: Int  = 100,             # maximum IHT iterations
    max_step  :: Int  = 5,               # maximum backtracking for each iteration
) where {T <: Float}
    
    #start timer
    start_time = time()

    # first handle errors
    @assert J >= 0        "Value of J (max number of groups) must be nonnegative!\n"
    @assert max_iter >= 0 "Value of max_iter must be nonnegative!\n"
    @assert max_step >= 0 "Value of max_step must be nonnegative!\n"
    @assert tol > eps(T)  "Value of global tol must exceed machine precision!\n"
    checky(y, d) # make sure response data y is in the form compatible with specified GLM
    check_group(k, group) # make sure sparsity parameter `k` is reasonable. 
    @assert typeof(d) <: NegativeBinomial "Only negative binomial regression currently supports nuisance parameter estimation"

    # initialize constants
    mm_iter     = 0                 # number of iterations 
    tot_time    = zero(T)           # compute time *within* L0_reg
    next_logl   = oftype(tol,-Inf)  # loglikelihood
    the_norm    = zero(T)           # norm(b - b0)
    scaled_norm = zero(T)           # the_norm / (norm(b0) + 1)
    η_step      = 0                 # counts number of backtracking steps for η
    converged   = false             # scaled_norm < tol?

    # Initialize variables. 
    v = IHTVariables(x, z, y, J, k, group, weight)     # Placeholder variable for cleaner code
    full_grad = zeros(T, size(x, 2) + size(z, 2))      # Preallocated vector for efficiency
    init_iht_indices!(v, x, z, y, d, l, J, k, group)   # initialize non-zero indices
    copyto!(v.xk, @view(x[:, v.idx]))                  # store relevant components of x for first iteration
    debias && (temp_glm = initialize_glm_object())     # Preallocated GLM variable for debiasing

    # If requested, compute initial guess for model b
    if init
        initialize_beta!(v, y, x, d, l)
        A_mul_B!(v.xb, v.zc, x, z, v.b, v.c)
    end

    # Begin 'iterative' hard thresholding algorithm
    for iter = 1:max_iter

        # notify and return current model if maximum iteration exceeded
        if iter >= max_iter
            mm_iter  = iter
            tot_time = time() - start_time
            verbose && printstyled("Did not converge after $max_iter iterations! The run time for IHT was " * string(tot_time) * " seconds\n", color=:red)
            break
        end

        # save values from previous iterate and update loglikelihood
        save_prev!(v)
        logl = next_logl

        # take one IHT step in positive score direction
        (η, η_step, next_logl, d) = iht_one_step!(v, x, z, y, J, k, d, l, logl, full_grad, iter, max_step, use_maf, est_r)
        
        # perform debiasing if requested
        if debias && sum(v.idx) == size(v.xk, 2)
            temp_glm = fit(GeneralizedLinearModel, v.xk, y, d, l)
            view(v.b, v.idx) .= temp_glm.pp.beta0
        end

        # track convergence
        the_norm    = max(chebyshev(v.b, v.b0), chebyshev(v.c, v.c0)) #max(abs(x - y))
        scaled_norm = the_norm / (max(norm(v.b0, Inf), norm(v.c0, Inf)) + 1.0)
        converged   = scaled_norm < tol
        if converged
        # if iter > 1 && abs(next_logl - logl) < tol * (abs(logl) + 1.0)
            tot_time = time() - start_time
            mm_iter  = iter
            break
        end
    end

    return ggIHTResults(tot_time, next_logl, mm_iter, v.b, v.c, J, k, v.group, d)
end #function L0_reg

"""
This function performs 1 iteration of the IHT algorithm, backtracking a maximum of 5 times.
While IHT can strictly increase loglikelihood, we still allow it to potentially decrease 
to avoid bad boundary cases.
"""
function iht_one_step!(v::IHTVariable{T}, x::SnpArray, xbm::SnpBitMatrix, z::AbstractMatrix{T}, 
    y::AbstractVector{T}, J::Int, k::Union{Int, Vector{Int}}, d::UnivariateDistribution, 
    l::Link, old_logl::T, full_grad::AbstractVector{T}, iter::Int, nstep::Int, use_maf::Bool, 
    est_r::Symbol) where {T <: Float}

    # first calculate step size 
    η = iht_stepsize(v, z, d, l)

    # update b and c by taking gradient step v.b = P_k(β + ηv) where v is the score direction
    _iht_gradstep(v, η, J, k, full_grad)

    # update the linear predictors `xb` with the new proposed b, and use that to compute the mean
    update_xb!(v, x, z)
    update_μ!(v.μ, v.xb + v.zc, l)
   
    # update r
    old_r = d.r
    d = mle_for_r(y, v.μ, d.r, est_r)

    # calculate current loglikelihood with the new computed xb and zc
    new_logl = loglikelihood(d, y, v.μ)

    η_step = 0
    while _iht_backtrack_(new_logl, old_logl, η_step, nstep)

        # stephalving
        η /= 2

        # if NegativeBinomial, reset r to previous r
        d = NegativeBinomial(old_r, 0.5)

        # recompute gradient step
        copyto!(v.b, v.b0)
        copyto!(v.c, v.c0)
        _iht_gradstep(v, η, J, k, full_grad)

        # recompute η = xb, μ = g(η), nuisance paramter, and loglikelihood to see if we're now increasing
        update_xb!(v, x, z)
        update_μ!(v.μ, v.xb + v.zc, l)
        d = mle_for_r(y, v.μ, d.r, est_r)
        new_logl = loglikelihood(d, y, v.μ)

        # increment the counter
        η_step += 1
    end

    # compute score with the new mean μ
    score!(d, l, v, xbm, z, y)

    # check for finiteness before moving to the next iteration
    isnan(new_logl) && throw(error("Loglikelihood function is NaN, aborting..."))
    isinf(new_logl) && throw(error("Loglikelihood function is Inf, aborting..."))
    isinf(η) && throw(error("step size not finite! it is $η and max gradient is " * string(maximum(v.gk)) * "!!\n"))

    return η::T, η_step::Int, new_logl::T, d::UnivariateDistribution
end #function iht_one_step

"""
Performs 1 iteration of the IHT algorithm given a general matrix of floating point numbers. 
"""
function iht_one_step!(v::IHTVariable{T}, x::AbstractMatrix, z::AbstractMatrix, 
    y::AbstractVector{T}, J::Int, k::Union{Int, Vector{Int}}, d::UnivariateDistribution, l::Link, old_logl::T, 
    full_grad::AbstractVector{T}, iter::Int, nstep::Int, use_maf::Bool, 
    est_r::Symbol) where {T <: Float}

    # first calculate step size 
    η = iht_stepsize(v, z, d, l)

    # update b and c by taking gradient step v.b = P_k(β + ηv) where v is the score direction
    _iht_gradstep(v, η, J, k, full_grad)

    # update the linear predictors `xb` with the new proposed b, and use that to compute the mean
    update_xb!(v, x, z)
    update_μ!(v.μ, v.xb + v.zc, l)
    
    # update r
    old_r = d.r
    d = mle_for_r(y, v.μ, d.r, est_r)

    # calculate current loglikelihood with the new computed xb and zc
    new_logl = loglikelihood(d, y, v.μ)

    η_step = 0
    while _iht_backtrack_(new_logl, old_logl, η_step, nstep)

        # stephalving
        η /= 2

        # reset r 
        d = NegativeBinomial(old_r, 0.5)

        # recompute gradient step
        copyto!(v.b, v.b0)
        copyto!(v.c, v.c0)
        _iht_gradstep(v, η, J, k, full_grad)

        # recompute η = xb, μ = g(η), nuisance paramter, and loglikelihood to see if we're now increasing
        update_xb!(v, x, z)
        update_μ!(v.μ, v.xb + v.zc, l)
        d = mle_for_r(y, v.μ, d.r, est_r)
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
end #function iht_one_step
