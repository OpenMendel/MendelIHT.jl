"""
This function runs Iterative Hard Thresholding for GWAS data `x`, response `y`, and non-genetic
covariates `z`. 

+ `d` is a distribution in the exponential family we are fitting to
+ `l` stores the link function. 
+ `k` is the maximum number of predictors per group. (i.e. a sparsity constraint)
+ `J` is the maximum number of groups. When J = 1, we get regular IHT algorithm
+ `xbm` is a BitArray version of `x` needed for linear algebras. It's possible to set scale=false for xbm, especially when rare SNPs exist
+ `use_maf` indicates whether we want to scale the projection with minor allele frequencies (see paper)
+ `tol` is used to track convergence
+ `max_iter` is the maximum IHT iteration for a model to converge. Defaults to 200, or 100 for cross validation
+ `max_step` is the maximum number of backtracking. Since l0 norm is not convex, we have no ascent guarantee
+ `debias` is boolean indicating whether we debias at each iteration (see paper)
+ `show_info` boolean indicating whether we want to print results. Should set to false for multithread/multicore computing
+ `init` boolean indicating whether we want to initialize β to sensible values through fitting. This is not efficient yet. 
"""
function L0_reg(
    x         :: SnpArray,
    xbm       :: SnpBitMatrix,
    z         :: AbstractMatrix{T},
    y         :: AbstractVector{T},
    J         :: Int,
    k         :: Int,
    d         :: UnivariateDistribution,
    l         :: Link;
    use_maf   :: Bool = false, 
    tol       :: T = 1e-4,     # tolerance for tracking convergence
    max_iter  :: Int = 200,    # maximum IHT iterations
    max_step  :: Int = 3,      # maximum backtracking for each iteration
    debias    :: Bool = false,
    show_info :: Bool = false,
    init      :: Bool = false,
) where {T <: Float}

    #start timer
    start_time = time()

    # first handle errors
    @assert J >= 0        "Value of J (max number of groups) must be nonnegative!\n"
    @assert k >= 0        "Value of k (max predictors per group) must be nonnegative!\n"
    @assert max_iter >= 0 "Value of max_iter must be nonnegative!\n"
    @assert max_step >= 0 "Value of max_step must be nonnegative!\n"
    @assert tol > eps(T)  "Value of global tol must exceed machine precision!\n"
    checky(y, d) # make sure response data y is in the form we need it to be 

    # initialize constants
    mm_iter     = 0                 # number of iterations 
    tot_time    = 0.0               # compute time *within* L0_reg
    next_logl   = oftype(tol,-Inf)  # loglikelihood
    the_norm    = 0.0               # norm(b - b0)
    scaled_norm = 0.0               # the_norm / (norm(b0) + 1)
    η_step      = 0                 # counts number of backtracking steps for η
    converged   = false             # scaled_norm < tol?

    # Initialize variables. 
    v = IHTVariables(x, z, y, J, k)                            # Placeholder variable for cleaner code
    debias && (temp_glm = initialize_glm_object())             # Preallocated GLM variable for debiasing
    full_grad = zeros(size(x, 2) + size(z, 2))                 # Preallocated vector for efficiency
    init_iht_indices!(v, xbm, z, y, d, l, J, k, full_grad)     # initialize non-zero indices
    copyto!(v.xk, @view(x[:, v.idx]), center=true, scale=true) # store relevant components of x
    
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
            show_info && printstyled("Did not converge!!!!! The run time for IHT was " * string(tot_time) * "seconds and model size was" * string(k), color=:red)
            break
        end

        # save values from previous iterate and update loglikelihood
        save_prev!(v)
        logl = next_logl

        # take one IHT step in positive score direction
        (η, η_step, next_logl) = iht_one_step!(v, x, xbm, z, y, J, k, d, l, logl, full_grad, iter, max_step, use_maf)

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
            tot_time = time() - start_time
            mm_iter  = iter
            break
        end
    end

    return ggIHTResults(tot_time, next_logl, mm_iter, v.b, v.c, J, k, v.group)
end #function L0_reg

"""
This function performs 1 iteration of the IHT algorithm. 
"""
function iht_one_step!(v::IHTVariable{T}, x::SnpArray, xbm::SnpBitMatrix, z::AbstractMatrix{T}, 
    y::AbstractVector{T}, J::Int, k::Int, d::UnivariateDistribution, l::Link, old_logl::T, 
    full_grad::AbstractVector{T}, iter::Int, nstep::Int, use_maf::Bool) where {T <: Float}

    # first calculate step size 
    η = iht_stepsize(v, z, d)

    # update b and c by taking gradient step v.b = P_k(β + ηv) where v is the score direction
    _iht_gradstep(v, η, J, k, full_grad, use_maf)

    # update the linear predictors `xb` with the new proposed b, and use that to compute the mean
    update_xb!(v, x, z)
    update_μ!(v.μ, v.xb + v.zc, l)

    # calculate current loglikelihood with the new computed xb and zc
    new_logl = loglikelihood(d, y, v.μ)

    η_step = 0
    while _iht_backtrack_(new_logl, old_logl, η_step, nstep)

        # stephalving
        η /= 2

        # recompute gradient step
        copyto!(v.b, v.b0)
        copyto!(v.c, v.c0)
        _iht_gradstep(v, η, J, k, full_grad, use_maf)

        # recompute η = xb, μ = g^{-1}(η), and loglikelihood to see if we're now increasing
        update_xb!(v, x, z)
        update_μ!(v.μ, v.xb + v.zc, l)
        new_logl = loglikelihood(d, y, v.μ)

        # increment the counter
        η_step += 1
    end

    # compute score with the new mean μ
    score!(v, xbm, z, y)

    # check for finiteness before moving to the next iteration
    isnan(new_logl) && throw(error("Loglikelihood function is NaN, aborting..."))
    isinf(new_logl) && throw(error("Loglikelihood function is Inf, aborting..."))
    isinf(η) && throw(error("step size not finite! it is $η and max df is " * string(maximum(v.gk)) * "!!\n"))

    return η::T, η_step::Int, new_logl::T
end #function iht_one_step
