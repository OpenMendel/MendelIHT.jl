function L0_reg(
    x         :: SnpArray,
    xbm       :: SnpBitMatrix,
    z         :: AbstractMatrix{T},
    y         :: AbstractVector{T},
    J         :: Int,
    k         :: Int,
    d         :: UnivariateDistribution,
    l         :: Link = canonicallink(d);
    use_maf   :: Bool = false,
    tol       :: T = 1e-4,
    max_iter  :: Int = 200,
    max_step  :: Int = 3,
    debias    :: Bool = false,
    show_info :: Bool = false,
    init      :: Bool = false,
    convg     :: Bool = false,
    temp_vec  :: Vector{T} = zeros(size(x, 2) + size(z, 2)),
) where {T <: Float}

    start_time = time()

    # first handle errors
    @assert J >= 0        "Value of J (max number of groups) must be nonnegative!\n"
    @assert k >= 0        "Value of k (max predictors per group) must be nonnegative!\n"
    @assert max_iter >= 0 "Value of max_iter must be nonnegative!\n"
    @assert max_step >= 0 "Value of max_step must be nonnegative!\n"
    @assert tol > eps(T)  "Value of global tol must exceed machine precision!\n"

    # make sure response data is in the form we need it to be 
    checky(y, d)

    # initialize return values
    mm_iter   = 0                 # number of iterations of L0_logistic_reg
    tot_time  = 0.0               # compute time *within* L0_logistic_reg
    next_logl = oftype(tol,-Inf)  # loglikelihood

    # initialize floats
    the_norm    = 0.0             # norm(b - b0)
    scaled_norm = 0.0             # the_norm / (norm(b0) + 1)

    # initialize integers
    η_step = 0                   # counts number of backtracking steps for η

    # initialize booleans
    converged = false             # scaled_norm < tol?
    
    # Begin IHT calculations
    v = IHTVariables(x, z, y, J, k)

    #initiliaze model and compute xb
    if init
        initialize_beta!(v, y, x, d, l)
        A_mul_B!(v.xb, v.zc, xbm, z, v.b, v.c)
    end

    # update mean vector, residual, then use them to compute score (gradient)
    update_mean!(v.μ, v.xb .+ v.zc, l)
    score!(v, xbm, z, y, d, l)

    for mm_iter = 1:max_iter
        # save values from previous iterate and update loglikelihood
        save_prev!(v)
        logl = next_logl

        #calculate the step size η and check loglikelihood is not NaN or Inf
        (η, η_step, next_logl) = iht!(v, x, z, y, J, k, d, l, logl, temp_vec, mm_iter, max_step)

        #perform debiasing (after v.b have been updated via iht!) whenever possible
        if debias && sum(v.idx) == size(v.xk, 2)
            test_result = fit(GeneralizedLinearModel, v.xk, y, d, l)
            all(test_result.pp.beta0 .≈ 0) || (view(v.b, v.idx) .= test_result.pp.beta0)
        end

        #print information about current iteration
        show_info && @info("iter = " * string(mm_iter) * ", loglikelihood = " * string(next_logl) * ", step size = " * string(η) * ", backtrack = " * string(η_step))
        if show_info
            temp_df = DataFrame(β = v.b, gradient = v.df)
            @show sort(temp_df, rev=true, by=abs)[1:2k, :]
        end

        # compute score, where the mean have been updated in iht!
        score!(v, xbm, z, y, d, l)

        # track convergence using kevin or ken's converegence criteria
        if convg
            the_norm    = max(chebyshev(v.b, v.b0), chebyshev(v.c, v.c0)) #max(abs(x - y))
            scaled_norm = the_norm / (max(norm(v.b0, Inf), norm(v.c0, Inf)) + 1.0)
            converged   = scaled_norm < tol
        else
            converged = abs(next_logl - logl) < tol
        end

        if converged && mm_iter > 1
            tot_time = time() - start_time
            return ggIHTResults(tot_time, next_logl, mm_iter, v.b, v.c, J, k, v.group)
        end

        if mm_iter == max_iter
            tot_time = time() - start_time
            printstyled("Did not converge!!!!! The run time for IHT was " * string(tot_time) * "seconds and model size was" * string(k), color=:red)
            return ggIHTResults(tot_time, next_logl, mm_iter, v.b, v.c, J, k, v.group)
        end
    end
end #function L0_reg

function iht!(v::IHTVariable{T}, x::SnpArray, z::AbstractMatrix{T}, y::AbstractVector{T},
    J::Int, k::Int, d::UnivariateDistribution, l::Link, old_logl::T, 
    temp_vec::AbstractVector{T}, iter::Int, nstep::Int) where {T <: Float}

    #initialize indices (idx and idc) based on biggest entries of v.df and v.df2
    if iter == 1
        init_iht_indices!(v, J, k, temp_vec = temp_vec)
        check_covariate_supp!(v) # make necessary resizing
    end

    # store relevant components of x
    if !isequal(v.idx, v.idx0) || iter < 2
        copyto!(v.xk, @view(x[:, v.idx]), center=true, scale=true)
    end

    # calculate step size 
    η = iht_stepsize(v, z, d)

    # update b and c by taking gradient step v.b = P_k(β + ηv) where v is the score direction
    _iht_gradstep(v, η, J, k, temp_vec)

    # make necessary resizing since grad step might include/exclude non-genetic covariates
    check_covariate_supp!(v) 

    # update xb and zc with the new computed b and c, clamping because might overflow for poisson
    copyto!(v.xk, @view(x[:, v.idx]), center=true, scale=true)
    A_mul_B!(v.xb, v.zc, v.xk, z, view(v.b, v.idx), v.c)
    clamp!(v.xb, -30, 30)
    clamp!(v.zc, -30, 30)

    # calculate current loglikelihood with the new computed xb and zc
    update_mean!(v.μ, v.xb .+ v.zc, l)
    new_logl = loglikelihood(d, y, v.μ)

    η_step = 0
    while _iht_backtrack_(new_logl, old_logl, η_step, nstep)

        # stephalving
        η /= 2

        # recompute gradient step
        copyto!(v.b, v.b0)
        copyto!(v.c, v.c0)
        _iht_gradstep(v, η, J, k, temp_vec)

        # make necessary resizing since grad step might include/exclude non-genetic covariates
        check_covariate_supp!(v) 

        # recompute xb
        copyto!(v.xk, @view(x[:, v.idx]), center=true, scale=true)
        A_mul_B!(v.xb, v.zc, v.xk, z, @view(v.b[v.idx]), v.c)
        clamp!(v.xb, -30, 30)
        clamp!(v.zc, -30, 30)

        # compute new loglikelihood again to see if we're now increasing
        update_mean!(v.μ, v.xb .+ v.zc, l)
        new_logl = loglikelihood(d, y, v.μ)

        # increment the counter
        η_step += 1
    end

    isnan(new_logl) && throw(error("Loglikelihood function is NaN, aborting..."))
    isinf(new_logl) && throw(error("Loglikelihood function is Inf, aborting..."))
    isinf(η) && throw(error("step size not finite! it is $η and max df is " * string(maximum(v.gk)) * "!!\n"))

    return η::T, η_step::Int, new_logl::T
end
