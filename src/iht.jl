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
    check_y_content(y, glm)

    # initialize return values
    mm_iter   = 0                 # number of iterations of L0_logistic_reg
    tot_time  = 0.0               # compute time *within* L0_logistic_reg
    next_logl = oftype(tol,-Inf)  # loglikelihood

    # initialize floats
    the_norm    = 0.0             # norm(b - b0)
    scaled_norm = 0.0             # the_norm / (norm(b0) + 1)

    # initialize integers
    mu_step = 0                   # counts number of backtracking steps for mu

    # initialize booleans
    converged = false             # scaled_norm < tol?
    
    # Begin IHT calculations
    v = IHTVariables(x, z, y, J, k)

    #initiliaze model and compute xb
    # if init
    #     initialize_beta!(v, y, x, d, l)
    #     A_mul_B!(v.xb, v.zc, xbm, z, v.b, v.c)
    # end

    # Calculate the score
    update_df!(v, xbm, z, y, l)

    println("reached here!")
    return fff

    for mm_iter = 1:max_iter
        # save values from previous iterate and update loglikelihood
        save_prev!(v)
        logl = next_logl

        #calculate the step size μ and check loglikelihood is not NaN or Inf
        (μ, μ_step, next_logl) = iht_logistic!(v, x, z, y, J, k, glm, logl, temp_vec, mm_iter, max_step)

        #perform debiasing (after v.b have been updated via iht_logistic) whenever possible
        if debias && sum(v.idx) == size(v.xk, 2)
            (β, obj) = regress(v.xk, y, glm)
            if !all(β .≈ 0)
                view(v.b, v.idx) .= β
            end
            # test_result = GLM.glm(v.xk, y, Binomial(), LogitLink()) #GLM package doesn't work idk why
            # test_result = test_result.pp.beta0
        end

        #print information about current iteration
        show_info && @info("iter = " * string(mm_iter) * ", loglikelihood = " * string(next_logl) * ", step size = " * string(μ) * ", backtrack = " * string(μ_step))
        if show_info
            temp_df = DataFrame(β = v.b, gradient = v.df)
            @show sort(temp_df, rev=true, by=abs)[1:2k, :]
        end

        # update score (gradient) and p vector using stepsize μ 
        update_df!(glm, v, xbm, z, y)

        return ff

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
    J::Int, k::Int, d::UnivariateDistribution, link::Link, old_logl::T, 
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
    μ = _logistic_stepsize(v, z)

    # update b and c by taking gradient step v.b = P_k(β + μv) where v is the score direction
    _iht_gradstep(v, μ, J, k, temp_vec)

    # make necessary resizing since grad step might include/exclude non-genetic covariates
    check_covariate_supp!(v) 

    # update xb and zc with the new computed b and c
    copyto!(v.xk, @view(x[:, v.idx]), center=true, scale=true)
    A_mul_B!(v.xb, v.zc, v.xk, z, view(v.b, v.idx), v.c)

    # calculate current loglikelihood with the new computed xb and zc
    new_logl = compute_logl(v, y, glm)

    μ_step = 0
    while _logistic_backtrack(new_logl, old_logl, μ_step, nstep)

        # stephalving
        μ /= 2

        # recompute gradient step
        copyto!(v.b, v.b0)
        copyto!(v.c, v.c0)
        _iht_gradstep(v, μ, J, k, temp_vec)

        # make necessary resizing since grad step might include/exclude non-genetic covariates
        check_covariate_supp!(v) 

        # recompute xb
        copyto!(v.xk, @view(x[:, v.idx]), center=true, scale=true)
        A_mul_B!(v.xb, v.zc, v.xk, z, @view(v.b[v.idx]), v.c)

        # compute new loglikelihood again to see if we're now increasing
        new_logl = compute_logl(v, y, glm)

        # increment the counter
        μ_step += 1
    end

    isnan(new_logl) && throw(error("Loglikelihood function is NaN, aborting..."))
    isinf(new_logl) && throw(error("Loglikelihood function is Inf, aborting..."))
    isinf(μ) && throw(error("step size weird! it is $μ and max df is " * string(maximum(v.gk)) * "!!\n"))

    return μ::T, μ_step::Int, new_logl::T
end
