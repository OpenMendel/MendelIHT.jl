"""
    iht_poisson! calculates the IHT step β+ = P_k(β + μv) and returns step size (μ), 
    number of times line search was done (μ_step), and the new loglikelihood (new_logl).

- `v` is a IHTVariable that holds many variables in the current and previous iteration. 
- `x` is the SNP matrix
- `z` is the non genetic covraiates. The grand mean (i.e. intercept) is the first column of this
- `y` is the response (phenotype) vector
- `J` is the maximum number of groups IHT should keep. When J = 1, we run normal IHT.  
- `k` is the maximum number of predictors per group. 
- `iter` is an integer storing the number of full iht steps taken (i.e. negative gradient + projection)
- `nstep` number of maximum backtracking. 
"""
function iht_poisson!(
    v         :: IHTVariable{T},
    x         :: SnpArray,
    z         :: AbstractMatrix{T},
    y         :: AbstractVector{T},
    J         :: Int,
    k         :: Int,
    glm       :: String,
    old_logl  :: T,
    temp_vec  :: Vector{T},
    iter      :: Int,
    nstep     :: Int
) where {T <: Float}

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
    μ = _poisson_stepsize(v, z)

    # update b and c by taking gradient step v.b = P_k(β + μv) where v is the score direction
    _iht_gradstep(v, μ, J, k, temp_vec)
    # println(v.b[v.idx])

    # make necessary resizing since grad step might include/exclude non-genetic covariates
    check_covariate_supp!(v) 

    # update xb and zc with the new computed b and c, truncating bad guesses to avoid overflow
    copyto!(v.xk, @view(x[:, v.idx]), center=true, scale=true)
    A_mul_B!(v.xb, v.zc, v.xk, z, view(v.b, v.idx), v.c)
    clamp!(v.xb, -20, 20)
    clamp!(v.zc, -20, 20)

    # calculate current loglikelihood with the new computed xb and zc
    new_logl = compute_logl(v, y, glm)

    μ_step = 0
    while _poisson_backtrack(v, new_logl, old_logl, μ_step, nstep)

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
        A_mul_B!(v.xb, v.zc, v.xk, z, view(v.b, v.idx), v.c)
        clamp!(v.xb, -20, 20)
        clamp!(v.zc, -20, 20)

        # compute new loglikelihood again to see if we're now increasing
        new_logl = compute_logl(v, y, glm)

        # increment the counter
        μ_step += 1
    end

    isnan(new_logl) && throw(error("Loglikelihood is NaN, aborting..."))
    isinf(new_logl) && throw(error("Loglikelihood is Inf, aborting..."))
    isinf(μ) && throw(error("step size weird! it is $μ and max df is " * string(maximum(v.gk)) * "!!\n"))

    return μ::T, μ_step::Int, new_logl::T
end

"""
    L0_poisson_reg runs IHT on GWAS data for a given sparsity constraint k and J where
    the response vector is count data. 

- `v` is a IHTVariable that holds many variables in the current and previous iteration. 
- `x` is the SNP matrix
- `z` is the non genetic covraiates. The grand mean (i.e. intercept) is the first column of this
- `y` is the response (phenotype) vector
- `J` is the maximum number of groups IHT should keep. When J = 1, we run the usual IHT.  
- `k` is the maximum number of predictors per group. 
- `glm` is the generalized linear model option. Can be either logistic, poisson, normal = default
- `use_maf` is a boolean. If true, IHT will scale each SNP using their minor allele frequency.
- `mask_n` is a bit masking vector of booleans. It is used in cross-validation where certain samples are excluded from the model
- `tol` and `max_iter` and `max_step` is self-explanatory.
"""
function L0_poisson_reg(
    x         :: SnpArray,
    xbm       :: SnpBitMatrix,
    z         :: AbstractMatrix{T},
    y         :: AbstractVector{T},
    J         :: Int,
    k         :: Int;
    use_maf   :: Bool = false,
    glm       :: String = "poisson",
    tol       :: T = 1e-4,
    max_iter  :: Int = 200,
    max_step  :: Int = 3,
    temp_vec  :: Vector{T} = zeros(size(x, 2) + size(z, 2)),
    debias    :: Bool = true,
    scale     :: Bool = false,
    convg     :: Bool = true, #use kevin's convergence criteria
    show_info :: Bool = true,
    init      :: Bool = false,
    true_beta :: Vector{T} = ones(size(x, 2)), # temporary, should be removed soon
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
    mm_iter   = 0                 # number of iterations of L0_reg
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

    # make the bit matrix 
    # xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=scale);

    #initialiaze model and compute xb
    if init
        initialize_beta!(v, y, x, glm)
        A_mul_B!(v.xb, v.zc, xbm, z, v.b, v.c)
        clamp!(v.xb, -20, 20)
        clamp!(v.zc, -20, 20)
    end

    #compute the gradient
    update_df!(glm, v, xbm, z, y)

    if show_info
        temp_df = DataFrame(true_β = true_beta, gradient = v.df, initial_β = v.b)
        @show sort(temp_df, rev=true, by=abs)[1:2k, :]
    end

    for mm_iter = 1:max_iter
        # save values from previous iterate and update loglikelihood
        save_prev!(v)
        logl = next_logl

        #take gradient step, return loglikelihood and step size
        (μ, μ_step, next_logl) = iht_poisson!(v, x, z, y, J, k, glm, logl, temp_vec, mm_iter, max_step)

        #perform debiasing (after v.b have been updated via iht_logistic) whenever possible
        if debias && sum(v.idx) == size(v.xk, 2)
            (β, obj) = regress(v.xk, y, glm)
            if !all(β .≈ 0)
                view(v.b, v.idx) .= β
            end
        end

        #print information about current iteration
        show_info && println("iter = " * string(mm_iter) * ", loglikelihood = " * string(round(next_logl, sigdigits=5)) * ", step size = " * string(round(μ, sigdigits=5)) * ", backtrack = " * string(μ_step))
        show_info && μ_step == 3 && @info("backtracked 3 times! loglikelihood not guaranteed to increase")
        if show_info
            temp_df = DataFrame(current_β = v.b0, ηv = μ.*v.df, true_β = true_beta, grad=v.df)
            @show sort(temp_df, rev=true, by=abs)[1:2k, :]
        end

        # update score (gradient) and p vector for next iteration using stepsize μ 
        update_df!(glm, v, xbm, z, y)

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
            show_info && printstyled("Did not converge!!!!! The run time for IHT was " * string(tot_time) * " seconds and model size was " * string(k), color=:red)
            return ggIHTResults(tot_time, next_logl, mm_iter, v.b, v.c, J, k, v.group)
        end
    end
end #function L0_poisson_reg

# """
# The code below uses omega backtracking. It performs like shit.
# """
# function iht_poisson2!(
#     v         :: IHTVariable{T},
#     x         :: SnpArray,
#     z         :: AbstractMatrix{T},
#     y         :: AbstractVector{T},
#     J         :: Int,
#     k         :: Int,
#     glm       :: String,
#     old_logl  :: T,
#     temp_vec  :: Vector{T},
#     iter      :: Int,
#     nstep     :: Int
# ) where {T <: Float}

#     #initialize indices (idx and idc) based on biggest entries of v.df and v.df2
#     if iter == 1
#         init_iht_indices!(v, J, k, temp_vec = temp_vec)
#         check_covariate_supp!(v) # make necessary resizing
#     end

#     # store relevant components of x
#     if !isequal(v.idx, v.idx0) || iter < 2
#         copyto!(v.xk, @view(x[:, v.idx]), center=true, scale=true)
#     end

#     # calculate step size 
#     μ = _poisson_stepsize(v, z)

#     # update b and c by taking gradient step v.b = P_k(β + μv) where v is the score direction
#     _iht_gradstep(v, μ, J, k, temp_vec)

#     # make necessary resizing since grad step might include/exclude non-genetic covariates
#     check_covariate_supp!(v) 

#     # update xb and zc with the new computed b and c, truncating bad guesses to avoid overflow
#     copyto!(v.xk, @view(x[:, v.idx]), center=true, scale=true)
#     A_mul_B!(v.xb, v.zc, v.xk, z, view(v.b, v.idx), v.c)
#     clamp!(v.xb, -20, 20)
#     clamp!(v.zc, -20, 20)

#     # calculate omega
#     @. v.p = exp(v.xb + v.zc)
#     ω_top, ω_bot = _iht_omega_poisson(v)

#     μ_step = 0
#     while _poisson_backtrack2(v, ω_top, ω_bot, μ, μ_step, nstep)

#         # stephalving
#         μ /= 2

#         # recompute gradient step
#         copyto!(v.b, v.b0)
#         copyto!(v.c, v.c0)
#         _iht_gradstep(v, μ, J, k, temp_vec)

#         # make necessary resizing since grad step might include/exclude non-genetic covariates
#         check_covariate_supp!(v) 

#         # recompute xb
#         copyto!(v.xk, @view(x[:, v.idx]), center=true, scale=true)
#         A_mul_B!(v.xb, v.zc, v.xk, z, view(v.b, v.idx), v.c)
#         clamp!(v.xb, -20, 20)
#         clamp!(v.zc, -20, 20)

#         # calculate omega
#         @. v.p = exp(v.xb + v.zc)
#         ω_top, ω_bot = _iht_omega_poisson(v)

#         # increment the counter
#         μ_step += 1
#     end

#     isinf(μ) && throw(error("step size weird! it is $μ and max df is " * string(maximum(v.gk)) * "!!\n", color=:red))

#     return μ::T, μ_step::Int
# end

# """
# The code below uses omega backtracking. It performs like shit.
# """
# function L0_poisson_reg2(
#     x         :: SnpArray,
#     z         :: AbstractMatrix{T},
#     y         :: AbstractVector{T},
#     J         :: Int,
#     k         :: Int;
#     use_maf   :: Bool = false,
#     glm       :: String = "normal",
#     tol       :: T = 1e-4,
#     max_iter  :: Int = 1000,
#     max_step  :: Int = 3,
#     temp_vec  :: Vector{T} = zeros(size(x, 2) + size(z, 2)),
#     debias    :: Bool = true,
#     scale     :: Bool = false,
#     convg     :: Bool = true, #use kevin's convergence criteria
#     show_info :: Bool = true,
#     init      :: Bool = false,
#     true_beta :: Vector{T} = ones(size(x, 2)), # temporary, should be removed soon
# ) where {T <: Float}

#     start_time = time()

#     # first handle errors
#     @assert J >= 0        "Value of J (max number of groups) must be nonnegative!\n"
#     @assert k >= 0        "Value of k (max predictors per group) must be nonnegative!\n"
#     @assert max_iter >= 0 "Value of max_iter must be nonnegative!\n"
#     @assert max_step >= 0 "Value of max_step must be nonnegative!\n"
#     @assert tol > eps(T)  "Value of global tol must exceed machine precision!\n"

#     # make sure response data is in the form we need it to be 
#     check_y_content(y, glm)

#     # initialize return values
#     mm_iter   = 0                 # number of iterations of L0_reg
#     next_logl = oftype(tol,-Inf)  # loglikelihood

#     # initialize floats
#     the_norm    = 0.0             # norm(b - b0)
#     scaled_norm = 0.0             # the_norm / (norm(b0) + 1)

#     # initialize integers
#     mu_step = 0                   # counts number of backtracking steps for mu

#     # initialize booleans
#     converged = false             # scaled_norm < tol?
    
#     # Begin IHT calculations
#     v = IHTVariables(x, z, y, J, k)

#     # make the bit matrix 
#     xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=scale);

#     #initialiaze model and compute xb
#     if init
#         initialize_beta!(v, y, x, glm)
#         A_mul_B!(v.xb, v.zc, xbm, z, v.b, v.c)
#         clamp!(v.xb, -20, 20)
#         clamp!(v.zc, -20, 20)
#     end

#     #compute the gradient
#     update_df2!(glm, v, xbm, z, y)
#     @. v.p = exp(v.xb + v.zc)

#     if show_info
#         temp_df = DataFrame(true_β = true_beta, gradient = v.df, initial_β = v.b)
#         @show sort(temp_df, rev=true, by=abs)[1:2k, :]
#     end

#     for mm_iter = 1:max_iter
#         # save values from previous iterate and update loglikelihood
#         save_prev!(v)
#         logl = next_logl

#         #take gradient step, return loglikelihood and step size
#         (μ, μ_step) = iht_poisson2!(v, x, z, y, J, k, glm, logl, temp_vec, mm_iter, max_step)

#         #perform debiasing (after v.b have been updated via iht_logistic) whenever possible
#         if debias && sum(v.idx) == size(v.xk, 2)
#             (β, obj) = regress(v.xk, y, glm)
#             if !all(β .≈ 0)
#                 view(v.b, v.idx) .= β
#             end
#         end

#         #print information about current iteration
#         show_info && println("iter = " * string(mm_iter) * ", loglikelihood = " * string(round(next_logl, sigdigits=5)) * ", step size = " * string(round(μ, sigdigits=5)) * ", backtrack = " * string(μ_step))
#         show_info && μ_step == 3 && @info("backtracked 3 times! loglikelihood not guaranteed to increase")
#         if show_info
#             temp_df = DataFrame(current_β = v.b0, ηv = μ.*v.df, true_β = true_beta, grad=v.df)
#             @show sort(temp_df, rev=true, by=abs)[1:2k, :]
#         end

#         # update score (gradient) and p vector for next iteration using stepsize μ 
#         update_df2!(glm, v, xbm, z, y)

#         # calculate current loglikelihood with the new computed xb and zc
#         next_logl = compute_logl(v, y, glm)

#         # track convergence using kevin or ken's converegence criteria
#         if convg
#             the_norm    = max(chebyshev(v.b, v.b0), chebyshev(v.c, v.c0)) #max(abs(x - y))
#             scaled_norm = the_norm / (max(norm(v.b0, Inf), norm(v.c0, Inf)) + 1.0)
#             converged   = scaled_norm < tol
#         else
#             converged = abs(next_logl - logl) < tol
#         end

#         if converged && mm_iter > 1
#             tot_time = time() - start_time
#             return ggIHTResults(tot_time, next_logl, mm_iter, v.b, v.c, J, k, v.group)
#         end

#         if mm_iter == max_iter
#             tot_time = time() - start_time
#             show_info && printstyled("Did not converge!!!!! The run time for IHT was " * string(tot_time) * " seconds and model size was " * string(k), color=:red)
#             return ggIHTResults(tot_time, next_logl, mm_iter, v.b, v.c, J, k, v.group)
#         end
#     end
# end #function L0_poisson_reg