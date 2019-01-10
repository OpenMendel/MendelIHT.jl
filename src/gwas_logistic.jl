"""
    iht_logistic! calculates the IHT step β+ = P_k(β - μv) and returns step size (μ), 
    number of times line search was done (μ_step), and the new loglikelihood (new_logl).

- `v` is a IHTVariable that holds many variables in the current and previous iteration. 
- `x` is the SNP matrix
- `z` is the non genetic covraiates. The grand mean (i.e. intercept) is the first column of this
- `y` is the response (phenotype) vector
- `J` is the maximum number of groups IHT should keep. When J = 1, we run normal IHT.  
- `k` is the maximum number of predictors per group. 
- `mean_vec` is a vector storing the mean of SNP frequency. Needed for standarization on-the-fly
- `std_vec` is a vector storing the inverse standard deviation of each SNP. Needed for standarization on-the-fly
- `storage` is a vector of matrices preallocated for (snpmatrix)-(dense vector) multiplication. This is needed for better garbage collection.
- `iter` is an integer storing the number of full iht steps taken (i.e. negative gradient + projection)
- `nstep` number of maximum backtracking. 
"""
function iht_logistic!(
    v         :: IHTVariable{T},
    x         :: SnpArray,
    z         :: AbstractMatrix{T},
    y         :: AbstractVector{T},
    J         :: Int,
    k         :: Int,
    glm       :: String,
    old_logl  :: T,
    temp_vec  :: AbstractVector{T},
    iter      :: Int,
    nstep     :: Int
) where {T <: Float}

    #initialize indices (idx and idc) based on biggest entries of v.df and v.df2
    if iter == 1
        init_iht_indices!(v, J, k, temp_vec = temp_vec)
        check_covariate_supp!(v) # make necessary resizing
    end

    # store relevant components of x
    if (!isequal(v.idx, v.idx0) && !isequal(v.idc, v.idc0)) || iter < 2
        copyto!(v.xk, @view(x[:, v.idx]), center=true, scale=true)
    end

    # calculate step size 
    μ = _logistic_stepsize(v, x, z)

    # update b and c by taking gradient step v.b = P_k(β + μv) where v is the score direction
    _iht_gradstep(v, μ, J, k, temp_vec)

    # perform debiasing (i.e. fit b on its support)
    # if all(v.idx .== v.idx0)
    #     xk = convert(Matrix{Float64}, v.xk)
    #     (estimate, obj) = regress(xk, y, glm)
    #     view(v.b, v.idx) .= estimate
    # end

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

        # perform debiasing (i.e. fit b on its support)
        # if all(v.idx .== v.idx0)
        #     xk = convert(Matrix{Float64}, v.xk)
        #     (estimate, obj) = regress(xk, y, glm)
        #     view(v.b, v.idx) .= estimate
        # end

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

    return μ::T, μ_step::Int, new_logl::T
end

"""
    L0_reg_glm runs IHT on GWAS data for a given sparsity constraint k and J. 
    GLM method (normal, logistic, poisson...etc) is specified through the optional `glm` input.

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
function L0_logistic_reg(
    v         :: IHTVariable, 
    x         :: SnpArray,
    z         :: AbstractMatrix{T},
    y         :: AbstractVector{T},
    J         :: Int,
    k         :: Int;
    use_maf   :: Bool = false,
    glm       :: String = "normal",
    mask_n    :: BitArray = trues(size(y)),
    tol       :: T = 1e-4,
    max_iter  :: Int = 200, # up from 100 for sometimes weighting takes more
    max_step  :: Int = 50,
    temp_vec  :: Vector{T} = zeros(size(x, 2) + size(z, 2))
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
    tot_time  = 0.0               # compute time *within* L0_reg
    next_logl = oftype(tol,-Inf)  # loglikelihood

    # initialize floats
    the_norm    = 0.0             # norm(b - b0)
    scaled_norm = 0.0             # the_norm / (norm(b0) + 1)
    μ           = 0.0             # Landweber step size, 0 < tau < 2/rho_max^2

    # initialize integers
    mu_step = 0                   # counts number of backtracking steps for mu

    # initialize booleans
    converged = false             # scaled_norm < tol?

    # initialize empty vectors to facilitate garbage collection in (snpmatrix)-(vector) computation
    # store = Vector{Vector{T}}(3)
    # store[1] = zeros(T, size(v.df))  # length p 
    # store[2] = zeros(T, size(v.xgk)) # length n
    # store[3] = zeros(T, size(v.gk))  # length J * k

    #weight snps based on maf or other user defined weights
    # if use_maf
    #     maf = deepcopy(mean_vec) 
    #     my_snpMAF, my_snpweights = calculate_snp_weights(x,y,k,v,use_maf,maf)
    #     hold_std_vec = deepcopy(std_vec)
    #     Base.A_mul_B!(std_vec, diagm(hold_std_vec), my_snpweights[1,:])
    # end
    
    # Begin IHT calculations
    fill!(v.xb, 0.0)       #initialize β = 0 vector, so Xβ = 0
    # copy!(v.r, y)          #redisual = y-Xβ-zc = y since initially β = c = 0
    # v.r[mask_n .== 0] .= 0 #bit masking, for cross validation only

    # Calculate the score
    x_bitmatrix = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true);
    update_df!(glm, v, x_bitmatrix, z, y)

    for mm_iter = 1:max_iter
        # save values from previous iterate and update loglikelihood
        save_prev!(v)
        logl = next_logl

        #calculate the step size μ and check loglikelihood is not NaN or Inf
        (μ, μ_step, next_logl) = iht_logistic!(v, x, z, y, J, k, glm, logl, temp_vec, mm_iter, max_step)
        !isnan(next_logl) || throw(error("Loglikelihood function is NaN, aborting..."))
        !isinf(next_logl) || throw(error("Loglikelihood function is Inf, aborting..."))

        # iht! gives us an updated x*b. Use it to recompute residuals and gradient
        # v.r .= y .- v.xb .- v.zc 
        # v.r[mask_n .== 0] .= 0 #bit masking, used for cross validation

        # update score (gradient) and p vector using stepsize μ 
        update_df!(glm, v, x_bitmatrix, z, y)

        # track convergence
        the_norm    = max(chebyshev(v.b, v.b0), chebyshev(v.c, v.c0)) #max(abs(x - y))
        scaled_norm = the_norm / (max(norm(v.b0, Inf), norm(v.c0, Inf)) + 1.0)
        converged   = scaled_norm < tol

        # info("current iteration is " * string(mm_iter) * ", loglikelihood is " * string(next_logl) * " and scaled norm is " * string(scaled_norm))

        if converged && mm_iter > 1
            tot_time = time() - start_time
            return gIHTResults(tot_time, next_logl, mm_iter, v.b, v.c, J, k, v.group)
        end

        if mm_iter == max_iter
            tot_time = time() - start_time
            println("Did not converge!!!!! The run time for IHT was " * string(mm_time) * "seconds and model size was" * string(k))
            return gIHTResults(tot_time, next_logl, mm_iter, v.b, v.c, J, k, v.group)
        end
    end
end #function L0_logistic_reg

