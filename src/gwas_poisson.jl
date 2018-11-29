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

function iht_poisson!(
    v         :: IHTVariable{T},
    x         :: SnpLike{2},
    z         :: Matrix{T},
    y         :: Vector{T},
    J         :: Int,
    k         :: Int,
    mean_vec  :: Vector{T},
    std_vec   :: Vector{T},
    glm       :: String,
    old_logl  :: T,
    storage   :: Vector{Vector{T}},
    temp_vec  :: Vector{T},
    iter      :: Int,
    nstep     :: Int
) where {T <: Float}

    #initialize indices (idx and idc) based on biggest entries of v.df and v.df2
    if iter == 1
        init_iht_indices!(v, J, k, temp_vec = temp_vec)
        check_covariate_supp!(v, storage) # make necessary resizing
    end

    # calculate step size 
    μ = _poisson_stepsize(v, x, z, mean_vec, std_vec)

    # update b and c by taking gradient step v.b = P_k(β + μv) where v is the score direction
    _iht_gradstep(v, μ, J, k, temp_vec)
    # println(v.b[v.idx])

    # perform debiasing (i.e. fit b on its support)
    # if all(v.idx .== v.idx0)
    #     xk = convert(Matrix{Float64}, v.xk)
    #     (estimate, obj) = regress(xk, y, glm)
    #     view(v.b, v.idx) .= estimate
    # end

    # make necessary resizing since grad step might include/exclude non-genetic covariates
    check_covariate_supp!(v, storage) 

    # update xb and zc with the new computed b and c
    v.xk .= view(x, :, v.idx)
    A_mul_B!(v.xb, v.zc, v.xk, z, view(v.b, v.idx), v.c, view(mean_vec, v.idx), view(std_vec, v.idx), storage)

    # to prevent overflow of loglikelihood, we don't allow xβ to have entries larger than 20
    # for i in eachindex(v.xb)
    #     if v.xb[i] > 20.0
    #         v.xb[i] = 20.0
    #     end
    # end

    # calculate current loglikelihood with the new computed xb and zc
    new_logl = compute_logl(v, x, z, y, glm, mean_vec, std_vec, storage)

    # println(new_logl)
    # println(μ)
    # println(maximum(v.b))
    # println(maximum(v.c))
    # println(sum(v.df))
    # println(sum(v.df2))
    # println(sum(v.b))
    # println(sum(v.c))

    μ_step = 0
    while _poisson_backtrack(v, new_logl, old_logl, μ_step, nstep)

        # stephalving
        μ /= 2

        # recompute gradient step
        copy!(v.b, v.b0)
        copy!(v.c, v.c0)
        _iht_gradstep(v, μ, J, k, temp_vec)

        # perform debiasing (i.e. fit b on its support)
        # if all(v.idx .== v.idx0)
        #     xk = convert(Matrix{Float64}, v.xk)
        #     (estimate, obj) = regress(xk, y, glm)
        #     view(v.b, v.idx) .= estimate
        # end

        # make necessary resizing since grad step might include/exclude non-genetic covariates
        check_covariate_supp!(v, storage) 

        # recompute xb
        v.xk .= view(x, :, v.idx)
        A_mul_B!(v.xb, v.zc, v.xk, z, view(v.b, v.idx), v.c, view(mean_vec, v.idx), view(std_vec, v.idx), storage)

        # compute new loglikelihood again to see if we're now increasing
        new_logl = compute_logl(v, x, z, y, glm, mean_vec, std_vec, storage)

        # increment the counter
        μ_step += 1
    end

    info("performed " * string(μ_step) * " backtracking")

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
    v         :: IHTVariable, 
    x         :: SnpLike{2},
    z         :: Matrix{T},
    y         :: Vector{T},
    J         :: Int,
    k         :: Int;
    use_maf   :: Bool = false,
    glm       :: String = "normal",
    mask_n    :: BitArray = trues(size(y)),
    tol       :: T = 1e-4,
    max_iter  :: Int = 1000, # up from 100 for sometimes weighting takes more
    max_step  :: Int = 50,
    temp_vec  :: Vector{T} = zeros(size(x, 2) + size(z, 2))
) where {T <: Float}

    # start timer
    tic()

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
    mm_time   = 0.0               # compute time *within* L0_reg
    next_logl = oftype(tol,-Inf)  # loglikelihood

    # initialize floats
    the_norm    = 0.0             # norm(b - b0)
    scaled_norm = 0.0             # the_norm / (norm(b0) + 1)

    # initialize integers
    mu_step = 0                   # counts number of backtracking steps for mu

    # initialize booleans
    converged = false             # scaled_norm < tol?

    # initialize empty vectors to facilitate garbage collection in (snpmatrix)-(vector) computation
    store = Vector{Vector{T}}(3)
    store[1] = zeros(T, size(v.df))  # length p 
    store[2] = zeros(T, size(v.xgk)) # length n
    store[3] = zeros(T, size(v.gk))  # length J * k

    #initilize the intercept to the log of the sample mean to avoid overflow
    # v.c[1] = log(mean(y))

    # compute some summary statistics for our snpmatrix
    mean_vec, minor_allele, = summarize(x)
    people, snps = size(x)

    #weight snps based on maf or other user defined weights
    if use_maf
        maf = deepcopy(mean_vec) 
        my_snpMAF, my_snpweights = calculate_snp_weights(x,y,k,v,use_maf,maf)
        hold_std_vec = deepcopy(std_vec)
        Base.A_mul_B!(std_vec, diagm(hold_std_vec), my_snpweights[1,:])
    end

    #precompute mean and standard deviations for each snp. 
    update_mean!(mean_vec, minor_allele, snps)
    std_vec = std_reciprocal(x, mean_vec)
    
    #
    # Begin IHT calculations
    #
    fill!(v.xb, 0.0)       #initialize β = 0 vector, so Xβ = 0
    # copy!(v.r, y)          #redisual = y-Xβ-zc = y since initially β = c = 0
    # v.r[mask_n .== 0] .= 0 #bit masking, for cross validation only

    # Calculate the score 
    update_df!(glm, v, x, z, y, mean_vec, std_vec, store)

    for mm_iter = 1:max_iter
        # save values from previous iterate and update loglikelihood
        save_prev!(v)
        logl = next_logl

        #calculate the step size μ and check loglikelihood is not NaN or Inf
        (μ, μ_step, next_logl) = iht_poisson!(v, x, z, y, J, k, mean_vec, std_vec, glm, logl, store, temp_vec, mm_iter, max_step)
        
        if isnan(next_logl) || isinf(next_logl)
            info("model size is" * string(k))
            info("current iteration is " * string(mm_iter))
            info("snpmatrix is " * string(size(x)))
            println("there are " * string(sum(v.b .== 20)) * " entries of b that is 20 or larger")
        end

        !isnan(next_logl) || throw(error("Loglikelihood function is NaN, aborting..."))
        !isinf(next_logl) || throw(error("Loglikelihood function is Inf, aborting..."))

        # iht! gives us an updated x*b. Use it to recompute residuals and gradient
        # v.r .= y .- v.xb .- v.zc 
        # v.r[mask_n .== 0] .= 0 #bit masking, used for cross validation

        # update score (gradient) and p vector using stepsize μ 
        update_df!(glm, v, x, z, y, mean_vec, std_vec, store)

        # track convergence
        the_norm    = max(chebyshev(v.b, v.b0), chebyshev(v.c, v.c0)) #max(abs(x - y))
        scaled_norm = the_norm / (max(norm(v.b0, Inf), norm(v.c0, Inf)) + 1.0)
        converged   = scaled_norm < tol

        info("current loglikelihood is " * string(logl))

        if converged && mm_iter > 1
            mm_time = toq()   # stop time
            return gIHTResults(mm_time, next_logl, mm_iter, v.b, v.c, J, k, v.group)
        end

        if mm_iter == max_iter
            mm_time = toq() # stop time
            println("Did not converge!!!!! The run time for IHT was " * string(mm_time) * "seconds")
            return gIHTResults(mm_time, next_logl, mm_iter, v.b, v.c, J, k, v.group)
        end
    end
end #function L0_poisson_reg

