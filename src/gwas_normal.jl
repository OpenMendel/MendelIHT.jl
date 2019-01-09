"""
    iht! calculates the IHT step β+ = P_k(β - μ ∇f(β)) and returns step size (μ), and number of times line search was done (μ_step).

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
function iht!(
    v         :: IHTVariable{T},
    x         :: SnpArray,
    z         :: Matrix{T},
    y         :: Vector{T},
    J         :: Int,
    k         :: Int,
    # mean_vec  :: Vector{T},
    # std_vec   :: Vector{T},
    # storage   :: Vector{Vector{T}},
    temp_vec  :: Vector{T},
    iter      :: Int,
    nstep     :: Int
) where {T <: Float}

    # compute indices of nonzeroes in beta
    # v.idx .= v.b .!= 0
    # v.idc .= v.c .!= 0

    #initialize indices (idx and idc) based on biggest entries of v.df and v.df2
    if iter == 1
        init_iht_indices!(v, J, k, temp_vec = temp_vec)
        check_covariate_supp!(v) # make necessary resizing
    end

    # store relevant columns of x.
    if (!isequal(v.idx, v.idx0) && !isequal(v.idc, v.idc0)) || iter < 2
        copyto!(v.xk, @view(x[:, v.idx]), center=true, scale=true)
    end

    # calculate step size and take gradient (score) step based on type of regression
    μ = _iht_stepsize(v, z)

    # take the gradient step v.b = P_k(β + μv) where v is the score direction
    _iht_gradstep(v, μ, J, k, temp_vec)

    # make necessary resizing since grad step might include/exclude non-genetic covariates
    check_covariate_supp!(v)

    # update xb (needed to calculate ω to determine line search criteria)
    copyto!(v.xk, @view(x[:, v.idx]), center=true, scale=true)
    A_mul_B!(v.xb, v.zc, v.xk, z, @view(v.b[v.idx]), v.c)

    # calculate omega
    ω_top, ω_bot = _iht_omega(v)

    # backtrack until mu < omega and until support stabilizes
    μ_step = 0
    while _normal_backtrack(v, ω_top, ω_bot, μ, μ_step, nstep)

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

        # calculate omega
        ω_top, ω_bot = _iht_omega(v)

        # increment the counter
        μ_step += 1
    end

    return μ::T, μ_step::Int
end

"""
    L0_reg run IHT on GWAS data for a given sparsity constraint k and J. 
- `v` is a IHTVariable that holds many variables in the current and previous iteration. 
- `x` is the SNP matrix
- `z` is the non genetic covraiates. The grand mean (i.e. intercept) is the first column of this
- `y` is the response (phenotype) vector
- `J` is the maximum number of groups IHT should keep. When J = 1, we run normal IHT.  
- `k` is the maximum number of predictors per group. 
- `use_maf` is a boolean. If true, IHT will scale each SNP using their minor allele frequency.
- `mask_n` is a bit masking vector of booleans. It is used in cross-validation where certain samples are excluded from the model
- `tol` and `max_iter` and `max_step` is self-explanatory.
"""
function L0_reg(
    v        :: IHTVariable, 
    x        :: SnpArray,
    z        :: Matrix{T},
    y        :: Vector{T},
    J        :: Int,
    k        :: Int;
    use_maf  :: Bool = false,
    mask_n   :: BitArray = trues(size(y)),
    tol      :: T = 1e-4,
    max_iter :: Int = 200, # up from 100 for sometimes weighting takes more
    max_step :: Int = 50,
    temp_vec :: Vector{T} = zeros(size(x, 2) + size(z, 2))
) where {T <: Float}

    start_time = time()

    # first handle errors
    @assert J >= 0        "Value of J (max number of groups) must be nonnegative!\n"
    @assert k >= 0        "Value of k (max predictors per group) must be nonnegative!\n"
    @assert max_iter >= 0 "Value of max_iter must be nonnegative!\n"
    @assert max_step >= 0 "Value of max_step must be nonnegative!\n"
    @assert tol > eps(T)  "Value of global tol must exceed machine precision!\n"

    # initialize return values
    mm_iter   = 0                 # number of iterations of L0_reg
    tot_time  = 0.0               # compute time *within* L0_reg
    next_loss = oftype(tol,Inf)   # loss function value

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
    #     maf = maf(x)
    #     my_snpMAF, my_snpweights = calculate_snp_weights(x,y,k,v,use_maf,maf)
    #     hold_std_vec = deepcopy(std_vec)
    #     Base.A_mul_B!(std_vec, diagm(hold_std_vec), my_snpweights[1,:])
    # end

    # Begin IHT calculations
    fill!(v.xb, 0.0)       #initialize β = 0 vector, so Xβ = 0
    copyto!(v.r, y)        #redisual = y-Xβ-zc = y since initially β = c = 0
    v.r[mask_n .== 0] .= 0 #bit masking, for cross validation only

    # Calculate the gradient v.df = -[X' ; Z']'(y - Xβ - Zc) = [X' ; Z'](-1*(Y-Xb - Zc))
    x_bitmatrix = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true);
    At_mul_B!(v.df, v.df2, x_bitmatrix, z, v.r, v.r)

    for mm_iter = 1:max_iter
        # save values from previous iterate and update loss
        save_prev!(v)
        loss = next_loss

        #calculate the step size μ.
        (μ, μ_step) = iht!(v, x, z, y, J, k, temp_vec, mm_iter, max_step)

        # iht! gives us an updated x*b. Use it to recompute residuals and gradient
        v.r .= y .- v.xb .- v.zc 
        v.r[mask_n .== 0] .= 0 #bit masking, used for cross validation

        # update v.df = [ X'(y - Xβ - zc) ; Z'(y - Xβ - zc) ]
        At_mul_B!(v.df, v.df2, x_bitmatrix, z, v.r, v.r)

        # update loss, objective, gradient, and check objective is not NaN or Inf
        next_loss = sum(abs2, v.r) / 2
        !isnan(next_loss) || throw(error("Objective function is NaN, aborting..."))
        !isinf(next_loss) || throw(error("Objective function is Inf, aborting..."))

        # track convergence
        the_norm    = max(chebyshev(v.b, v.b0), chebyshev(v.c, v.c0)) #max(abs(x - y))
        scaled_norm = the_norm / (max(norm(v.b0, Inf), norm(v.c0, Inf)) + 1.0)
        converged   = scaled_norm < tol

        if converged && mm_iter > 1
            tot_time = time() - start_time
            return gIHTResults(tot_time, next_loss, mm_iter, v.b, v.c, J, k, v.group)
        end

        if mm_iter == max_iter
            tot_time = time() - start_time
            throw(error("Did not converge!!!!! The run time for IHT was " *
                string(tot_time) * " seconds"))
        end
    end
end #function L0_reg
