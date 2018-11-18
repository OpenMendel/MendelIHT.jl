"""
This is the wrapper function for the Iterative Hard Thresholding analysis option in Open Mendel.
"""
function MendelIHT(control_file = ""; args...)
    const MENDEL_IHT_VERSION :: VersionNumber = v"0.5.0"
    #
    # Print the logo. Store the initial directory.
    #
    print(" \n \n")
    println("     Welcome to OpenMendel's")
    println("      IHT analysis option")
    println("        version ", MENDEL_IHT_VERSION)
    print(" \n \n")
    println("Reading the data.\n")
    initial_directory = pwd()
    #
    # The user specifies the analysis to perform via a set of keywords.
    # Start the keywords at their default values.
    #
    keyword = set_keyword_defaults!(Dict{AbstractString, Any}())
    #
    # Define some keywords unique to this analysis option.
    #
    keyword["predictors"] = 0
    keyword["max_groups"] = 1
    keyword["group_membership"] = ""
    keyword["maf_weights"] = ""
    keyword["pw_algorithm_value"] = 1.0     # not user defined at this time
    keyword["non_genetic_covariates"] = ""
    keyword["run_cross_validation"] = false
    keyword["model_sizes"] = ""
    keyword["cv_folds"] = ""
    #
    # Process the run-time user-specified keywords that will control the analysis.
    # This will also initialize the random number generator.
    #
    process_keywords!(keyword, control_file, args)
    @assert typeof(keyword["max_groups"]) == Int "Number of groups must be an integer. Set as 1 to run normal IHT"
    @assert typeof(keyword["predictors"]) == Int "Sparsity constraint must be positive integer"
    @assert 0 <= keyword["predictors"]           "Need positive number of predictors per group"
    #
    # Import genotype/non-genetic/phenotype data
    #
    info("Reading in data")
    snpmatrix = SnpArray(keyword["plink_input_basename"]) #requires .bim .bed .fam files
    phenotype = readdlm(keyword["plink_input_basename"] * ".fam", header = false)[:, 6]
    non_genetic_cov = ones(size(snpmatrix, 1), 1) #defaults to just the intercept
    if keyword["non_genetic_covariates"] != ""
        non_genetic_cov = readdlm(keyword["non_genetic_covariates"], keyword["field_separator"], Float64)
    end
    #
    # Determine what weighting (if any) the user specified for each predictors
    #
    keyword["maf_weights"] == "maf" ? maf_weights = true : maf_weights = false
    #
    # Determine the maximum number of groups and max number of predictors per group_membership.
    # Defaults to only 1 group containing 10 predictors
    #
    J = 1
    k = 10
    if keyword["max_groups"] != 0
        J = keyword["max_groups"]
    end
    if keyword["predictors"] != 0 
        k = keyword["predictors"]
    end
    @assert k >= 1 "Number of predictors must be positive integer"
    @assert J >= 1 "Number of predictors must be positive integer"
    #
    # Execute the specified analysis.
    #
    if keyword["run_cross_validation"]
        #
        # Find the model sizes the user wants. Defaults to 1~10
        #
        path = collect(1:10)
        if keyword["model_sizes"] != ""
            path = [parse(Int, ss) for ss in split(keyword["model_sizes"], ',')]
            @assert typeof(path) == Vector{Int} "Cannot parse input paths!"
        end
        #
        # Specify how many folds of cross validation the user wants. Defaults to 5
        #
        num_folds = 5
        if keyword["cv_folds"] != "" 
            num_folds = keyword["cv_folds"]
            @assert typeof(num_folds) == Int "Please provide positive integer value for the number of folds for cross validation"
            @assert num_folds >= 1           "Please provide positive integer value for the number of folds for cross validation"
        end
        info("Running " * string(num_folds) * "-fold cross validation on the following model sizes:\n" * keyword["model_sizes"] * ".\nIgnoring keyword predictors.")
        folds = rand(1:num_folds, size(snpmatrix, 1))
        return cv_iht(snpmatrix, non_genetic_cov, phenotype, 1, path, folds, num_folds, use_maf = maf_weights)

    elseif keyword["model_sizes"] != ""
        path = [parse(Int, ss) for ss in split(keyword["model_sizes"], ',')]
        info("Running the following model sizes: " * string(path))
        @assert typeof(path) == Vector{Int} "Cannot parse input paths!"
        #
        # Compute the various models and associated errors
        #
        return iht_path_threaded(snpmatrix, non_genetic_cov, phenotype, J, path, use_maf = maf_weights)
    else
        #
        # Define variables for group membership, max number of predictors for each group, and max number of groups
        # If no group_membership file is provided, defaults every predictor to the same group
        #
        v = IHTVariables(snpmatrix, non_genetic_cov, phenotype, J, k)
        if keyword["group_membership"] != ""
            v.group = vec(readdlm(keyword["group_membership"], Int64))
        end
        #
        # Determine the type of analysis
        #
        keyword["analysis_option"] == "" ? glm = "normal" : glm = lowercase(keyword["analysis_option"])
        #
        # Run IHT
        #
        info("Running " * string(glm) * " IHT for model size k = $k and groups J = $J") 
        if glm == "normal"
            return L0_reg(v, snpmatrix, non_genetic_cov, phenotype, J, k, use_maf = maf_weights)
        elseif glm == "logistic"
            return L0_logistic_reg(v, snpmatrix, non_genetic_cov, phenotype, J, k, use_maf = maf_weights, glm = glm)
        end
    end
    #
    # Finish up by closing, and thus flushing, any output files.
    # Return to the initial directory.
    #
    close(keyword["output_unit"])
    cd(initial_directory)
    return nothing
end #function MendelIHT

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
    x         :: SnpLike{2},
    z         :: Matrix{T},
    y         :: Vector{T},
    J         :: Int,
    k         :: Int,
    mean_vec  :: Vector{T},
    std_vec   :: Vector{T},
    storage   :: Vector{Vector{T}},
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
        check_covariate_supp!(v, storage) # make necessary resizing
    end

    # store relevant columns of x.
    if (!isequal(v.idx, v.idx0) && !isequal(v.idc, v.idc0)) || iter < 2
        copy!(v.xk, view(x, :, v.idx))
    end

    # calculate step size and take gradient (score) step based on type of regression
    μ = _iht_stepsize(v, z, mean_vec, std_vec, storage)

    # take the gradient step v.b = P_k(β + μv) where v is the score direction
    _iht_gradstep(v, μ, J, k, temp_vec)

    # make necessary resizing since grad step might include/exclude non-genetic covariates
    check_covariate_supp!(v, storage) 

    # update xb (needed to calculate ω to determine line search criteria)
    v.xk .= view(x, :, v.idx)
    A_mul_B!(v.xb, v.zc, v.xk, z, view(v.b, v.idx), v.c, view(mean_vec, v.idx), view(std_vec, v.idx), storage)

    # calculate omega
    ω_top, ω_bot = _iht_omega(v)

    # backtrack until mu < omega and until support stabilizes
    μ_step = 0
    while _iht_backtrack(v, ω_top, ω_bot, μ, μ_step, nstep)

        # stephalving
        μ /= 2

        # recompute gradient step
        copy!(v.b, v.b0)
        copy!(v.c, v.c0)
        _iht_gradstep(v, μ, J, k, temp_vec)

        # make necessary resizing since grad step might include/exclude non-genetic covariates
        check_covariate_supp!(v, storage) 

        # recompute xb
        v.xk .= view(x, :, v.idx)
        A_mul_B!(v.xb, v.zc, v.xk, z, view(v.b, v.idx), v.c, view(mean_vec, v.idx), view(std_vec, v.idx), storage)

        # calculate omega
        ω_top, ω_bot = _iht_omega(v)

        # increment the counter
        μ_step += 1
    end

    #println("μ = " * string(μ) * ". ω_top, ω_bot = " * string(ω_top) * ", " * string(ω_bot))

    return μ::T, μ_step::Int
end

function iht_logistic!(
    v         :: IHTVariable{T},
    x         :: SnpLike{2},
    z         :: Matrix{T},
    y         :: Vector{T},
    J         :: Int,
    k         :: Int,
    mean_vec  :: Vector{T},
    std_vec   :: Vector{T},
    storage   :: Vector{Vector{T}},
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
        check_covariate_supp!(v, storage) # make necessary resizing
    end

    # store relevant columns of x.
    if (!isequal(v.idx, v.idx0) && !isequal(v.idc, v.idc0)) || iter < 2
        copy!(v.xk, view(x, :, v.idx))
    end

    # calculate step size and take gradient (score) step based on type of regression
    μ = _logistic_stepsize(v, z, mean_vec, std_vec, storage)

    # take the gradient step v.b = P_k(β + μv) where v is the score direction
    _iht_gradstep(v, μ, J, k, temp_vec)

    # make necessary resizing since grad step might include/exclude non-genetic covariates
    check_covariate_supp!(v, storage) 

    # update xb (needed to calculate ω to determine line search criteria)
    v.xk .= view(x, :, v.idx)
    A_mul_B!(v.xb, v.zc, v.xk, z, view(v.b, v.idx), v.c, view(mean_vec, v.idx), view(std_vec, v.idx), storage)

    # calculate omega
    ω_top, ω_bot = _iht_omega(v)

    # backtrack until mu < omega and until support stabilizes
    μ_step = 0
    while _iht_backtrack(v, ω_top, ω_bot, μ, μ_step, nstep)

        # stephalving
        μ /= 2

        # recompute gradient step
        copy!(v.b, v.b0)
        copy!(v.c, v.c0)
        _iht_gradstep(v, μ, J, k, temp_vec)

        # make necessary resizing since grad step might include/exclude non-genetic covariates
        check_covariate_supp!(v, storage) 

        # recompute xb
        v.xk .= view(x, :, v.idx)
        A_mul_B!(v.xb, v.zc, v.xk, z, view(v.b, v.idx), v.c, view(mean_vec, v.idx), view(std_vec, v.idx), storage)

        # calculate omega
        ω_top, ω_bot = _iht_omega(v)

        # increment the counter
        μ_step += 1
    end

    println("μ = " * string(μ) * ". ω_top, ω_bot = " * string(ω_top) * ", " * string(ω_bot))

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
    x        :: SnpLike{2},
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

    # start timer
    tic()

    # first handle errors
    @assert J >= 0        "Value of J (max number of groups) must be nonnegative!\n"
    @assert k >= 0        "Value of k (max predictors per group) must be nonnegative!\n"
    @assert max_iter >= 0 "Value of max_iter must be nonnegative!\n"
    @assert max_step >= 0 "Value of max_step must be nonnegative!\n"
    @assert tol > eps(T)  "Value of global tol must exceed machine precision!\n"

    # initialize return values
    mm_iter   = 0                 # number of iterations of L0_reg
    mm_time   = 0.0               # compute time *within* L0_reg
    next_loss = oftype(tol,Inf)   # loss function value

    # initialize floats
    current_obj = oftype(tol,Inf) # tracks previous objective function value
    the_norm    = 0.0             # norm(b - b0)
    scaled_norm = 0.0             # the_norm / (norm(b0) + 1)
    μ           = 0.0             # Landweber step size, 0 < tau < 2/rho_max^2

    # initialize integers
    mu_step = 0                   # counts number of backtracking steps for mu

    # initialize booleans
    converged = false             # scaled_norm < tol?

    # initialize empty vectors to facilitate garbage collection in (snpmatrix)-(vector) computation
    store = Vector{Vector{T}}(3)
    store[1] = zeros(T, size(v.df))  # length p 
    store[2] = zeros(T, size(v.xgk)) # length n
    store[3] = zeros(T, size(v.gk))  # length J * k

    # compute some summary statistics for our snpmatrix
    maf, minor_allele, = summarize(x)
    people, snps = size(x)
    mean_vec = deepcopy(maf) # Gordon wants maf below
    
    #precompute mean and standard deviations for each snp. Note that (1) the mean is
    #given by 2 * maf, and (2) based on which allele is the minor allele, might need to do
    #2.0 - the maf for the mean vector.
    for i in 1:snps
        minor_allele[i] ? mean_vec[i] = 2.0 - 2.0mean_vec[i] : mean_vec[i] = 2.0mean_vec[i]
    end
    std_vec = std_reciprocal(x, mean_vec)

    #weight snps based on maf or other user defined weights
    if use_maf
        my_snpMAF, my_snpweights = calculate_snp_weights(x,y,k,v,use_maf,maf)
        hold_std_vec = deepcopy(std_vec)
        Base.A_mul_B!(std_vec, diagm(hold_std_vec), my_snpweights[1,:])
    end
    #
    # Begin IHT calculations
    #
    # if sum(v.idx) + sum(v.idc) == 0
        fill!(v.xb, 0.0)       #initialize β = 0 vector, so Xβ = 0
        copy!(v.r, y)          #redisual = y-Xβ-zc = y since initially β = c = 0
        v.r[mask_n .== 0] .= 0 #bit masking, for cross validation only
    # else
    #     SnpArrays.A_mul_B!(v.xb, x, v.b, mean_vec, std_vec)
    #     BLAS.A_mul_B!(v.zc, z, v.c)
    #     v.r .= y .- v.xb .- v.zc
    #     v.r[mask_n .== 0] .= 0
    # end

    # Calculate the gradient v.df = -[X' ; Z']'(y - Xβ - Zc) = [X' ; Z'](-1*(Y-Xb - Zc))
    At_mul_B!(v.df, v.df2, x, z, v.r, v.r, mean_vec, std_vec, store)

    for mm_iter = 1:max_iter
        # save values from previous iterate and update loss
        save_prev!(v)
        loss = next_loss

        #calculate the step size μ.
        (μ, μ_step) = iht!(v, x, z, y, J, k, mean_vec, std_vec, store, temp_vec, mm_iter, max_step)

        # iht! gives us an updated x*b. Use it to recompute residuals and gradient
        v.r .= y .- v.xb .- v.zc 
        v.r[mask_n .== 0] .= 0 #bit masking, used for cross validation

        # update v.df = [ X'(y - Xβ - zc) ; Z'(y - Xβ - zc) ]
        At_mul_B!(v.df, v.df2, x, z, v.r, v.r, mean_vec, std_vec, store)

        # update loss, objective, gradient, and check objective is not NaN or Inf
        next_loss = sum(abs2, v.r) / 2
        !isnan(next_loss) || throw(error("Objective function is NaN, aborting..."))
        !isinf(next_loss) || throw(error("Objective function is Inf, aborting..."))

        # track convergence
        the_norm    = max(chebyshev(v.b, v.b0), chebyshev(v.c, v.c0)) #max(abs(x - y))
        scaled_norm = the_norm / (max(norm(v.b0, Inf), norm(v.c0, Inf)) + 1.0)
        converged   = scaled_norm < tol

        if converged && mm_iter > 1
            mm_time = toq()   # stop time
            return gIHTResults(mm_time, next_loss, mm_iter, v.b, v.c, J, k, v.group)
        end

        if mm_iter == max_iter
            mm_time = toq() # stop time
            throw(error("Did not converge!!!!! The run time for IHT was " *
                string(mm_time) * "seconds"))
        end
    end
end #function L0_reg

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
- `glm_scale` is a constant that some glm methods (e.g. logistic and poisson) would need to compute the information martix
- `use_maf` is a boolean. If true, IHT will scale each SNP using their minor allele frequency.
- `mask_n` is a bit masking vector of booleans. It is used in cross-validation where certain samples are excluded from the model
- `tol` and `max_iter` and `max_step` is self-explanatory.
"""
function L0_logistic_reg(
    v         :: IHTVariable, 
    x         :: SnpLike{2},
    z         :: Matrix{T},
    y         :: Vector{T},
    J         :: Int,
    k         :: Int;
    use_maf   :: Bool = false,
    glm       :: String = "normal",
    glm_scale :: T = 0.0,
    mask_n    :: BitArray = trues(size(y)),
    tol       :: T = 1e-4,
    max_iter  :: Int = 200, # up from 100 for sometimes weighting takes more
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
    next_loss = oftype(tol,Inf)   # loss function value

    # initialize floats
    current_obj = oftype(tol,Inf) # tracks previous objective function value
    the_norm    = 0.0             # norm(b - b0)
    scaled_norm = 0.0             # the_norm / (norm(b0) + 1)
    μ           = 0.0             # Landweber step size, 0 < tau < 2/rho_max^2

    # initialize integers
    mu_step = 0                   # counts number of backtracking steps for mu

    # initialize booleans
    converged = false             # scaled_norm < tol?

    # initialize empty vectors to facilitate garbage collection in (snpmatrix)-(vector) computation
    store = Vector{Vector{T}}(3)
    store[1] = zeros(T, size(v.df))  # length p 
    store[2] = zeros(T, size(v.xgk)) # length n
    store[3] = zeros(T, size(v.gk))  # length J * k

    # compute some summary statistics for our snpmatrix
    maf, minor_allele, = summarize(x)
    people, snps = size(x)
    mean_vec = deepcopy(maf) # Gordon wants maf below
    
    #precompute mean and standard deviations for each snp. Note that (1) the mean is
    #given by 2 * maf, and (2) based on which allele is the minor allele, might need to do
    #2.0 - the maf for the mean vector.
    @inbounds @simd for i in 1:snps
        minor_allele[i] ? mean_vec[i] = 2.0 - 2.0mean_vec[i] : mean_vec[i] = 2.0mean_vec[i]
    end
    std_vec = std_reciprocal(x, mean_vec)

    #weight snps based on maf or other user defined weights
    if use_maf
        my_snpMAF, my_snpweights = calculate_snp_weights(x,y,k,v,use_maf,maf)
        hold_std_vec = deepcopy(std_vec)
        Base.A_mul_B!(std_vec, diagm(hold_std_vec), my_snpweights[1,:])
    end
    #
    # Begin IHT calculations
    #
    fill!(v.xb, 0.0)       #initialize β = 0 vector, so Xβ = 0
    copy!(v.r, y)          #redisual = y-Xβ-zc = y since initially β = c = 0
    v.r[mask_n .== 0] .= 0 #bit masking, for cross validation only

    #update the mean of glm 
    inverse_link!(v.p, x, z, v.b, v.c) 

    # Calculate the score 
    update_df!(glm, v, x, z, y, mean_vec, std_vec, store)

    for mm_iter = 1:max_iter
        # save values from previous iterate and update loss
        save_prev!(v)
        loss = next_loss

        info("current iteration is " * string(mm_iter) * " and loss is " * string(loss))

        #calculate the step size μ.
        (μ, μ_step) = iht!(v, x, z, y, J, k, mean_vec, std_vec, glm, glm_scale, store, temp_vec, mm_iter, max_step)

        # iht! gives us an updated x*b. Use it to recompute residuals and gradient
        v.r .= y .- v.xb .- v.zc 
        v.r[mask_n .== 0] .= 0 #bit masking, used for cross validation

        # update gradient (score): [v.df; v.df2] = [ X'(y - Xβ - zc) ; Z'(y - Xβ - zc) ]
        update_df!(glm, v, x, z, y, mean_vec, std_vec, store)

        # update loss, objective, gradient, and check objective is not NaN or Inf
        next_loss = sum(abs2, v.r) / 2
        !isnan(next_loss) || throw(error("Objective function is NaN, aborting..."))
        !isinf(next_loss) || throw(error("Objective function is Inf, aborting..."))

        # track convergence
        the_norm    = max(chebyshev(v.b, v.b0), chebyshev(v.c, v.c0)) #max(abs(x - y))
        scaled_norm = the_norm / (max(norm(v.b0, Inf), norm(v.c0, Inf)) + 1.0)
        converged   = scaled_norm < tol

        if converged && mm_iter > 1
            mm_time = toq()   # stop time
            return gIHTResults(mm_time, next_loss, mm_iter, v.b, v.c, J, k, v.group)
        end

        if mm_iter == max_iter
            mm_time = toq() # stop time
            throw(error("Did not converge!!!!! The run time for IHT was " *
                string(mm_time) * "seconds"))
        end
    end
end #function L0_logistic_reg

"""
This function computes and stores different models in each column of the matrix `betas` and 
matrix `cs`. 

The additional optional arguments are:
- `mask_n`, a `Bool` vector used as a bitmask for crossvalidation purposes. Defaults to a vector of trues.
"""
function iht_path(
    x        :: SnpLike{2},
    z        :: Matrix{T},
    y        :: Vector{T},
    J        :: Int64,
    path     :: DenseVector{Int};
    use_maf  :: Bool = false,
    #pids     :: Vector{Int} = procs(x),
    mask_n   :: BitArray = trues(size(y)),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    #quiet    :: Bool = true
) where {T <: Float}

    # size of problem?
    n, p = size(x)
    q = size(z, 2) #number of non-genetic covariates

    # how many models will we compute?
    nmodels = length(path)

    # also preallocate matrix to store betas and errors
    betas  = spzeros(T, p, nmodels) # a sparse matrix to store calculated models
    cs     = zeros(T, q, nmodels)   # matrix of models of non-genetic covariates

    # compute the specified paths
    @inbounds for i = 1:nmodels

        # current model size?
        k = path[i]

        #define the IHTVariable used for cleaner code
        v = IHTVariables(x, z, y, J, k)

        # now compute current model
        output = L0_reg(v, x, z, y, J, k, use_maf=use_maf, mask_n=mask_n)

        # put model into sparse matrix of betas
        betas[:, i] .= sparsevec(output.beta)
        cs[:, i] .= output.c
    end

    # return a sparsified copy of the models
    return betas, cs
end

"""
Multi-threaded version of `iht_path`. Each thread writes to a different matrix of betas
and cs, and the reduction step is to sum all these matrices. The increase in memory usage 
increases linearly with the number of paths, which is negligible as long as the number of 
paths is reasonable (e.g. less than 100). 
"""
function iht_path_threaded(
    x        :: SnpLike{2},
    z        :: Matrix{T},
    y        :: Vector{T},
    J        :: Int64,
    path     :: DenseVector{Int};
    use_maf  :: Bool = false,
    #pids     :: Vector{Int} = procs(x),
    mask_n   :: BitArray = trues(size(y)),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    #quiet    :: Bool = true

) where {T <: Float}
    
    # number of threads available?
    num_threads = Threads.nthreads()

    # size of problem?
    n, p = size(x)
    q    = size(z, 2) #number of non-genetic covariates

    # how many models will we compute?
    nmodels = length(path)

    # Initialize vector of matrices to hold a separate matrix for each thread to access. This makes everything thread-safe
    betas = [zeros(p, nmodels) for i in 1:num_threads]
    cs    = [zeros(q, nmodels) for i in 1:num_threads]

    # compute the specified paths
    @inbounds Threads.@threads for i = 1:nmodels

        # current thread?
        cur_thread = Threads.threadid()

        # current model size?
        k = path[i]

        #define the IHTVariable used for cleaner code #TODO: should declare this only 1 time for max efficiency. 
        v = IHTVariables(x, z, y, J, k)

        # now compute current model
        output = L0_reg(v, x, z, y, J, k, use_maf=use_maf, mask_n=mask_n)

        # put model into sparse matrix of betas in the corresponding thread
        betas[cur_thread][:, i] = output.beta
        cs[cur_thread][:, i]    = output.c
    end

    # reduce the vector of matrix into a single matrix, where each column stores a different model 
    return sum(betas), sum(cs)
end


"""
In cross validation we separate samples into `q` disjoint subsets. This function fits a model on 
q-1 of those sets (indexed by train_idx), and then tests the model's performance on the 
qth set (indexed by test_idx). We loop over all sparsity level specified in `path` and 
returns the out-of-sample errors in a vector. 

- `path` , a vector of various model sizes
- `folds`, a vector indicating which of the q fold each sample belongs to. 
- `fold` , the current fold that is being used as test set. 
- `pids` , a vector of process IDs. Defaults to `procs(x)`.
"""
function one_fold(
    x        :: SnpLike{2},
    z        :: Matrix{T},
    y        :: Vector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int}, 
    fold     :: Int;
    use_maf  :: Bool = false,
    #pids     :: Vector{Int} = procs(x),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    # quiet    :: Bool = true
) where {T <: Float}
    # dimensions of problem
    n, p = size(x)
    q    = size(z, 2)

    # find entries that are for test sets and train sets
    test_idx  = folds .== fold
    train_idx = .!test_idx
    test_size = sum(test_idx)

    # compute the regularization path on the training set
    betas, cs = iht_path_threaded(x, z, y, J, path, use_maf=use_maf, mask_n=train_idx, max_iter=max_iter, max_step=max_step, tol=tol)
    # betas, cs = iht_path(x, z, y, J, path, use_maf=use_maf, mask_n=train_idx, max_iter=max_iter, max_step=max_step, tol=tol)

    # preallocate vector for output
    myerrors = zeros(T, length(path))

    # allocate the arrays for the test set
    xb = zeros(test_size,)
    zc = zeros(test_size,)
    r  = zeros(test_size,)
    b  = zeros(p,)
    c  = zeros(q,)

    # compute the mean out-of-sample error for the TEST set
    # do this for every computed model in regularization path
    for i = 1:size(betas,2)

        # pull ith model in dense vector format
        b .= betas[:, i]
        c .= cs[:, i] 

        # compute estimated response Xb with $(path[i]) nonzeroes
        A_mul_B!(xb, x[test_idx, :], b) #should use view(x, test_idx, :) when SnpArray code gets fixe
        A_mul_B!(zc, z[test_idx, :], c)

        # compute residuals
        r .= view(y, test_idx) .- xb .- zc

        # reduction step. Return out-of-sample error as squared residual averaged over size of test set
        myerrors[i] = sum(abs2, r) / test_size / 2
    end

    return myerrors :: Vector{T}
end

"""
Wrapper function for one_fold. Returns the averaged MSE for each fold of cross validation.
mse[i, j] stores the ith model size for fold j. Thus to obtain the mean mse for each fold, 
we take average along the rows and find the minimum.  
"""
function pfold_naive(
    x        :: SnpLike{2},
    z        :: Matrix{T},
    y        :: Vector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    num_fold :: Int64;
    use_maf  :: Bool = false,
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
) where {T <: Float}

    @assert num_fold >= 1 "number of folds must be positive integer"

    mses = zeros(length(path), num_fold)
    for fold in 1:num_fold
        mses[:, fold] = one_fold(x, z, y, J, path, folds, fold, use_maf=use_maf)
    end
    return vec(sum(mses, 2) ./ num_fold)
end

"""
This function runs q-fold cross-validation across a specified regularization path in a 
maximum of 8 parallel threads. 

Important arguments and defaults include:
- `x` is the genotype matrix
- `z` is the matrix of non-genetic covariates (which includes the intercept as the first column)
- `y` is the response (phenotype) vector
- `J` is the maximum allowed active groups. 
- `path` is vector of model sizes k1, k2...etc. IHT will compute all model sizes on each fold.
- `folds` a vector of integers, with the same length as the number predictor, indicating a partitioning of the samples into q disjoin subsets
- `num_folds` indicates how many disjoint subsets the samples are partitioned into. 
- `use_maf` whether IHT wants to scale predictors using their minor allele frequency. This is experimental feature
"""
function cv_iht(
    x        :: SnpLike{2},
    z        :: Matrix{T},
    y        :: Vector{T},
    J        :: Int64,
    path     :: DenseVector{Int},
    folds    :: DenseVector{Int},
    num_fold :: Int64;
    use_maf  :: Bool = false,
    # pids     :: Vector{Int} = procs(),
    # tol      :: Float = convert(T, 1e-4),
    # max_iter :: Int   = 100,
    # max_step :: Int   = 50,
    # quiet    :: Bool  = true,
    # header   :: Bool  = false
) where {T <: Float}

    # how many elements are in the path?
    nmodels = length(path)

    # compute folds
    mses = pfold_naive(x, z, y, J, path, folds, num_fold, use_maf=use_maf)

    # find best model size and print cross validation result
    k = path[indmin(mses)] :: Int
    print_cv_results(mses, path, k)

    return nothing
end
