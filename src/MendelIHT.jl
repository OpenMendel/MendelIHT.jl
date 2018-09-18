"""
This is the wrapper function for the Iterative Hard Thresholding analysis option in Open Mendel.
"""
function MendelIHT(control_file = ""; args...)
    const MENDEL_IHT_VERSION :: VersionNumber = v"0.2.0"
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
    keyword["data_type"] = ""
    keyword["predictors_per_group"] = ""
    keyword["manhattan_plot_file"] = ""
    keyword["max_groups"] = ""
    keyword["group_membership"] = ""
    keyword["prior_weights"] = ""
    keyword["pw_algorithm_value"] = 1.0     # not user defined at this time
    keyword["non_genetic_covariates"] = ""
    #
    # Process the run-time user-specified keywords that will control the analysis.
    # This will also initialize the random number generator.
    #
    process_keywords!(keyword, control_file, args)
    #
    # Check that the correct analysis option was specified.
    #
    lc_analysis_option = lowercase(keyword["analysis_option"])
    if (lc_analysis_option != "" && lc_analysis_option != "iht")
        throw(ArgumentError("An incorrect analysis option was specified.\n \n"))
    end
    keyword["analysis_option"] = "Iterative Hard Thresholding"
    @assert keyword["max_groups"] != ""           "Need number of groups. Set as 1 to run normal IHT"
    @assert keyword["predictors_per_group"] != "" "Need number of predictors per group"
    #
    # Import genotype/non-genetic/phenotype data
    #
    println(" \nReading in data.\n")
    snpmatrix = SnpArray(keyword["plink_input_basename"]) #requires .bim .bed .fam files
    phenotype = readdlm(keyword["plink_input_basename"] * ".fam", header = false)[:, 6]
    if keyword["non_genetic_covariates"] != ""
        non_genetic_cov = readdlm(keyword["non_genetic_covariates"], Float64)
    else
        non_genetic_cov = Vector{Float64}(0, 0)
    end
    #
    # Define variables for group membership, max number of predictors for each group, and max number of groups
    #
    groups = vec(readdlm(keyword["group_membership"], Int64))
    k      = keyword["predictors_per_group"]
    J      = keyword["max_groups"]
    v      = IHTVariables(snpmatrix, non_genetic_cov, phenotype, J, k)
    #
    #assign groups memberships
    #
    if size(v.group) != size(groups)
        throw(error("Error: Group membership file has $(size(group)), should be $(size(v.group))."))
    end
    v.group = groups
    #
    # Execute the specified analysis.
    #
    println(" \nAnalyzing the data.\n")
    return L0_reg(v, snpmatrix, non_genetic_cov, phenotype, J, k, groups, keyword)
    #
    # Finish up by closing, and thus flushing, any output files.
    # Return to the initial directory.
    #
    close(keyword["output_unit"])
    cd(initial_directory)
    return nothing
end #function MendelIHT

"""
Calculates the IHT step β+ = P_k(β - μ ∇f(β)).
Returns step size (μ), and number of times line search was done (μ_step).

This function updates: b, xb, xk, gk, xgk, idx
"""
function iht!(
    v        :: IHTVariable{T},
    x        :: SnpLike{2},
    z        :: Matrix{T},
    y        :: Vector{T},
    J        :: Int,
    k        :: Int,
    mean_vec :: Vector{T},
    std_vec  :: Vector{T},
    iter     :: Int = 1,
    nstep    :: Int = 50,
) where {T <: Float}

    # compute indices of nonzeroes in beta
    v.idx .= v.b .!= 0

    println(sum(abs2, v.df) + sum(abs2, v.df2))
    if sum(v.idx) == 0
        _init_iht_indices(v, J, k)
    end
    println(sum(abs2, v.df) + sum(abs2, v.df2))

    # store relevant columns of x. Need to do this on 1st iteration, and when support changes
    if !isequal(v.idx, v.idx0) || iter < 2
        copy!(v.xk, view(x, :, v.idx))
    end

    # store relevant components of gradient (gk is numerator of step size)
    v.gk .= v.df[v.idx]

    # compute the denominator of step size, store it in xgk (recall gk = df[idx])
    SnpArrays.A_mul_B!(v.xgk, v.xk, v.gk, view(mean_vec, v.idx), view(std_vec, v.idx))

    # compute z * df2 needed in the denominator of step size calculation
    BLAS.A_mul_B!(v.zdf2, z, v.df2)

    println(v.gk)
    println(v.df2)
    println(sum(abs2, v.zdf2))

    # warn if xgk only contains zeros
    all(v.xgk .== zero(T)) && warn("Entire active set has values equal to 0")

    # compute step size. Note intercept is separated from x, so gk & xgk is missing an extra entry equal to 1^T (y-Xβ-intercept) = sum(v.r)
    μ = ((sum(abs2, v.gk) + sum(abs2, v.df2) / (sum(abs2, v.xgk) + sum(abs2, v.zdf2)))) :: T

    println("below is numerator and denominator")
    println(sum(abs2, v.gk) + sum(abs2, v.df2))
    println(sum(abs2, v.xgk) + sum(abs2, v.zdf2))

    # check for finite stepsize
    isfinite(μ) || throw(error("Step size is not finite, is active set all zero?"))

    # compute gradient step
    _iht_gradstep(v, μ, J, k)

    # update xb (needed to calculate ω to determine line search criteria)
    v.xk .= view(x, :, v.idx)
    SnpArrays.A_mul_B!(v.xb, v.xk, view(v.b, v.idx), view(mean_vec, v.idx), view(std_vec, v.idx))

    # calculate omega
    ω_top, ω_bot = _iht_omega(v)

    # backtrack until mu < omega and until support stabilizes
    μ_step = 0
    while _iht_backtrack(v, ω_top, ω_bot, μ, μ_step, nstep)

        # stephalving
        μ /= 2

        # recompute gradient step
        copy!(v.b,v.b0)
        copy!(v.c,v.c0)
        _iht_gradstep(v, μ, J, k)

        # recompute xb
        v.xk .= view(x, :, v.idx)
        SnpArrays.A_mul_B!(v.xb, v.xk, view(v.b, v.idx), view(mean_vec, v.idx), view(std_vec, v.idx))

        # calculate omega
        ω_top, ω_bot = _iht_omega(v)

        # increment the counter
        μ_step += 1
    end

    return μ::T, μ_step::Int
end

"""
This function performs IHT on GWAS data.
"""
function L0_reg(
    v        :: IHTVariable, 
    x        :: SnpLike{2},
    z        :: Matrix{T},
    y        :: Vector{T},
    J        :: Int,
    k        :: Int,
    group    :: Vector{Int},
    keyword  :: Dict{AbstractString, Any};
    mask_n   :: Vector{Int} = ones(Int, size(y)),
    tol      :: T = 1e-4,
    max_iter :: Int = 200, # up from 100 for sometimes weighting takes more
    max_step :: Int = 50,
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

    # compute some summary statistics for our snpmatrix
    maf, minor_allele, missings_per_snp, missings_per_person = summarize(x)
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
    if keyword["prior_weights"] == "maf"
        my_snpMAF, my_snpweights = calculate_snp_weights(x,y,k,v,keyword,maf)
        hold_std_vec = deepcopy(std_vec)
        Base.A_mul_B!(std_vec, diagm(hold_std_vec), my_snpweights[1,:])
    else
        # need dummies for my_snpMAF and my_snpweights for Gordon's reports
        my_snpMAF = convert(Matrix{Float64},maf')
        my_snpweights = ones(my_snpMAF)
    end

    #
    # Begin IHT calculations
    #
    fill!(v.xb, 0.0)       #initialize β = 0 vector, so Xβ = 0
    copy!(v.r, y)          #redisual = y-Xβ-zc = y since initially β = c = 0
    v.r[mask_n .== 0] .= 0 #bit masking? idk why we need this yet

    # Calculate the gradient v.df = -[X' ; Z']'(y - Xβ - Zc) = [X' ; Z'](-1*(Y-Xb - Zc))
    SnpArrays.At_mul_B!(v.df, x, v.r, mean_vec, std_vec)
    BLAS.At_mul_B!(v.df2, z, v.r)

    for mm_iter = 1:max_iter
        # save values from previous iterate
        copy!(v.b0, v.b)   # b0 = b    CONSIDER BLASCOPY!
        copy!(v.xb0, v.xb) # Xb0 = Xb  CONSIDER BLASCOPY!
        copy!(v.c0, v.c)   # c0 = c    CONSIDER BLASCOPY!
        copy!(v.zc0, v.zc) # Zc0 = Zc  CONSIDER BLASCOPY!
        loss = next_loss

        #calculate the step size μ.
        (μ, μ_step) = iht!(v, x, z, y, J, k, mean_vec, std_vec, max_step, mm_iter)

        println("below is step size and number of step-halving")
        println(μ)
        println(μ_step)
        return 1.1111

        # iht! gives us an updated x*b. Use it to recompute residuals and gradient
        v.r .= y .- v.xb .- v.zc 
        v.r[mask_n .== 0] .= 0 #bit masking, idk why we need this yet

        # update v.df = [ X'(y - Xβ - zc) ; Z'(y - Xβ - zc) ]
        SnpArrays.At_mul_B!(v.df, x, v.r, mean_vec, std_vec, similar(v.df))
        BLAS.At_mul_B!(v.df2, z, v.r)

        # update loss, objective, gradient, and check objective is not NaN or Inf
        next_loss = sum(abs2, v.r) / 2
        !isnan(next_loss) || throw(error("Objective function is NaN, aborting..."))
        !isinf(next_loss) || throw(error("Objective function is Inf, aborting..."))

        # track convergence
        the_norm    = max(chebyshev(v.b, v.b0), chebyshev(v.c, v.c0)) #max(abs(x - y))
        scaled_norm = the_norm / (max(norm(v.b0, Inf), norm(v.c0, Inf)) + 1.0)
        converged   = scaled_norm < tol

        if converged
            mm_time = toq()   # stop time
            return gIHTResults(mm_time, next_loss, mm_iter, v.b, v.itc, J, k, group)
        end

        if mm_iter == max_iter
            mm_time = toq() # stop time
            throw(error("Did not converge!!!!! The run time for IHT was " *
                string(mm_time) * "seconds"))
        end
    end
end #function L0_reg
