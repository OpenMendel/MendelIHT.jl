# """
# This is the wrapper function for the Iterative Hard Thresholding analysis option in Open Mendel.
# """
# function MendelIHT(file_name::String, k::Int64)
#     const MENDEL_IHT_VERSION :: VersionNumber = v"0.2.0"
#     #
#     # Print the logo. Store the initial directory.
#     #
#     print(" \n \n")
#     println("     Welcome to OpenMendel's")
#     println("      IHT analysis option")
#     println("        version ", MENDEL_IHT_VERSION)
#     print(" \n \n")

#     snpmatrix = SnpArray(file_name)
#     phenotype = readdlm(file_name * ".fam", header = false)[:, 6]
#     # phenotype = randn(959) #testing GAW data since it has no phenotype

#     return L0_reg(snpmatrix, phenotype, k)
# end #function MendelIHT

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

    keyword["experiment_description"] = ""
    keyword["experiment_id"] = ""
    keyword["experiment_folder"] = ""
    keyword["mendeliht_version"] = ""

    keyword["plotfilename_prefix"] = ""
    keyword["datafilename_prefix"] = ""
    keyword["use_intercept"] = "keyword not used"
    keyword["snps_first_in_model"] = ""
    keyword["use_weights"] = "true"
    keyword["pw_algorithm"] = ""
    keyword["pw_algorithm_value"] = 1.0

    keyword["null_weight"] = 1.0
    keyword["cut_off"] = 0.001
    keyword["trim_top"] = 0.0

    keyword["pw_pathway1"] = ""
    keyword["pw_pathway1_constantvalue"] = ""
    keyword["pw_pathway1_factorvalue"] = ""
    keyword["pw_pathway2"] = ""
    keyword["pw_pathway2_constantvalue"] = ""
    keyword["pw_pathway2_factorvalue"] = ""
    keyword["pw_pathway3"] = ""
    keyword["pw_pathway3_constantvalue"] = ""
    keyword["pw_pathway3_factorvalue"] = ""

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
    @assert (keyword["max_groups"] != "")           "Need number of groups. Choose 1 to run normal IHT"
    @assert (keyword["predictors_per_group"] != "") "Need number of predictors per group"

    #
    # Read the genetic data from the external files named in the keywords.
    #
    # (pedigree, person, nuclear_family, locus, snpdata, locus_frame, phenotype_frame,
    #     pedigree_frame, snp_definition_frame) = read_external_data_files(keyword)
    #
    # Execute the specified analysis.
    #
    println(" \nAnalyzing the data.\n")
##
    file_name = keyword["plink_input_basename"]
    snpmatrix = SnpArray(file_name)
    phenotype = readdlm(file_name * ".fam", header = false)[:, 6]
    # y_copy = copy(phenotype)
    # y_copy .-= mean(y_copy)
    groups = vec(readdlm(keyword["group_membership"], Int64))
    k = keyword["predictors_per_group"]
    J = keyword["max_groups"]
    return L0_reg(snpmatrix, phenotype, J, k, groups)
##
    # execution_error = iht_gwas(person, snpdata, pedigree_frame, keyword)
    # if execution_error
    #   println(" \n \nERROR: Mendel terminated prematurely!\n")
    # else
    #   println(" \n \nMendel's analysis is finished.\n")
    # end
    # #
    # # Finish up by closing, and thus flushing, any output files.
    # # Return to the initial directory.
    # #
    # close(keyword["output_unit"])
    # cd(initial_directory)
    # return nothing
end #function MendelIHT

"""
Calculates the IHT step β+ = P_k(β - μ ∇f(β)).
Returns step size (μ), and number of times line search was done (μ_step).

This function updates: b, xb, xk, gk, xgk, idx
"""
function iht!(
    v        :: IHTVariable{T},
    x        :: SnpLike{2},
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
    if sum(v.idx) == 0
        _init_iht_indices(v, J, k)
    end

    # store relevant columns of x. Need to do this on 1st iteration.
    # afterwards, only do if support changes
    if !isequal(v.idx, v.idx0) || iter < 2
        copy!(v.xk, view(x, :, v.idx))
    end

    # store relevant components of gradient
    v.gk .= v.df[v.idx]

    # now compute subset of x*g
    SnpArrays.A_mul_B!(v.xgk, v.xk, v.gk, mean_vec[v.idx], std_vec[v.idx])
    v.xgk .+= sum(v.r)

    # warn if xgk only contains zeros
    all(v.xgk .== zero(T)) && warn("Entire active set has values equal to 0")

    # compute step size. Note intercept is separated from x, so gk & xgk is missing an extra entry equal to 1^T (y-Xβ-intercept) = sum(v.r)
    μ = ((sum(abs2, v.gk) + sum(v.r)^2) / sum(abs2, v.xgk)) :: T

    # check for finite stepsize
    isfinite(μ) || throw(error("Step size is not finite, is active set all zero?"))

    # compute gradient step
    _iht_gradstep(v, μ, J, k)

    # update xb
    v.xk .= view(x, :, v.idx)
    SnpArrays.A_mul_B!(v.xb, v.xk, v.b[v.idx], mean_vec[v.idx], std_vec[v.idx])
    v.xb .+= v.itc

    # calculate omega
    ω_top, ω_bot = _iht_omega(v)

    # backtrack until mu < omega and until support stabilizes
    μ_step = 0
    while _iht_backtrack(v, ω_top, ω_bot, μ, μ_step, nstep)

        # stephalving
        μ /= 2

        # recompute gradient step
        copy!(v.b,v.b0)
        v.itc = v.itc0
        _iht_gradstep(v, μ, J, k)

        # recompute xb
        v.xk .= view(x, :, v.idx)
        SnpArrays.A_mul_B!(v.xb, v.xk, v.b[v.idx], mean_vec[v.idx], std_vec[v.idx])
        v.xb .+= v.itc

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
    x        :: SnpLike{2},
    y        :: Vector{T},
    J        :: Int,
    k        :: Int,
    group    :: Vector{Int},
    keyword  :: Dict{AbstractString, Any};
    v        :: IHTVariable = IHTVariables(x, y, J, k),
    # v        :: IHTVariables = IHTVariables(x, y, J, k),
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

    println("Note: Set keyword[\"use_weights\"] = true to use weights.")
    if keyword["use_weights"] == true
        my_snpMAF, my_snpweights = calculatePriorWeightsforIHT(x,y,k,v,keyword)
        # NOTICE - WE ARE NOT USING MY snpmatrix, just my_snpweights and my_snpMAF
        hold_std_vec = deepcopy(std_vec)
        #my_snpweights  = [my_snpweights ones(size(my_snpweights, 1))]
        println("size(std_vec) = $(size(std_vec))")
        println("size(my_snpweights) = $(size(my_snpweights))")
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
    copy!(v.r, y)          #redisual = y-Xβ-intercept = y  CONSIDER BLASCOPY!
    v.r[mask_n .== 0] .= 0 #bit masking? idk why we need this yet
    if size(v.group) != size(group)
        throw(error("Error: Group membership file has $(size(group)), should be $(size(v.group))."))
    end
    v.group .= group       #assign the groups in the beginning

    # Calculate the gradient v.df = -X'(y - Xβ) = X'(-1*(Y-Xb)). All future gradient
    # calculations are done in iht!. Note the negative sign will be cancelled afterwards
    # when we do b+ = P_k( b - μ∇f(b)) = P_k( b + μ(-∇f(b))) = P_k( b + μ*v.df)
    SnpArrays.At_mul_B!(v.df, x, v.r, mean_vec, std_vec)

    for mm_iter = 1:max_iter
        # save values from previous iterate
        copy!(v.b0, v.b)   # b0 = b    CONSIDER BLASCOPY!
        copy!(v.xb0, v.xb) # Xb0 = Xb  CONSIDER BLASCOPY!
        v.itc0 = v.itc     # update intercept as well
        loss = next_loss

        #calculate the step size μ.
        (μ, μ_step) = iht!(v, x, y, J, k, mean_vec, std_vec, max_step, mm_iter)

        # iht! gives us an updated x*b. Use it to recompute residuals and gradient
        v.r .= y .- v.xb # v.r = (y - Xβ - intercept)
        v.r[mask_n .== 0] .= 0 #bit masking, idk why we need this yet

        # v.df = X'(y - Xβ - intercept)
        SnpArrays.At_mul_B!(v.df, x, v.r, mean_vec, std_vec, similar(v.df))

        # update loss, objective, gradient, and check objective is not NaN or Inf
        next_loss = sum(abs2, v.r) / 2
        !isnan(next_loss) || throw(error("Objective function is NaN, aborting..."))
        !isinf(next_loss) || throw(error("Objective function is Inf, aborting..."))

        # track convergence
        the_norm    = max(chebyshev(v.b, v.b0), abs(v.itc - v.itc0)) #max(abs(x - y))
        scaled_norm = the_norm / (max(norm(v.b0, Inf), v.itc0) + 1.0)
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
