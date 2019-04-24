"""
This is the wrapper function for the Iterative Hard Thresholding analysis option in Open Mendel.
"""
function IHT(control_file = ""; args...)
    #
    # Print the logo. Store the initial directory.
    #
    print(" \n \n")
    println("     Welcome to OpenMendel's")
    println("      IHT analysis option")
    print(" \n \n")
    println("Reading the data.\n")
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
    keyword["glm"] = ""
    keyword["cpu_cores"] = 1
    #
    # Process the run-time user-specified keywords that will control the analysis.
    # This will also initialize the random number generator.
    #
    process_keywords!(keyword, control_file, args)
    @assert typeof(keyword["max_groups"]) == Int "Number of groups must be an integer. Set as 1 to run normal IHT"
    @assert typeof(keyword["predictors"]) == Int "Sparsity constraint must be positive integer"
    @assert 0 <= keyword["predictors"]           "Need positive number of predictors per group"
    @assert keyword["glm"] != ""                 "GLM not specified! Please choose from Normal, Bernoulli, Poisson, Negative_Binomial, Gamma"
    #
    # Import genotype/non-genetic/phenotype data
    #
    @info("Reading in data")
    snpmatrix = SnpArray(keyword["plink_input_basename"] * ".bed") #requires .bim .bed .fam files
    phenotype = readdlm(keyword["plink_input_basename"] * ".fam", header = false)[:, 6]
    non_genetic_cov = ones(size(snpmatrix, 1), 1) #defaults to just the intercept
    if keyword["non_genetic_covariates"] != ""
        non_genetic_cov = readdlm(keyword["non_genetic_covariates"], keyword["field_separator"], Float64)
    end
    #
    # Determine what weighting (if any) the user specified for each predictors
    #
    keyword["maf_weights"] == "maf" ? use_maf = true : use_maf = false
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
        # Find the model sizes the user wants. Defaults to 1~20
        #
        path = collect(1:20)
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
        @info("Running " * string(num_folds) * "-fold cross validation on the following model sizes:\n" * keyword["model_sizes"] * ".\nIgnoring keyword predictors.")
        folds = rand(1:num_folds, size(snpmatrix, 1))
        #
        # Determine number of cores specified
        #
        if keyword["cpu_cores"] != 1
            addprocs(keyword["cpu_cores"])
        end
        return cv_iht(snpmatrix, non_genetic_cov, phenotype, 1, path, folds, num_folds, use_maf=maf_weights, glm="normal", debias=false)

    elseif keyword["model_sizes"] != ""
        path = [parse(Int, ss) for ss in split(keyword["model_sizes"], ',')]
        @info("Running the following model sizes: " * string(path))
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
                # v = IHTVariables(snpmatrix, non_genetic_cov, phenotype, J, k)
                # if keyword["group_membership"] != ""
                #     v.group = vec(readdlm(keyword["group_membership"], Int64))
                # end
        #
        # Determine the type of analysis and run IHT
        #
        glm = keyword["glm"]
        @info("Running " * string(glm) * " IHT for model size k = $k and groups J = $J") 
        if glm == "normal"
            return L0_reg(snpmatrix, non_genetic_cov, phenotype, J, k, use_maf=maf_weights, debias=false)
        elseif glm == "logistic"
            return L0_logistic_reg(snpmatrix, non_genetic_cov, phenotype, J, k, glm=glm, debias=true, show_info=false)
        elseif glm == "poisson"
            return L0_poisson_reg(snpmatrix, non_genetic_cov, phenotype, J, k, glm=glm, debias=false, convg=false)
        else
            throw(error("unsupported glm option: $glm"))
        end
    end
end #function IHT
