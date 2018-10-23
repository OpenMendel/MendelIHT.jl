"""
This is the wrapper function for the Iterative Hard Thresholding analysis option in Open Mendel.
"""
function MendelIHT(control_file = ""; args...)
    const MENDEL_IHT_VERSION :: VersionNumber = v"0.4.0"
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
    #
    # Check that the correct analysis option was specified.
    #
    lc_analysis_option = lowercase(keyword["analysis_option"])
    if (lc_analysis_option != "" && lc_analysis_option != "iht")
        throw(ArgumentError("An incorrect analysis option was specified.\n \n"))
    end
    keyword["analysis_option"] = "Iterative Hard Thresholding"
    @assert typeof(keyword["max_groups"]) == Int "Number of groups must be an integer. Set as 1 to run normal IHT"
    @assert typeof(keyword["predictors"]) == Int "Sparsity constraint must be positive integer"
    @assert 0 <= keyword["predictors"]           "Need positive number of predictors per group"
    #
    # Import genotype/non-genetic/phenotype data
    #
    info("Reading in data")
    snpmatrix = SnpArray(keyword["plink_input_basename"]) #requires .bim .bed .fam files
    phenotype = readdlm(keyword["plink_input_basename"] * ".fam", header = false)[:, 6]
    if keyword["non_genetic_covariates"] == ""
        non_genetic_cov = ones(size(snpmatrix, 1), 1)
    else
        non_genetic_cov = readdlm(keyword["non_genetic_covariates"], keyword["field_separator"], Float64)
    end
    #
    # Define variables for group membership, max number of predictors for each group, and max number of groups
    #
    k = keyword["predictors"]
    J = keyword["max_groups"]
    v = IHTVariables(snpmatrix, non_genetic_cov, phenotype, J, k)
    if keyword["group_membership"] != ""
        v.group = vec(readdlm(keyword["group_membership"], Int64))
    else
        v.group = ones(size(snpmatrix, 2))
    end
    #
    # Determine what weighting (if any) the user specified for each predictors
    #
    keyword["maf_weights"] == "maf" ? maf_weights = true : maf_weights = false
    #
    # Execute the specified analysis.
    #
    if keyword["run_cross_validation"]
        #
        # Specify how many folds of cross validation the user wants. Defaults to 5
        #
        num_folds = 5
        if keyword["cv_folds"] != "" 
            num_folds = keyword["cv_folds"]
            @assert typeof(num_folds) == Int "Please provide an integer value for the number of folds for cross validation"
        end
        info("Running " * string(num_folds) * "-fold cross validation on the following model sizes:\n" * keyword["model_sizes"] * ".\nIgnoring keyword predictors.")
        folds = rand(1:num_folds, size(snpmatrix, 1))
        #
        # Find the model sizes the user wants. Defaults to 1~10
        #
        if keyword["model_sizes"] == ""
            path = collect(1:10)
            info("Running the following model sizes: " * string(path))
        else
            path = [parse(Int, ss) for ss in split(keyword["model_sizes"], ',')]
            @assert typeof(path) == Vector{Int} "Cannot parse input paths!"
        end
        return cv_iht(snpmatrix, non_genetic_cov, phenotype, J, path, folds, num_folds, use_maf = maf_weights)

    elseif keyword["model_sizes"] != ""
        path = [parse(Int, ss) for ss in split(keyword["model_sizes"], ',')]
        info("Running the following model sizes: " * string(path))
        @assert typeof(path) == Vector{Int} "Cannot parse input paths!"
        #
        # Compute the various models and associated errors
        #
        return iht_path_threaded(snpmatrix, non_genetic_cov, phenotype, J, path, use_maf = maf_weights)

    else
        info("Analyzing the data for model size k = $k") 
        return L0_reg(v, snpmatrix, non_genetic_cov, phenotype, J, k, use_maf = maf_weights)
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
- `nstep` number of backtracking done. 
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
    storage  :: Vector{Vector{T}},
    iter     :: Int = 1,
    nstep    :: Int = 50,
) where {T <: Float}

    # compute indices of nonzeroes in beta
    v.idx .= v.b .!= 0

    if sum(v.idx) == 0
        _init_iht_indices(v, J, k)
    end

    # store relevant columns of x. Need to do this on 1st iteration, and when support changes
    if !isequal(v.idx, v.idx0) || iter < 2
        copy!(v.xk, view(x, :, v.idx))
    end

    # store relevant components of gradient (gk is numerator of step size)
    v.gk .= view(v.df, v.idx)

    # compute the denominator of step size, store it in xgk (recall gk = df[idx])
    SnpArrays.A_mul_B!(v.xgk, v.xk, v.gk, view(mean_vec, v.idx), view(std_vec, v.idx), storage[2], storage[3])

    # compute z * df2 needed in the denominator of step size calculation
    BLAS.A_mul_B!(v.zdf2, z, v.df2)

    # warn if xgk only contains zeros
    all(v.xgk .== zero(T)) && warn("Entire active set has values equal to 0")

    # compute step size. Note intercept is separated from x, so gk & xgk is missing an extra entry equal to 1^T (y-Xβ-intercept) = sum(v.r)
    μ = (((sum(abs2, v.gk) + sum(abs2, v.df2)) / (sum(abs2, v.xgk) + sum(abs2, v.zdf2)))) :: T

    # check for finite stepsize
    isfinite(μ) || throw(error("Step size is not finite, is active set all zero?"))

    # compute gradient step
    _iht_gradstep(v, μ, J, k)

    # update xb (needed to calculate ω to determine line search criteria)
    v.xk .= view(x, :, v.idx)
    SnpArrays.A_mul_B!(v.xb, v.xk, view(v.b, v.idx), view(mean_vec, v.idx), view(std_vec, v.idx), storage[2], storage[3])
    BLAS.A_mul_B!(v.zc, z, v.c)

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
        SnpArrays.A_mul_B!(v.xb, v.xk, view(v.b, v.idx), view(mean_vec, v.idx), view(std_vec, v.idx), storage[2], storage[3])
        BLAS.A_mul_B!(v.zc, z, v.c)

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

    # Calculate the gradient v.df = -[X' ; Z']'(y - Xβ - Zc) = [X' ; Z'](-1*(Y-Xb - Zc))
    SnpArrays.At_mul_B!(v.df, x, v.r, mean_vec, std_vec, store[1])
    BLAS.At_mul_B!(v.df2, z, v.r)

    for mm_iter = 1:max_iter
        # save values from previous iterate
        copy!(v.b0, v.b)   # b0 = b    CONSIDER BLASCOPY!
        copy!(v.xb0, v.xb) # Xb0 = Xb  CONSIDER BLASCOPY!
        copy!(v.c0, v.c)   # c0 = c    CONSIDER BLASCOPY!
        copy!(v.zc0, v.zc) # Zc0 = Zc  CONSIDER BLASCOPY!
        loss = next_loss

        #calculate the step size μ.
        (μ, μ_step) = iht!(v, x, z, y, J, k, mean_vec, std_vec, store, max_step, mm_iter)

        # iht! gives us an updated x*b. Use it to recompute residuals and gradient
        v.r .= y .- v.xb .- v.zc 
        v.r[mask_n .== 0] .= 0 #bit masking, used for cross validation

        # update v.df = [ X'(y - Xβ - zc) ; Z'(y - Xβ - zc) ]
        SnpArrays.At_mul_B!(v.df, x, v.r, mean_vec, std_vec, store[1])
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

# """
# This is a wrapper function that called one_fold on each of the q subset. 

# - `path` , a vector of various model sizes
# - `folds`, a vector indicating which of the q fold each sample belongs to. 
# - `fold` , the current fold that is being used as test set. 
# - `pids` , a vector of process IDs. Defaults to `procs(x)`.
# """
# function pfold(
#     x        :: SnpLike{2},
#     z        :: Matrix{T},
#     y        :: Vector{T},
#     path     :: DenseVector{Int},
#     folds    :: DenseVector{Int},
#     J        :: Int64, 
#     keyword  :: Dict{AbstractString, Any};
#     # pids     :: Vector{Int},
#     # q        :: Int, 
#     max_iter :: Int  = 100,
#     max_step :: Int  = 50,
#     quiet    :: Bool = true,
#     header   :: Bool = false
# ) where {T <: Float}

#     # ensure correct type
#     @assert T <: Float "Argument T must be either Float32 or Float64"

#     # do not allow crossvalidation with fewer than 3 folds
#     @assert q > 2 "Number of folds q = $q must be at least 3."

#     # how many CPU processes can pfold use?
#     np = length(pids)

#     # report on CPU processes
#     quiet || println("pfold: np = ", np)
#     quiet || println("pids = ", pids)

#     # set up function to share state (indices of folds)
#     i = 1
#     nextidx() = (idx=i; i+=1; idx)

#     # preallocate array for results
#     results = SharedArray{T}((length(path),q), pids=pids) :: SharedMatrix{T}

#     # master process will distribute tasks to workers
#     # master synchronizes results at end before returning
#     @sync begin

#         # loop over all workers
#         for worker in pids

#             # exclude process that launched pfold, unless only one process is available
#             if worker != myid() || np == 1

#                 # asynchronously distribute tasks
#                 @async begin
#                     while true

#                         # grab next fold
#                         current_fold = nextidx()

#                         # if current fold exceeds total number of folds then exit loop
#                         current_fold > q && break

#                         # report distribution of fold to worker and device
#                         quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker.\n\n")

#                         # launch job on worker
#                         # worker loads data from file paths and then computes the errors in one fold
#                         r = remotecall_fetch(worker) do
#                             processes = [worker]
#                             quiet || println("xfile = $(xfile)")
#                             quiet || println("x2file = $(x2file)")
#                             quiet || println("here 530, abspath(yfile) = $(abspath(yfile))")


#                             file_name = xfile[1:end-4] #BenBenBen
#                             #snpmatrix = SnpArray("x_test")
#                             snpmatrix = SnpArray(file_name)
#                             # HERE I READ THE PHENOTYPE FROM THE BEDFILE TO MATCH THE TUTORIAL RESULTS
#                             # I'M SURE THERE IS A BETTER WAY TO GET IT BenBenBen
#                             #phenotype is already set in tutorial_simulation.jl above  DIDN'T WORK - SAYS IT'S NOT DEFINED HERE
#                             #phenotype = readdlm(file_name * ".fam", header = false)[:, 6] # NO GOOD, THE PHENOTYPE HERE IS ALL ONES
#                             x = BEDFile(T, xfile, x2file, header=header, pids=[1]) :: BEDFile{T}
#                             y = SharedArray{T}(abspath(yfile), (x.geno.n,), pids=[1]) :: SharedVector{T}
#                             phenotype = convert(Array{T,1}, y)
#                             # y_copy = copy(phenotype)
#                             # y_copy .-= mean(y_copy)

#                             #OLDWAY x = BEDFile(T, xfile, x2file, pids=processes, header=header)
#                             #OLDWAY y = SharedArray{T}(abspath(yfile), (x.geno.n,), pids=processes) :: SharedVector{T}
#                             quiet || println("here 533, abspath(yfile) = $(abspath(yfile))")
#                             #y = SharedArray{T}(yfile, (x.geno.n,), pids=processes) :: SharedVector{T}

#                             one_fold(snpmatrix, phenotype, path, folds, current_fold, J, groups, keyword, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=processes)
#                             #OLDWAY one_fold(x, y, path, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=processes)
#                             # !!! don't put any code here, return from one_fold() is return for remotecall_fetch() !!!
#                         end # end remotecall_fetch()
#                         quiet || println("Fold done.")
#                         setindex!(results, r, :, current_fold)
#                     end # end while
#                 end # end @async
#             end # end if
#         end # end for
#     end # end @sync

#     # return reduction (row-wise sum) over results
#     return (vec(sum(results, 2) ./ q)) :: Vector{T}
# end

# # default for previous function is Float64
# pfold(xfile::String, x2file::String, yfile::String, path::DenseVector{Int}, folds::DenseVector{Int}, pids::Vector{Int}, q::Int, J::Int64, groups::Vector{Int64}, keyword::Dict{AbstractString, Any}; max_iter::Int = 100, max_step::Int = 50, quiet::Bool = true, header::Bool = false) = pfold(Float64, xfile, x2file, yfile, path, folds, pids, q, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)



# """
#     cv_iht(xfile, xtfile, x2file, yfile, meanfile, precfile, [q=max(3, min(CPU_CORES,5)), path=collect(1:min(p,20)), folds=cv_get_folds(n,q), pids=procs()])

# This variant of `cv_iht()` performs `q`-fold crossvalidation with a `BEDFile` object loaded by `xfile`, `xtfile`, and `x2file`,
# with column means stored in `meanfile` and column precisions stored in `precfile`.
# The continuous response is stored in `yfile` with data particioned by the `Int` vector `folds`.
# The folds are distributed across the processes given by `pids`.
# The dimensions `n` and `p` are inferred from BIM and FAM files corresponding to the BED file path `xpath`.
# """
# function cv_ihtNOTUSED(
#     T        :: Type,
#     xfile    :: String,
#     xtfile   :: String,
#     x2file   :: String,
#     yfile    :: String,
#     meanfile :: String,
#     precfile :: String;
#     q        :: Int = cv_get_num_folds(3,5),
#     path     :: DenseVector{Int} = begin
#            # find p from the corresponding BIM file, then make path
#             bimfile = xfile[1:(endof(xfile)-3)] * "bim"
#             p       = countlines(bimfile)
#             collect(1:min(20,p))
#             end,
#     folds    :: DenseVector{Int} = begin
#            # find n from the corresponding FAM file, then make folds
#             famfile = xfile[1:(endof(xfile)-3)] * "fam"
#             n       = countlines(famfile)
#             cv_get_folds(n, q)
#             end,
#     pids     :: Vector{Int} = procs(),
#     tol      :: Float = convert(T, 1e-4),
#     max_iter :: Int   = 100,
#     max_step :: Int   = 50,
#     quiet    :: Bool  = true,
#     header   :: Bool  = false
# )

#     # enforce type
#     @assert T <: Float "Argument T must be either Float32 or Float64"

#     # how many elements are in the path?
#     nmodels = length(path)

#     # compute folds in parallel
#     mses = pfold(T, xfile, xtfile, x2file, yfile, meanfile, precfile, path, folds, pids, q, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)

#     # what is the best model size?
#     k = path[indmin(errors)] :: Int

#     # print results
#     !quiet && print_cv_results(mses, path, k)

#     # recompute ideal model
#     # first load data on *all* processes
#     x = BEDFile(T, xfile, xtfile, x2file, meanfile, precfile, header=header, pids=pids) :: BEDFile{T}
#     y = SharedArray{T}(abspath(yfile), (x.geno.n,), pids=pids) :: SharedVector{T}

#     # first use L0_reg to extract model
#     output = L0_reg(x, y, k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, pids=pids)

#     # which components of beta are nonzero?
#     inferred_model = output.beta .!= zero(T)
#     bidx = find(inferred_model)

#     # allocate the submatrix of x corresponding to the inferred model
#     x_inferred = zeros(T, x.geno.n, sum(inferred_model))
#     decompress_genotypes!(x_inferred, x, inferred_model)

#     # refit the best model
#     b, bidx = refit_iht(x_inferred, y, k, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet)

#     bids = prednames(x)[bidx]
#     return IHTCrossvalidationResults{T}(mses, sdata(path), b, bidx, k, bids)
# end

# encodes default type FLoat64 for previous function
### 22 Sep 2016: Julia v0.5 warns that this conflicts with cv_iht for GPUs
### since this is no longer the default interface for cv_iht with CPUs,
### then it is commented out here
#cv_iht(xfile::String, xtfile::String, x2file::String, yfile::String, meanfile::String, precfile::String; q::Int = max(3, min(CPU_CORES, 5)), path::DenseVector{Int} = begin bimfile=xfile[1:(endof(xfile)-3)] * "bim"; p=countlines(bimfile); collect(1:min(20,p)) end, folds::DenseVector{Int} = begin famfile=xfile[1:(endof(xfile)-3)] * "fam"; n=countlines(famfile); cv_get_folds(n, q) end, pids::DenseVector{Int}=procs(), tol::Float64=1e-4, max_iter::Int=100, max_step::Int=50, quiet::Bool=true, header::Bool=false) = cv_iht(Float64, xfile, xtfile, x2file, yfile, meanfile, precfile, path=path, folds=folds, q=q, pids=pids, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)

"""
    cv_iht(T::Type, x, z, y, J, k, groups, use_maf, [q=max(3, min(CPU_CORES,5)), path=collect(1:min(p,20)), folds=cv_get_folds(n,q), pids=procs()])

The default call to `cv_iht`. Here `x` points to the PLINK BED file stored on disk, `x2file` points to the nongenetic covariates stored in a delimited file, and `yfile` points to the response variable stored in a **binary** file.

Important optional arguments and defaults include:

- `q`, the number of crossvalidation folds. Defaults to `max(3, min(CPU_CORES,5))`
- `path`, an `Int` vector that contains the model sizes to test. Defaults to `collect(1:min(p,20))`, where `p` is the number of genetic predictors read from the PLINK BIM file.
- `folds`, an `Int` vector that specifies the fold structure. Defaults to `cv_get_folds(n,q)`, where `n` is the number of cases read from the PLINK FAM file.
- `pids`, an `Int` vector of process IDs. Defaults to `procs()`.
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
    tol      :: Float = convert(T, 1e-4),
    max_iter :: Int   = 100,
    max_step :: Int   = 50,
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
end
