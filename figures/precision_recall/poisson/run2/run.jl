using Distributed
addprocs(4)
nprocs()

using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using Random
using LinearAlgebra
using DelimitedFiles
using GLM
using MendelGWAS
using CSV
using RCall
R"library(glmnet)"
R"require(pscl)"
R"require(boot)"

function iht_lasso_marginal(n::Int64, p::Int64, d::UnionAll, l::Link)
    #construct snpmatrix, covariate files, and true model b
    mafs = clamp!(0.5rand(p), 0.01, 0.5) #set lowest minor allele freq to be 0.01 
    x = simulate_random_snparray(n, p, "tmp.bed", mafs=mafs)
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # the intercept
    x_float = [convert(Matrix{Float64}, x, center=true, scale=true) z] #Float64 version of x
    
    # simulate response, true model b, and the correct non-0 positions of b
    true_b = zeros(p)
    true_b[1:10] .= collect(0.1:0.1:1.0)
    shuffle!(true_b)
    correct_position = findall(x -> x != 0, true_b)

    #simulate phenotypes (e.g. vector y)
    if d == Normal || d == Poisson || d == Bernoulli
        prob = linkinv.(l, xbm * true_b)
        clamp!(prob, -20, 20)
        y = [rand(d(i)) for i in prob]
    elseif d == NegativeBinomial
        μ = linkinv.(l, xbm * true_b)
        prob = 1 ./ (1 .+ μ ./ nn)
        y = [rand(d(nn, i)) for i in prob] #number of failtures before nn success occurs
    elseif d == Gamma
        μ = linkinv.(l, xbm * true_b)
        β = 1 ./ μ # here β is the rate parameter for gamma distribution
        y = [rand(d(α, i)) for i in β] # α is the shape parameter for gamma
    end
    y = Float64.(y)

    #specify path and folds
    num_folds = 3
    folds = rand(1:num_folds, size(x, 1));

    #run glmnet via Rcall
    @rput x_float y folds num_folds #make variables visible to R
    R"lasso_cv_result = cv.glmnet(x_float, y, nfolds = num_folds, foldid = folds, family='poisson')"
    R"lasso_beta_tmp = glmnet(x_float, y, lambda=lasso_cv_result$lambda.min, family='poisson')$beta"
    R"lasso_beta = as.vector(lasso_beta_tmp)"
    @rget lasso_cv_result lasso_beta #pull result from R to Julia
    lasso_k_est = count(!iszero, lasso_beta)
    
    #find non-zero entries returned by best lasso model as largest k estimate
    path = collect(1:50);
    
    #run IHT's cross validation routine 
    mses = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, use_maf=false, debias=false, showinfo=false, parallel=true);
    iht_k_est = argmin(mses)
    iht_result = L0_reg(x, xbm, z, y, 1, iht_k_est, d(), l, debias=false, init=false, use_maf=false, show_info=false)
    iht_beta = iht_result.beta
    
    #Now run MendelGWAS
    make_bim_fam_files(x, y, "tmp") #create .bim and .bed files for MendelGWAS

    #create a control file for MendelGWAS
    open("tmp_control.txt", "w") do f
        write(f, "plink_input_basename = tmp \n")
        write(f, "plink_field_separator = '\t' \n")
        write(f, "output_table = tmp_table.txt \n")
        write(f, "regression = poisson \n")
        write(f, "regression_formula = Trait ~ \n")
    end

    #run marginal analysis
    GWAS("tmp_control.txt");

    # calculate false positive/negatives based on bonferroni correction
    p_values = CSV.read("tmp_table.txt", delim=',', header=true)[:Pvalue]
    significance = 0.05 / Float64(p)
    passing_snps_position = findall(p_values .<= significance)
    k = 10
    marginal_found = [correct_position[snp] in passing_snps_position for snp in 1:k]
    
    #clean up
    rm("tmp.bed", force=true)
    rm("tmp.bim", force=true)
    rm("tmp.fam", force=true)
    rm("tmp_table.txt", force=true)
    rm("tmp_control.txt", force=true)
    rm("Mendel_Output.txt", force=true)
    
    # to test if MendelGWAS is correct, fit using GLM.jl too (MendelGWAS is correct)
#     data = DataFrame(X=zeros(n), Y=y)
#     placeholder = zeros(n)
#     pvalues = zeros(p)
#     for i in 1:p
#         copyto!(placeholder, @view(x[:, i]), center=true, scale=true)
#         data.X .= placeholder
#         result = glm(@formula(Y ~ X), data, Poisson(), LogLink())
#         pvalues[i] = coeftable(result).cols[4][2].v
#     end
#     passing_snps_position2 = findall(pvalues .<= significance)
#     marginal_found2 = [correct_position[snp] in passing_snps_position2 for snp in 1:k]

    #Fit 0-inflated Poisson using R package because MendelGWAS has tons of false positives
    y = Int64.(y)
    @rput y n p
    R"
    zip_pvalues = matrix(0.0, p, 1)
    tmp = matrix(0, n, 1)
    for (i in 1:p) {
        tmp[,] = x_float[, i]
        zip_pvalues[i] = summary(zeroinfl(y ~ tmp))$coefficient[2]$zero[2,4]
    }
    "
    @rget zip_pvalues
    zip_pvalues = reshape(zip_pvalues, (10000,))
    zip_passing_snps = findall(zip_pvalues .<= significance)
    zip_found = [correct_position[snp] in zip_passing_snps for snp in 1:k]
    
    #show lasso and IHT's reconstruction result
    compare_model = DataFrame(
        true_β  = true_b[correct_position], 
        IHT_β   = iht_beta[correct_position],
        lasso_β = lasso_beta[correct_position],
        marginal_found = marginal_found,
        zero_inf_Pois = zip_found)
    @show compare_model
    
    #compute true/false positives/negatives for IHT and lasso
    iht_tp = count(!iszero, iht_beta[correct_position])
    iht_fp = iht_k_est - iht_tp
    iht_fn = k - iht_tp
    lasso_tp = count(!iszero, lasso_beta[correct_position])
    lasso_fp = lasso_k_est - lasso_tp
    lasso_fn = k - lasso_tp
    marginal_tp = count(!iszero, true_b[passing_snps_position])
    marginal_fp = length(passing_snps_position) - marginal_tp
    marginal_fn = k - marginal_tp
#     juliaglm_tp = count(!iszero, true_b[passing_snps_position2])
#     juliaglm_fp = length(passing_snps_position2) - juliaglm_tp
#     juliaglm_fn = k - juliaglm_tp
    zip_tp = count(!iszero, true_b[zip_passing_snps])
    zip_fp = length(zip_passing_snps) - zip_tp
    zip_fn = k - zip_tp
    
    println("IHT true positives = $iht_tp")
    println("IHT false positives = $iht_fp")
    println("IHT false negatives = $iht_fn")
    println("LASSO true positives = $lasso_tp")
    println("LASSO false positives = $lasso_fp")
    println("LASSO false negatives = $lasso_fn")
    println("marginal true positives = $marginal_tp")
    println("marginal false positives = $marginal_fp")
    println("marginal false negatives = $marginal_fn")
#     println("julia glm true positives = $juliaglm_tp")
#     println("julia glm false positives = $juliaglm_fp")
#     println("julia glm false negatives = $juliaglm_fn" * "\n")
    println("zero inflated poisson true positives = $zip_tp")
    println("zero inflated poisson  false positives = $zip_fp")
    println("zero inflated poisson  false negatives = $zip_fn" * "\n")
    
    return iht_tp, iht_fp, iht_fn, lasso_tp, lasso_fp, lasso_fn, 
        marginal_tp, marginal_fp, marginal_fn, zip_tp, zip_fp, zip_fn
end

function run()
    #simulat data with k true predictors, from distribution d and with link l.
    n = 1000
    p = 10000
    d = Poisson
    l = canonicallink(d())

    #run function above, saving results in 4 vectors
    total_runs = 50
    iht_true_positives = zeros(total_runs)
    iht_false_positives = zeros(total_runs)
    iht_false_negatives = zeros(total_runs)
    lasso_true_positives = zeros(total_runs)
    lasso_false_positives = zeros(total_runs)
    lasso_false_negatives = zeros(total_runs)
    marginal_true_positives = zeros(total_runs)
    marginal_false_positives = zeros(total_runs)
    marginal_false_negatives = zeros(total_runs)
    zip_true_positives = zeros(total_runs)
    zip_false_positives = zeros(total_runs)
    zip_false_negatives = zeros(total_runs)
    for i in 11:20
        println("current run = $i")
        
        #set random seed
        Random.seed!(i)
        
        iht_tp, iht_fp, iht_fn, lasso_tp, lasso_fp, lasso_fn, 
            marginal_tp, marginal_fp, marginal_fn, zip_tp, zip_fp, 
                zip_fn = iht_lasso_marginal(n, p, d, l)
        iht_true_positives[i] = iht_tp
        iht_false_positives[i] = iht_fp
        iht_false_negatives[i] = iht_fn
        lasso_true_positives[i] = lasso_tp
        lasso_false_positives[i] = lasso_fp
        lasso_false_negatives[i] = lasso_fn
        marginal_true_positives[i] = marginal_tp
        marginal_false_positives[i] = marginal_fp
        marginal_false_negatives[i] = marginal_fn
        zip_true_positives[i] = zip_tp
        zip_false_positives[i] = zip_fp
        zip_false_negatives[i] = zip_fn
    end
    
    return iht_true_positives, iht_false_positives, iht_false_negatives, 
            lasso_true_positives, lasso_false_positives, lasso_false_negatives, 
            marginal_true_positives, marginal_false_positives, marginal_false_negatives,
            zip_true_positives, zip_false_positives, zip_false_negatives
end

function run2()
	iht_true_positives, iht_false_positives, iht_false_negatives, 
    lasso_true_positives, lasso_false_positives, lasso_false_negatives, 
    marginal_true_positives, marginal_false_positives, marginal_false_negatives,
    zip_true_positives, zip_false_positives, zip_false_negatives = run()

    poisson_iht_true_positives = sum(iht_true_positives)
	poisson_iht_false_positives = sum(iht_false_positives)
	poisson_iht_false_negatives = sum(iht_false_negatives)
	poisson_lasso_true_positives = sum(lasso_true_positives)
	poisson_lasso_false_positives = sum(lasso_false_positives)
	poisson_lasso_false_negatives = sum(lasso_false_negatives)
	poisson_marginal_true_positives = sum(marginal_true_positives)
	poisson_marginal_false_positives = sum(marginal_false_positives)
	poisson_marginal_false_negatives = sum(marginal_false_negatives)
	poisson_zip_true_positives = sum(zip_true_positives)
	poisson_zip_false_positives = sum(zip_false_positives)
	poisson_zip_false_negatives = sum(zip_false_negatives)

	result = [poisson_iht_true_positives poisson_iht_false_positives poisson_iht_false_negatives; 
        poisson_lasso_true_positives poisson_lasso_false_positives poisson_lasso_false_negatives;
        poisson_marginal_true_positives poisson_marginal_false_positives poisson_marginal_false_negatives;
        poisson_zip_true_positives poisson_zip_false_positives poisson_zip_false_negatives]

    writedlm("result2", result)
end

run2()
