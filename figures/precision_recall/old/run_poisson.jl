using Distributed
addprocs(4)
nprocs()

using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic
using Random
using LinearAlgebra
using DelimitedFiles
using GLM
using RCall
R"library(glmnet)"

function iht_lasso_poisson(n :: Int64, p :: Int64, k :: Int64)
    glm = "poisson"

    #construct snpmatrix, covariate files, and true model b
    x, maf = simulate_random_snparray(n, p, "poisson.bed")
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # non-genetic covariates, just the intercept
    true_b = zeros(p)
    true_b[1:k] = rand(Normal(0, 0.3), k)
    shuffle!(true_b)
    correct_position = findall(x -> x != 0, true_b)
    x_float = [convert(Matrix{Float64}, x, center=true, scale=true) z] #Float64 version of x

    # Simulate poisson data
    y_temp = xbm * true_b
    λ = exp.(y_temp) #inverse log link
    y = [rand(Poisson(x)) for x in λ]
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
    lasso_k_est = length(findall(!iszero, lasso_beta))
    
    #find non-zero entries returned by best lasso model as largest k estimate
    path = collect(1:50);
    
    #run IHT's cross validation routine 
    mses = cv_iht_distributed(x, z, y, 1, path, folds, num_folds, glm, use_maf=false, debias=true)
    iht_k_est = argmin(mses)
    iht_result = L0_poisson_reg(x, xbm, z, y, 1, iht_k_est, glm = "poisson", debias=true, convg=false, show_info=false, true_beta=true_b, scale=false, init=false)
    iht_beta = iht_result.beta
        
    #show lasso and IHT's reconstruction result
    compare_model = DataFrame(
        true_β  = true_b[correct_position],
        IHT_β   = iht_beta[correct_position],
        lasso_β = lasso_beta[correct_position])
    @show compare_model

    #compute precision/recall for IHT and lasso
    iht_tp = length(findall(!iszero, iht_beta[correct_position]))
    iht_fp = iht_k_est - iht_tp
    iht_fn = k - iht_tp
    iht_precision = iht_tp / (iht_tp + iht_fp)
    iht_recall = iht_tp / (iht_tp + iht_fn)

    lasso_tp = length(findall(!iszero, lasso_beta[correct_position]))
    lasso_fp = lasso_k_est - lasso_tp
    lasso_fn = k - lasso_tp
    lasso_precision = lasso_tp / (lasso_tp + lasso_fp)
    lasso_recall = lasso_tp / (lasso_tp + lasso_fn) 
    
    return iht_precision, iht_recall, lasso_precision, lasso_recall
end

function run()
    Random.seed!(2019)

    total_runs = 30
    iht_precision = zeros(total_runs)
    iht_recall = zeros(total_runs)
    lasso_precision = zeros(total_runs)
    lasso_recall = zeros(total_runs)
    for i in 1:total_runs
        println("current run = $i")
        n = 1000
        p = 10000
        k = 10
        ihtp, ihtr, lassop, lassor = iht_lasso_poisson(n, p, k)
        iht_precision[i] = ihtp
        iht_recall[i] = ihtr
        lasso_precision[i] = lassop
        lasso_recall[i] = lassor
    end
    
    writedlm("./poisson_results/iht_precision", iht_precision)
    writedlm("./poisson_results/iht_recall", iht_recall)
    writedlm("./poisson_results/lasso_precision", lasso_precision)
    writedlm("./poisson_results/lasso_precision", lasso_precision)
end

run()

