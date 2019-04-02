using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using StatsFuns: logistic
using DelimitedFiles

function run_normal_cv(n :: Int64, p :: Int64, k :: Int64, debias :: Bool)
    #construct snpmatrix, covariate files, and true model b
    x, maf = simulate_random_snparray(n, p)
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # non-genetic covariates, just the intercept
    true_b = zeros(p)
    true_b[1:k] = randn(k)
    shuffle!(true_b)
    correct_position = findall(x -> x != 0, true_b)
    noise = rand(Normal(0, 0.1), n) # noise vectors from N(0, s) 

    #simulate phenotypes (e.g. vector y) via: y = Xb + noise
    y = xbm * true_b + noise

    #specify path and folds
    path = collect(1:20)
    num_folds = 5
    folds = rand(1:num_folds, size(x, 1))

    #compute cross validation
    result = @timed cv_iht(x, z, y, 1, path, folds, num_folds, use_maf = false, glm = "normal", debias=debias)
    mses = result[1]
    runtime = result[2] #seconds
    memory = result[3] / 1e6  #MB

    return mses, runtime, memory
end

function run_normal_cv(n :: Int64, p :: Int64)
    #run the real code to 
    Random.seed!(2019)
    k = 10 # number of true predictors
    repeats = 30
    mses = zeros(20, repeats)
    run_time = zeros(repeats)
    memory = zeros(repeats)
    debias = false
    for i in 1:repeats
        mse, t, m = run_normal_cv(n, p, k, debias)
        mses[:, i] .= mse
        run_time[i] = t
        memory[i] = m
    end

    writedlm("normal_cv_mses_nodebias", mses)
    writedlm("normal_cv_run_times_nodebias", run_time)
    writedlm("normal_cv_memory_nodebias", memory)
end

run_normal_cv(5000, 50000)


