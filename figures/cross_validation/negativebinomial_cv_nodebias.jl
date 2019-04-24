#first add workers
using Distributed
addprocs(5)
nprocs()

#load packages into all worker
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using BenchmarkTools
using GLM
using DelimitedFiles

function run_cv(n :: Int64, p :: Int64, k :: Int64, debias :: Bool, d::UnionAll, l::Link)
    #construct snpmatrix, covariate files, and true model b
    x, = simulate_random_snparray(n, p, undef)
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # the intercept

    # simulate response, true model b, and the correct non-0 positions of b
    y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

    #specify path and folds
    path = collect(1:20)
    num_folds = 5
    folds = rand(1:num_folds, size(x, 1))

    #compute cross validation
    result = @timed cv_iht_distributed(d(), l, x, z, y, 1, path, folds, num_folds, use_maf=false, debias=debias, parallel=true);
    dr = result[1]
    runtime = result[2] #seconds
    memory = result[3] / 1e6  #MB

    return dr, runtime, memory
end

function run()
    # set simulation parameters and seed
    repeats = 30
    n = 5000
    p = 50000
    k = 10 # number of true predictors
    d = NegativeBinomial
    l = LogLink()
    debias = false
    Random.seed!(2019)

    # initialize variables then run
    drs = zeros(20, repeats)
    run_time = zeros(repeats)
    memory = zeros(repeats)
    for i in 1:repeats
        dr, t, m = run_cv(n, p, k, debias, d, l)
        drs[:, i] .= dr
        run_time[i] = t
        memory[i] = m
    end

    writedlm("negativebinomial_nodebias_cv_drs", drs)
    writedlm("negativebinomial_nodebias_cv_run_times", run_time)
    writedlm("negativebinomial_nodebias_cv_memory", memory)
end

run()
