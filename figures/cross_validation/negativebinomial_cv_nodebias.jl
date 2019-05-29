#first add workers
using Distributed
addprocs(10)
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
    x = simulate_random_snparray(n, p, "test1.bed")
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1)

    #define true_b 
    true_b = zeros(p)
    true_b[1:10] .= collect(0.1:0.1:1.0)
    shuffle!(true_b)
    correct_position = findall(!iszero, true_b)

    #simulate phenotypes (e.g. vector y)
    if d == Normal || d == Poisson || d == Bernoulli
        prob = linkinv.(l, xbm * true_b)
        clamp!(prob, -20, 20)
        y = [rand(d(i)) for i in prob]
    elseif d == NegativeBinomial
        nn = 10
        μ = linkinv.(l, xbm * true_b)
        clamp!(μ, -20, 20)
        prob = 1 ./ (1 .+ μ ./ nn)
        y = [rand(d(nn, i)) for i in prob] #number of failtures before nn success occurs
    elseif d == Gamma
        μ = linkinv.(l, xbm * true_b)
        β = 1 ./ μ # here β is the rate parameter for gamma distribution
        y = [rand(d(α, i)) for i in β] # α is the shape parameter for gamma
    end
    y = Float64.(y)
    
    #specify path and folds
    path = collect(1:20)
    num_folds = 5
    folds = rand(1:num_folds, size(x, 1))

    #compute cross validation
    result = @timed cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, use_maf=false, debias=debias, parallel=true);
    dr = result[1]
    runtime = result[2] #seconds
    memory = result[3] / 1e6  #MB

    rm("test1.bed", force=true)

    return dr, runtime, memory
end

function run()
    # set simulation parameters and seed
    repeats = 30
    n = 5000
    p = 50000
    k = 10
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
