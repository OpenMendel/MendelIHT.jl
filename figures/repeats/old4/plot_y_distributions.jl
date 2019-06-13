using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using GLM
using DelimitedFiles
using Plots

function run()
    #simulat data with k true predictors, from distribution d and with link l.
    repeats = 100 #how many repeats should I run
    n = 1000
    p = 10000
    k = 10
    d = Normal
    l = canonicallink(d())
    # debias = true

    # set random seed for reproducibility
    Random.seed!(2019)

    #plot y distribution
    for i in 1:repeats
        x, = simulate_random_snparray(n, p, undef)
        xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
        z = ones(n, 1)

        # simulate response, true model b, and the correct non-0 positions of b
        y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

        plt = histogram(y)
        savefig(plt, "y_distributions/normal/$i")
    end
end

run()
