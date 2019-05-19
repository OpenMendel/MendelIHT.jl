using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using GLM
using DelimitedFiles

#run n repeats of L0_reg using the same X and b, but different y
function repeat(
    repeats  :: Int, 
    x        :: SnpArray,
    xbm      :: SnpBitMatrix,
    z        :: AbstractMatrix{Float64},
    true_b   :: Vector{Float64},
    cor_pos  :: Vector{Int}, #correct position of the true model
    d        :: UnionAll,
    l        :: Link,
    debias   :: Bool
)
    n = size(xbm, 1)
    k = size(cor_pos, 1)
    estimated_β = zeros(k, repeats)
    
    for i in 1:repeats
        #simulate phenotypes (e.g. vector y)
        if d == Normal || d == Poisson || d == Bernoulli
            prob = linkinv.(l, xbm * true_b)
            y = [rand(d(i)) for i in prob]
        elseif d == NegativeBinomial
            nn = 10
            μ = linkinv.(l, xbm * true_b)
            prob = 1 ./ (1 .+ μ ./ nn)
            y = [rand(d(nn, i)) for i in prob] #number of failtures before nn success occurs
        elseif d == Gamma
            μ = linkinv.(l, xbm * true_b)
            β = 1 ./ μ # here β is the rate parameter for gamma distribution
            y = [rand(d(α, i)) for i in β] # α is the shape parameter for gamma
        end
        y = Float64.(y)

        #compute IHT result for less noisy data
        result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=debias, init=false, show_info=false)

        #store the correct position in estimated model
        estimated_β[:, i] .= result.beta[cor_pos]
    end
    
    return estimated_β
end

function run()
    #simulat data with k true predictors, from distribution d and with link l.
    repeats = 100 #how many repeats should I run
    n = 5000
    p = 100000
    d = Normal
    l = canonicallink(d())
    debias = false

    # set random seed for reproducibility
    Random.seed!(2019)

    # simulat SNP data
    x, = simulate_random_snparray(n, p, undef)
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # intercept

    #construct true model b
    true_b = zeros(p)
    true_b[1:4] = [0.1; 0.25; 0.5; 0.8]
    shuffle!(true_b)
    correct_position = findall(x -> x != 0, true_b)

    #run repeats and save to file
    result = repeat(repeats, x, xbm, z, true_b, correct_position, d, l, debias)
    if debias
        writedlm("./repeats/$d" * "_$repeats", result)
    else
        writedlm("./repeats_nodebias/$d" * "_$repeats", result)
    end
end

run()
