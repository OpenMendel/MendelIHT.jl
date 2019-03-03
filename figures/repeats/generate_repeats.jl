using IHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using StatsFuns: logistic
using DelimitedFiles

#run n repeats of L0_reg using the same X and b, but different y
function normal_repeat(
    repeats  :: Int, 
    x        :: SnpArray,
    xbm      :: SnpBitMatrix,
    true_b   :: Vector{Float64},
    noise    :: Float64,
    position :: Vector{Int} #correct position of the true model
)
    n = size(xbm, 1)
    k = size(position, 1)
    estimated_β = zeros(k, repeats)
    
    for i in 1:repeats
        # simulate random noise
        ϵ = rand(Normal(0, noise), n)

        #simulate phenotypes (e.g. vector y) via: y = Xb + noise
        y = xbm * true_b + ϵ

        #compute IHT result for less noisy data
        z = ones(n, 1) # intercept
        result = L0_reg(x, z, y, 1, k, debias=false)
        
        #store the correct position in estimated model
        estimated_β[:, i] .= result.beta[position]
    end
    
    return estimated_β
end

#run n repeats of L0_reg using the same X and b, but different y
function logistic_repeat(
    repeats  :: Int, 
    x        :: SnpArray,
    xbm      :: SnpBitMatrix,
    true_b   :: Vector{Float64},
    noise    :: Float64,
    position :: Vector{Int} #correct position of the true model
)
    n = size(xbm, 1)
    k = size(position, 1)
    estimated_β = zeros(k, repeats)
    
    for i in 1:repeats
        #simulate phenotypes 
        y_temp = xbm * true_b
        
        # Apply inverse logit link and sample from the vector of distributions
        prob = logistic.(y_temp) #inverse logit link
        y = [rand(Bernoulli(x)) for x in prob]
        y = Float64.(y)

        #compute IHT result for less noisy data
        z = ones(n, 1) # intercept
        result = L0_logistic_reg(x, z, y, 1, k, glm = "logistic", debias=true, show_info=false)

        #store the correct position in estimated model
        estimated_β[:, i] .= result.beta[position]
    end
    
    return estimated_β
end

#run n repeats of L0_reg using the same X and b, but different y
function poisson_repeat(
    repeats  :: Int, 
    x        :: SnpArray,
    xbm      :: SnpBitMatrix,
    true_b   :: Vector{Float64},
    noise    :: Float64,
    position :: Vector{Int} #correct position of the true model
)
    n = size(xbm, 1)
    k = size(position, 1)
    estimated_β = zeros(k, repeats)
    
    for i in 1:repeats
        #simulate phenotypes 
        y_temp = xbm * true_b
        
        # Simulate poisson data
        λ = exp.(y_temp) #inverse log link
        y = [rand(Poisson(x)) for x in λ]
        y = Float64.(y)

        #compute IHT result for less noisy data
        z = ones(n, 1) # intercept
        result = L0_poisson_reg(x, z, y, 1, k, glm = "poisson", debias=false, convg=false, show_info=false)

        #store the correct position in estimated model
        estimated_β[:, i] .= result.beta[position]
    end
    
    return estimated_β
end

function run()
    repeats = 10

    #set random seed for reproducibility
    Random.seed!(123)

    #simulat data
    n = 2000
    p = 10000
    bernoulli_rates = 0.5rand(p) #minor allele frequencies are drawn from uniform (0, 0.5)
    x = simulate_random_snparray(n, p, bernoulli_rates)
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

    #specify true model size and noise of data
    k = 10   # number of true predictors
    s = 0.1  # noise 

    #construct true model b
    true_b = zeros(p)       # model vector
    true_b[1:k] = randn(k)  # Initialize k non-zero entries in the true model
    shuffle!(true_b)        # Shuffle the entries
    correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are
    true_b[correct_position]

    #run repeats and save to file
    normal_result = normal_repeat(repeats, x, xbm, true_b, s, correct_position)
    logistic_result = logistic_repeat(repeats, x, xbm, true_b, s, correct_position)
    poisson_result = poisson_repeat(repeats, x, xbm, true_b, s, correct_position)

    writedlm("./repeats/normal_$repeats", normal_result)
    writedlm("./repeats/logistic_$repeats", logistic_result)
    writedlm("./repeats/poisson_$repeats", poisson_result)

    println("completed $repeats repeats.")
end

run()