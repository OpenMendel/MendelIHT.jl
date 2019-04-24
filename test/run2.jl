function run_poisson(n :: Int64, p :: Int64)
    k = 10

    #set random seed
    Random.seed!(1111)

    x, maf = simulate_random_snparray(n, p)
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # non-genetic covariates, just the intercept
    true_b = zeros(p)
    true_b[1:k] = rand(Normal(0, 0.4), k)
    shuffle!(true_b)
    correct_position = findall(x -> x != 0, true_b)

    #simulate phenotypes under different noises by: y = Xb + noise
    y_temp = xbm * true_b

    # Simulate poisson data
    λ = exp.(y_temp) #inverse log link
    y = [rand(Poisson(x)) for x in λ]
    y = Float64.(y)

    #compute poisson IHT result
    result = L0_poisson_reg(x, z, y, 1, k, glm = "poisson", debias=true, convg=false, show_info=false, true_beta=true_b, scale=false, init=false)
    result2 = L0_poisson_reg2(x, z, y, 1, k, glm = "poisson", debias=true, convg=false, show_info=false, true_beta=true_b, scale=false, init=false)

    #check result
    old_beta = result.beta[correct_position]
    new_beta = result2.beta[correct_position]
    true_model = true_b[correct_position]
    compare_model = DataFrame(
        true_β           = true_model, 
        β_old_backtract  = old_beta,
        β_new_backtract  = new_beta)
    
    #display results
    @show compare_model
    println("Old iteration number was " * string(result.iter))
    println("Old time was " * string(result.time))
    println("New iteration number was " * string(result2.iter))
    println("New time was " * string(result2.time) * "\n\n")

    found_new_backtract = length(findall(!iszero, new_beta))
    found_old_backtract = length(findall(!iszero, old_beta))
    new_backtract_iter = result2.iter
    old_backtract_iter = result.iter

    return found_new_backtract, found_old_backtract, new_backtract_iter, old_backtract_iter
end

Random.seed!(2019)
function test_poisson()
    total_found_new_backtract = 0
    total_found_old_backtract = 0
    total_iter_new_backtract = 0
    total_iter_old_backtract = 0
    for i = 1:25
        @info("running the $i th model")
        n = rand(500:2000) 
        p = rand(1:10)n
        println("n, p = " * string(n) * ", " * string(p))
        fnb, fob, nbi, obi = run_poisson(n, p)
        total_found_new_backtract += fnb
        total_found_old_backtract += fob
        total_iter_new_backtract += nbi
        total_iter_old_backtract += obi
    end
    println("New backtrack: found a total of $total_found_new_backtract predictors and total iteration was $total_iter_new_backtract")
    println("Old backtrack: found a total of $total_found_old_backtract predictors and total iteration was $total_iter_old_backtract")
end
test_poisson()



function run_normal(n :: Int64, p :: Int64)
    k = 10

    #set random seed
    Random.seed!(1111)

    x, maf = simulate_random_snparray(n, p)
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # non-genetic covariates, just the intercept
    true_b = zeros(p)
    true_b[1:k] = randn(k)
    shuffle!(true_b)
    correct_position = findall(x -> x != 0, true_b)
    noise = rand(Normal(0, 0.1), n) # noise vectors from N(0, s) 

    #simulate phenotypes under different noises by: y = Xb + noise
    y = xbm * true_b + noise

    #compute poisson IHT result
    result = L0_normal_reg(x, z, y, 1, k, debias=true)
    # result2 = L0_normal_reg2(x, z, y, 1, k, debias=false)

    #check result
    # old_beta = result2.beta[correct_position]
    new_beta = result.beta[correct_position]
    true_model = true_b[correct_position]
    compare_model = DataFrame(
        true_β           = true_model, 
        # β_old_backtract  = old_beta,
        β_new_backtract  = new_beta)
    
    #display results
    @show compare_model
    # println("Old iteration number was " * string(result2.iter))
    # println("Old time was " * string(result2.time))
    println("New iteration number was " * string(result.iter))
    println("New time was " * string(result.time) * "\n\n")

    found_new_backtract = length(findall(!iszero, new_beta))
    # found_old_backtract = length(findall(!iszero, old_beta))
    new_backtract_iter = result.iter
    # old_backtract_iter = result2.iter

    # return found_new_backtract, found_old_backtract, new_backtract_iter, old_backtract_iter
    return found_new_backtract, new_backtract_iter
end

Random.seed!(2019)
function test_normal()
    total_found_new_backtract = 0
    # total_found_old_backtract = 0
    total_iter_new_backtract = 0
    # total_iter_old_backtract = 0
    for i = 1:25
        @info("running the $i th model")
        n = rand(500:2000) 
        p = rand(1:10)n
        println("n, p = " * string(n) * ", " * string(p))
        # fnb, fob, nbi, obi = run_normal(n, p)
        fnb, nbi = run_normal(n, p)
        total_found_new_backtract += fnb
        # total_found_old_backtract += fob
        total_iter_new_backtract += nbi
        # total_iter_old_backtract += obi
    end
    println("New backtrack: found a total of $total_found_new_backtract predictors and total iteration was $total_iter_new_backtract")
    # println("Old backtrack: found a total of $total_found_old_backtract predictors and total iteration was $total_iter_old_backtract")
end
test_normal()






function run_logistic(n :: Int64, p :: Int64)
    k = 10

    #set random seed
    Random.seed!(1111)

    x, maf = simulate_random_snparray(n, p)
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # non-genetic covariates, just the intercept
    true_b = zeros(p)
    true_b[1:k] = randn(k)
    shuffle!(true_b)
    correct_position = findall(x -> x != 0, true_b)

    #simulate bernoulli data
    y_temp = xbm * true_b
    prob = logistic.(y_temp) #inverse logit link
    y = [rand(Bernoulli(x)) for x in prob]
    y = Float64.(y)

    #compute poisson IHT result
    result = L0_logistic_reg(x, z, y, 1, k, glm = "logistic", debias=true, show_info=false, convg=true,init=false)
    result2 = L0_logistic_reg2(x, z, y, 1, k, glm = "logistic", debias=true, show_info=false, convg=true,init=false)

    #check result
    old_beta = result.beta[correct_position]
    new_beta = result2.beta[correct_position]
    true_model = true_b[correct_position]
    compare_model = DataFrame(
        true_β           = true_model, 
        β_old_backtract  = old_beta,
        β_new_backtract  = new_beta)
    
    #display results
    @show compare_model
    println("Old iteration number was " * string(result.iter))
    println("Old time was " * string(result.time))
    println("New iteration number was " * string(result2.iter))
    println("New time was " * string(result2.time) * "\n\n")

    found_new_backtract = length(findall(!iszero, new_beta))
    found_old_backtract = length(findall(!iszero, old_beta))
    new_backtract_iter = result2.iter
    old_backtract_iter = result.iter

    return found_new_backtract, found_old_backtract, new_backtract_iter, old_backtract_iter
end

Random.seed!(2019)
function test_logistic()
    total_found_new_backtract = 0
    total_found_old_backtract = 0
    total_iter_new_backtract = 0
    total_iter_old_backtract = 0
    for i = 1:25
        @info("running the $i th model")
        n = rand(500:2000) 
        p = rand(1:10)n
        println("n, p = " * string(n) * ", " * string(p))
        fnb, fob, nbi, obi = run_logistic(n, p)
        total_found_new_backtract += fnb
        total_found_old_backtract += fob
        total_iter_new_backtract += nbi
        total_iter_old_backtract += obi
    end
    println("New backtrack: found a total of $total_found_new_backtract predictors and total iteration was $total_iter_new_backtract")
    println("Old backtrack: found a total of $total_found_old_backtract predictors and total iteration was $total_iter_old_backtract")
end
test_logistic()

