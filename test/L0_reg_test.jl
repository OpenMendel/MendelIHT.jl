@testset "fit normal SnpLinAlg" begin
	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Normal
	l = canonicallink(d())

	#construct SnpArraym, snpmatrix, and non genetic covariate (intercept)
	x = simulate_random_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n)

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(xla, k, d, l)

	# run IHT
	result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l)
	show(result)

	@test length(result.beta) == 10000
	@test count(!iszero, result.beta) == k
	@test result.c[1] != 0
	@test result.k == 10
end

@testset "fit Bernoulli SnpLinAlg" begin
	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Bernoulli
	l = canonicallink(d())

	#construct SnpArraym, snpmatrix, and non genetic covariate (intercept)
	x = simulate_random_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n)

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(xla, k, d, l)

	# run IHT
	result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l)

	@test length(result.beta) == 10000
	@test count(!iszero, result.beta) == k
	@test result.c[1] != 0
	@test result.k == 10
end

@testset "fit Poisson SnpLinAlg" begin
	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Poisson
	l = canonicallink(d())

	#construct SnpArraym, snpmatrix, and non genetic covariate (intercept)
	x = simulate_random_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n)

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(xla, k, d, l)

	# run IHT
	result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l)

	@test length(result.beta) == 10000
	@test count(!iszero, result.beta) == k
	@test result.c[1] != 0
	@test result.k == 10
end

@testset "fit NegativeBinomial SnpLinAlg" begin
	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = NegativeBinomial
	l = LogLink()

	#construct SnpArraym, snpmatrix, and non genetic covariate (intercept)
	x = simulate_random_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n)

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(xla, k, d, l)

	#run result
	result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l)

	@test length(result.beta) == 10000
	@test count(!iszero, result.beta) == k
	@test result.c[1] != 0
	@test result.k == 10
end

@testset "fit with >1 non-genetic covariates SnpLinAlg" begin
	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Normal
	l = canonicallink(d())

	#construct SnpArraym, snpmatrix, and non genetic covariate (intercept)
	x = simulate_random_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n, 2) # 1st column intercept
	z[:, 2] .= randn(n)

	#define true_b and true_c
	true_b = zeros(p)
	true_b[1:k-2] = randn(k-2)
	shuffle!(true_b)
	correct_position = findall(!iszero, true_b)
	true_c = [3.0; 3.5]

	#simulate phenotype
	prob = GLM.linkinv.(l, xla * true_b .+ z * true_c)
	y = [rand(d(i)) for i in prob]

	# run IHT
	result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l)

	@test length(result.beta) == 10000
	@test length(result.c) == 2
	@test length(findall(!iszero, result.beta)) == 10
	@test findall(!iszero, result.c) == [1;2]
	@test result.k == 10
end

@testset "model selection on non-genetic covariates" begin
	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Normal
	l = canonicallink(d())

	#construct SnpArraym, snpmatrix, and non genetic covariate (intercept)
	x = simulate_random_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n, 2) # 1st column intercept
	z[:, 2] .= randn(n)

	# define true_b and true_c
	true_b = zeros(p)
	true_b[1:k-2] = randn(k-2)
	shuffle!(true_b)
	correct_position = findall(!iszero, true_b)
	true_c = [3.0; 0]

	# simulate phenotype
	prob = GLM.linkinv.(l, xla * true_b .+ z * true_c)
	y = [rand(d(i)) for i in prob]

	#run result, keeping only intercept
	zkeep = convert(BitVector, [true, false])
	result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l, zkeep = zkeep)

	@test length(result.beta) == 10000
	@test length(result.c) == 2
	@test length(findall(!iszero, result.beta)) == 10
	@test count(!iszero, result.c) == 1 # the 2nd covariate should be excluded
	@test result.k == 10
end

@testset "Correlated predictors and double sparsity Float64 matrix" begin
	#simulat data with k true predictors, from distribution d and with link l.
    n = 1000
    p = 10000
    d = Normal
	l = canonicallink(d())
    block_size = 20
    num_blocks = Int(p / block_size)

    # assign group membership
    membership = collect(1:num_blocks)
    g = zeros(Int64, p)
    for i in 1:length(membership)
        for j in 1:block_size
            cur_row = block_size * (i - 1) + j
            g[block_size*(i - 1) + j] = membership[i]
        end
    end

    #simulate correlated snparray
    x = simulate_correlated_snparray(undef, n, p)
    z = ones(n) # the intercept
    x_float = convert(Matrix{Float64}, x, model=ADDITIVE_MODEL, center=true, scale=true)

    #simulate true model, where 5 groups each with 3 snps contribute
    true_b = zeros(p)
    true_groups = randperm(num_blocks)[1:5]
    within_group = [randperm(block_size)[1:3] randperm(block_size)[1:3] randperm(block_size)[1:3] randperm(block_size)[1:3] randperm(block_size)[1:3]]
    correct_position = zeros(Int64, 15)
    for i in 1:5
        cur_group = block_size * (true_groups[i] - 1)
        cur_group_snps = cur_group .+ within_group[:, i]
        correct_position[3*(i-1)+1:3i] .= cur_group_snps
    end
    for i in 1:15
        true_b[correct_position[i]] = rand(-1:2:1) * 0.2
    end
    sort!(correct_position)

    # simulate phenotype
    if d == Normal || d == Bernoulli || d == Poisson
        prob = GLM.linkinv.(l, x_float * true_b)
        clamp!(prob, -20, 20)
        y = [rand(d(i)) for i in prob]
    elseif d == NegativeBinomial
        nn = 10
        μ = GLM.linkinv.(l, x_float * true_b)
        clamp!(μ, -20, 20)
        prob = 1 ./ (1 .+ μ ./ nn)
        y = [rand(d(nn, i)) for i in prob] #number of failtures before nn success occurs
    end
    y = Float64.(y)

    #run IHT without groups
    k = 15
    ungrouped = fit_iht(y, x_float, z, J=1, k=k, d=d(), l=l)

    #run IHT with 5 groups each with 3 predictor
    J = 5
    k = 3
    grouped = fit_iht(y, x_float, z, J=J, k=k, d=d(), l=l, group=g)

    @test length(findall(!iszero, ungrouped.beta)) == 15
    @test length(findall(!iszero, grouped.beta)) == 15
end

@testset "Negative binomial nuisance parameter Float32 matrix" begin
	n = 1000
	p = 10000
	k = 10
	d = NegativeBinomial
	l = LogLink()

	# simulate SnpArrays data
	x = simulate_correlated_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n, 1) 
	y, true_b, correct_position = simulate_random_response(xla, k, d, l, r=10);

	@time newton = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l, est_r=:Newton)
	@test typeof(newton.d) == NegativeBinomial{Float64}
	@test newton.d.p == 0.5 # p parameter not used 
	@test newton.d.r ≥ 1 # r converges to 10 faster

	@time mm = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l, est_r=:MM)
	@test typeof(mm.d) == NegativeBinomial{Float64}
	@test mm.d.p == 0.5
	@test mm.d.r ≥ 1 # r converges to 10 slower

	# simulate floating point data
	T = Float32
	x = randn(T, n, p)
	z = ones(T, n) 

	# simulate true model b, and the correct non-0 positions of b
	true_b = zeros(T, p)
	true_b[1:k] .= collect(0.1:0.1:1.0)
	true_c = [T.(4.0)]
	shuffle!(true_b)

	# simulate response
	r = 20
    μ = GLM.linkinv.(l, xla * true_b)
    clamp!(μ, -20, 20)
    prob = 1 ./ (1 .+ μ ./ r)
    y = [rand(d(r, Float64(i))) for i in prob] 
    y = T.(y)

	d = d(r, Float32(0.5)) # need Float32 for eltype of d
	@time newton = fit_iht(y, x, z, J=1, k=k, d=d, l=l, est_r=:Newton)
	@test typeof(newton.d) == NegativeBinomial{Float32}
	@test newton.d.p == 0.5 
	@test newton.d.r ≥ 0

	@time mm = fit_iht(y, x, z, J=1, k=k, d=d, l=l, est_r=:MM)
	@test typeof(mm.d) == NegativeBinomial{Float32}
	@test mm.d.p == 0.5
	@test mm.d.r ≥ 0
end

@testset "initialze beta" begin
	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Normal
	l = canonicallink(d())

	#set random seed
	Random.seed!(1111)

	#construct SnpArraym, snpmatrix, and non genetic covariate (intercept)
	x = simulate_random_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n)

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(xla, k, d, l)

	#run with and without initializing beta
	@time result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l, init_beta=false)
	@time result2 = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l, init_beta=true)
	@test all(findall(!iszero, result.beta) .== findall(!iszero, result2.beta))
end
