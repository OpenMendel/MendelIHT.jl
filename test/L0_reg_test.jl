@testset "fit normal" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

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

	#run result
	result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l)
	show(result)

	@test length(result.beta) == 10000
	@test count(!iszero, result.beta) == k
	@test findall(!iszero, result.beta) == [2384;3352;3353;4093;5413;5609;7403;8753;9089;9132]
	@test all(result.beta[findall(!iszero, result.beta)] .≈ [-1.2601335154934064, -0.26740665871958935, 
		0.14117990315654064, 0.28997442760192266, 0.3667114534086605, -0.13719881241645873, -0.3082574702110094, 
		0.3328995818048077, 0.9645955209824065, -0.5094675198978443])
	@test result.c[1] ≈ -0.026283123286410772
	@test result.k == 10
	@test result.logl ≈ -1406.899627901812
end

@testset "fit Bernoulli" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Bernoulli
	l = canonicallink(d())

	#set random seed
	Random.seed!(1111)

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
	@test findall(!iszero, result.beta) == [1816, 2384, 2917, 5413, 7067, 8753, 8908, 9089, 9132, 9765]
	@test all(result.beta[findall(!iszero, result.beta)] .≈ [0.2927850181979294, -1.1371390185890102, 
		-0.2735411947195354, 0.4777936460488939, -0.310486779584213, 0.41808467980710756, -0.3436991152303386,
		0.8844652838423853, -0.5297482667320204, -0.34131839197378927])
	@test result.c[1] ≈ 0.013569002607988524
	@test result.k == 10
	@test result.logl ≈ -490.2010219408794
end

@testset "fit Poisson" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Poisson
	l = canonicallink(d())

	#set random seed
	Random.seed!(1111)

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
	@test findall(!iszero, result.beta) == [2384, 2631, 3157, 5891, 8753, 8755, 8931, 9089, 9132, 9884]
	@test all(result.beta[findall(!iszero, result.beta)] .≈ [-0.3704728719867845, 0.09550968238347861, 
		0.1012695640714917, 0.12454842375797137, 0.10750826591937211, 0.11630352461070562, 0.12423472147439997, 
		0.2856470190801652, -0.12495713352581836, 0.08889994853065768])
	@test result.c[1] ≈ -0.010771749331668815
	@test result.k == 10
	@test result.logl ≈ -1294.747256187064
end

@testset "fit NegativeBinomial" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = NegativeBinomial
	l = LogLink()

	#set random seed
	Random.seed!(1111)

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
	@test findall(!iszero, result.beta) == [1245; 1774; 1982; 2384; 5413; 5440; 5614; 7166; 9089; 9132;]
	@test all(result.beta[findall(!iszero, result.beta)] .≈ [-0.13536960133758671, -0.17233491524789818, 
		0.13127935664692003, -0.3142802598723658, 0.12254638737412701, 0.1277401870478348, 0.10421067139834642,
		0.13187029926894303, 0.27527241210172276, -0.2019964741423514])
	@test result.c[1] ≈ -0.04109496358855896
	@test result.k == 10
	@test result.logl ≈ -1386.95309289023
end

@testset "fit with non-genetic covariates" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

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

	#run result
	result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l)

	@test length(result.beta) == 10000
	@test length(result.c) == 2
	@test length(findall(!iszero, result.beta)) == 10
	@test findall(!iszero, result.c) == [1;2]
	@test result.k == 10
end

@testset "model selection on non-genetic covariates" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

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

@testset "Correlated predictors and double sparsity" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

	#simulat data with k true predictors, from distribution d and with link l.
    n = 1000
    p = 10000
    d = Normal
	l = canonicallink(d())
    block_size = 20
    num_blocks = Int(p / block_size)

	#set random seed
	Random.seed!(1111)

    # assign group membership
    membership = collect(1:num_blocks)
    g = zeros(Int64, p + 1)
    for i in 1:length(membership)
        for j in 1:block_size
            cur_row = block_size * (i - 1) + j
            g[block_size*(i - 1) + j] = membership[i]
        end
    end
    g[end] = membership[end]
    
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

    #run IHT with groups
    J = 5
    k = 3
    grouped = fit_iht(y, x_float, z, J=J, k=k, d=d(), l=l, group=g)

    @test length(findall(!iszero, ungrouped.beta)) == 15
    @test length(findall(!iszero, grouped.beta)) == 15
end

@testset "Negative binomial nuisance parameter" begin
	n = 1000
	p = 10000
	k = 10
	d = NegativeBinomial
	l = LogLink()

	# set random seed for reproducibility
	Random.seed!(1111) 

	# simulate SnpArrays data
	x = simulate_correlated_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n, 1) 
	y, true_b, correct_position = simulate_random_response(xla, k, d, l, r=10);

	@time newton = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l, est_r=:Newton)
	@test typeof(newton.d) == NegativeBinomial{Float64}
	@test newton.d.p == 0.5 # p parameter not used 
	@test newton.d.r ≥ 7 # r converges to 10 faster

	@time mm = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l, est_r=:MM)
	@test typeof(mm.d) == NegativeBinomial{Float64}
	@test mm.d.p == 0.5
	@test mm.d.r ≥ 2 # r converges to 10 slower

	# simulate floating point data
	Random.seed!(1111) 
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
