@testset "L0_reg normal" begin
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
	x = simulate_random_snparray(n, p, "test1.bed")
	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
	z = ones(n, 1)

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

	#run result
	result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, use_maf=false)

	@test length(result.beta) == 10000
	@test findall(!iszero, result.beta) == [2384;3352;3353;4093;5413;5609;7403;8753;9089;9132]
	@test all(result.beta[findall(!iszero, result.beta)] .≈ [-1.2601406011046452;
	 				-0.2674202492177914; 0.14120810664715883; 0.289955803600036;
	  				0.3666894767520663; -0.1371805027382694; -0.3082545756160329;
	  				0.3328814701200445; 0.9645980728400257; -0.5094607091364866])
	@test result.c[1] == 0.0
	@test result.k == 10
	@test result.logl ≈ -1407.2533232402275
end

@testset "L0_reg Bernoulli" begin
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
	x = simulate_random_snparray(n, p, "test1.bed")
	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
	z = ones(n, 1)

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

	#run result
	result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, use_maf=false)

	@test length(result.beta) == 10000
	@test findall(!iszero, result.beta) == [1733;1816;2384;5413;7067;8753;8908;9089;9132;9765]
	@test all(result.beta[findall(!iszero, result.beta)] .≈ [-0.2787326116508012;
					  0.3113511410050774;-1.1292096054341005;0.5001816459301949;
					 -0.32694130827328116;0.4134742776599116;-0.3275424847038566;
					  0.8619785898062307;-0.5068258295825918;-0.32972421733995294])
	@test result.c[1] == 0.0
	@test result.k == 10
	@test result.logl ≈ -489.8770526620568
end

@testset "L0_reg Poisson" begin
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
	x = simulate_random_snparray(n, p, "test1.bed")
	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
	z = ones(n, 1)

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

	#run result
	result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, use_maf=false)
	@test length(result.beta) == 10000
	@test findall(!iszero, result.beta) == [298; 606; 2384; 5891; 7067; 8753; 8755; 8931; 9089; 9132]
	@test all(result.beta[findall(!iszero, result.beta)] .≈ [0.10999211487301704;
		-0.09628969787009399; -0.3660582504298778;  0.11767397809862554;  
		0.09686501699067837;  0.11419451741236888;  0.12373749347128933;  
		0.11916107757737655;  0.2904980599350941; -0.12920302008477738])
	@test result.c[1] == 0.0
	@test result.k == 10
	@test result.logl ≈ -1293.4456256102478
	@test result.iter == 14
end

@testset "L0_reg NegativeBinomial" begin
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
	x = simulate_random_snparray(n, p, "test1.bed")
	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
	z = ones(n, 1)

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

	#run result
	result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, use_maf=false)

	@test length(result.beta) == 10000
	@test findall(!iszero, result.beta) == [1245; 1774; 1982; 2384; 5413; 5440; 5614; 7166; 9089; 9132;]
	@test all(result.beta[findall(!iszero, result.beta)] .≈ [-0.13251205076442696;
		-0.16906821875893546;  0.12865324984770152; -0.30791709019019947;  
		0.1202449253579259; 0.12545318690748591;  0.10252799982767402;  
		0.12937785947321034;  0.27113364351607033;-0.19797373419860373])
	@test result.c[1] == 0.0
	@test result.k == 10
	@test result.logl ≈ -1387.341396480908
	@test result.iter == 9
end

@testset "L0_reg with non-genetic covariates" begin
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
	x = simulate_random_snparray(n, p, "test1.bed")
	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
	z = ones(n, 2) # the intercept
	z[:, 2] .= randn(n)

	#define true_b and true_c
	true_b = zeros(p)
	true_b[1:k-2] = randn(k-2)
	shuffle!(true_b)
	correct_position = findall(!iszero, true_b)
	true_c = [3.0; 3.5]

	#simulate phenotype
	prob = linkinv.(l, xbm * true_b .+ z * true_c)
	y = [rand(d(i)) for i in prob]

	#run result
	result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, use_maf=false)

	@test length(result.beta) == 10000
	@test length(result.c) == 2
	@test findall(!iszero, result.beta) == [2984;4147;4604;6105;6636;7575;8271;9300]
	@test findall(!iszero, result.c) == [1;2]
	@test all(result.beta[findall(!iszero, result.beta)] .≈ [  1.0813690725698488;
 		-0.21167725381808253; -0.3320058055771102; 0.40296120706642935; 
 		-0.1320095894527104;  1.6694054572246806;  0.3151337065844457; 
 		-1.5967591054279828])
	@test all(result.c .≈ [2.931093515105184; 3.4674609432737893])
	@test result.k == 10
	@test result.logl ≈ -1372.8963328607294

	#clean up
	rm("test1.bed", force=true)
end
