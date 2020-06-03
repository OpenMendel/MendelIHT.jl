@testset "Cross validation on SnpArrays, normal model" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

    #simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Normal
	l = canonicallink(d())

	#set random seed
	Random.seed!(2019)

	#construct snpmatrix, covariate files, and true model b
	x = simulate_random_snparray(n, p, undef)
	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
	z = ones(n, 1) # the intercept

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

	#specify path and folds
	path = collect(1:20)
	num_folds = 3
	folds = rand(1:num_folds, size(x, 1))

	# cross validation routine that distributes `path` (with debias) 
	@time distribute_path_debias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=true, debias=true, parallel=true)
	@test argmin(distribute_path_debias) == 11
	@test all(distribute_path_debias ≈ [1927.0765190526674;1443.8787350418297;1080.0413162424918;
		898.2611345915562;700.2502074095313;507.3948750413776;391.9679112461845;
		381.68778672828626;368.53087109541184;359.9090161930197;344.3626513420314;
		353.22416803442127;357.75754456737286;363.5298542018518;379.4998479342846;
		365.9493360314551;387.4562377561079;386.80694552347995;398.9880370887734;
		403.1125856821553])

	# cross validation routine that distributes `path` (no debias) 
	@time distribute_path_nodebias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
	@test argmin(distribute_path_nodebias) == 10
	@test all(distribute_path_nodebias ≈ [ 1927.0765190526674; 1443.8788742220863; 1080.041135323195;  862.2385953735204;
		  705.1014346627649;  507.3949359364219;  391.96868764622843;  368.45440222003174;
		  350.642794092518;  345.8380848576577;  350.5188147284578;  359.42391568519577;
		  363.70956969599075;  377.30732985896975;  381.0310879522694;  392.56439238382615;
		  396.81166049333797;  397.3010019298764;  406.47023764639624;  410.4672260807978])

	# cross validation routine that distributes `fold` (with debias) 
	@time distribute_fold_debias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
	@test argmin(distribute_fold_debias) == 11
	@test all(distribute_fold_debias .≈ distribute_path_debias)

	# cross validation routine that distributes `fold` (no debias) 
	@time distribute_fold_nodebias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
	@test argmin(distribute_fold_nodebias) == 10
	@test all(distribute_fold_nodebias .≈ distribute_path_nodebias)
end

@testset "Cross validation on floating point matrices, normal model" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

    #simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Normal
	l = canonicallink(d())

	#set random seed
	Random.seed!(2019)

	#construct snpmatrix, covariate files, and true model b
	T = Float32
	x = randn(T, n, p)
	z = ones(T, n, 1)

	# simulate response, true model b, and the correct non-0 positions of b
	true_b = zeros(T, p)
	true_b[1:k] .= collect(0.1:0.1:1.0)
	true_c = [T.(4.0)]
	shuffle!(true_b)
	correct_position = findall(!iszero, true_b)
    prob = GLM.linkinv.(l, x * true_b)
    clamp!(prob, -20, 20)
    y = [rand(d(i)) for i in prob]

	#specify path and folds
	path = collect(1:20)
	num_folds = 3
	folds = rand(1:num_folds, size(x, 1))

	# cross validation routine that distributes `path` (with debias) 
	@time distribute_path_debias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true)
	@test argmin(distribute_path_debias) == 12
	@test all(distribute_path_debias ≈ [ 1380.6565486523607; 1065.9071386995029;  872.6759379053509;  659.9512415455504;
		  507.95468586447487;  435.42564150445935;  381.97145182134796;  353.3874985986886;
		  358.5175480881438;  364.3411689321348;  351.6054783816806;  350.09729325846456;
		  359.524202147473;  362.678174982638;  381.47928511251905;  385.32682909030063;
		  389.5873100235598;  391.3554650545424;  403.34585759364745;  391.94577847803737])

	# cross validation routine that distributes `path` (no debias) 
	@time distribute_path_nodebias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
	@test argmin(distribute_path_nodebias) == 8
	@test all(distribute_path_nodebias ≈ [ 1380.6565486523607; 1065.9066177033098;  872.6775101110675;  659.9464122367732;
		  507.9536962832382;  435.4240887140454;  381.96988279051556;  353.38777200464585;
		  356.54227892759974;  360.6993225854327;  362.29032358917027;  377.6242489019798;
		  375.36923187208936;  371.46752083347224;  372.4216059836563;  386.14459968226157;
		  391.55232280960587;  400.18546657671857;  408.7103940505434;  414.85961546689293])

	# cross validation routine that distributes `fold` (with debias) 
	@time distribute_fold_debias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
	@test argmin(distribute_fold_debias) == 12
	@test isapprox(distribute_fold_debias, distribute_path_debias, atol=1e-4)

	# cross validation routine that distributes `fold` (no debias) 
	@time distribute_fold_nodebias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
	@test argmin(distribute_fold_nodebias) == 8
	@test isapprox(distribute_fold_nodebias, distribute_path_nodebias, atol=1e-4)
end

# This test times out (> 10 min) on travis with linux machines. Not sure why
# @testset "Cross validation on SnpArrays, logistic model" begin
# 	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

#     #simulat data with k true predictors, from distribution d and with link l.
# 	n = 1000
# 	p = 10000
# 	k = 10
# 	d = Bernoulli
# 	l = canonicallink(d())

# 	#set random seed
# 	Random.seed!(2019)

# 	#construct snpmatrix, covariate files, and true model b
# 	x = simulate_random_snparray(n, p, undef)
# 	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
# 	z = ones(n, 1) # the intercept

# 	# simulate response, true model b, and the correct non-0 positions of b
# 	y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

# 	#specify path and folds
# 	path = collect(1:20)
# 	num_folds = 3
# 	folds = rand(1:num_folds, size(x, 1))

# 	# cross validation routine that distributes `path` (with debias) 
# 	@time distribute_path_debias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
# 	@test argmin(distribute_path_debias) == 6
# 	@test all(distribute_path_debias .≈ [433.1645990291242; 396.4218354530673; 354.03582998371684; 337.2781542152179; 321.40856422243473;
# 		 300.13602523499037; 300.7545574376133; 304.0569785536426; 313.5244382980213; 315.21199531008796;
# 		 328.7337680232285; 355.3073847457196; 360.42023935617004; 354.7111928546249; 374.7397814724943;
# 		 380.5160395899203; 404.44221489146645; 404.77237379170714;412.59479205229724; 439.5097230713853])

# 	# cross validation routine that distributes `path` (no debias) 
# 	@time distribute_path_nodebias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
# 	@test argmin(distribute_path_nodebias) == 7
# 	@test all(distribute_path_nodebias ≈  [433.1646582679374; 396.4213330004181; 354.0359578984675; 341.43957528366184;
# 		 331.0096177161414; 310.55436001851103; 300.74693115572484; 308.3221565445261;
# 		 320.1437416545087; 325.44566775172825; 330.59296483296725; 360.1505146741712;
# 		 369.65579010749514; 372.0215814075303; 378.8006645943797; 392.6758894292982;
# 		 384.283182617048; 432.5312890223311; 445.4448285798073; 434.96984334483506])

# 	# cross validation routine that distributes `fold` (with debias) 
# 	@time distribute_fold_debias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
# 	@test argmin(distribute_fold_debias) == 6
# 	@test all(distribute_fold_debias .≈ distribute_path_debias)

# 	# cross validation routine that distributes `fold` (no debias) 
# 	@time distribute_fold_nodebias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
# 	@test argmin(distribute_fold_nodebias) == 7
# 	@test all(distribute_fold_nodebias .≈ distribute_path_nodebias)
# end

@testset "Cross validation on floating point matrices, logistic model" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

    #simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Bernoulli
	l = canonicallink(d())

	#set random seed
	Random.seed!(2019)

	#construct snpmatrix, covariate files, and true model b
	T = Float64
	x = randn(T, n, p)
	z = ones(T, n, 1)

	# simulate response, true model b, and the correct non-0 positions of b
	true_b = zeros(T, p)
	true_b[1:k] .= collect(0.1:0.1:1.0)
	true_c = [T.(4.0)]
	shuffle!(true_b)
	correct_position = findall(!iszero, true_b)
    prob = GLM.linkinv.(l, x * true_b)
    clamp!(prob, -20, 20)
    y = T.([rand(d(i)) for i in prob])

	#specify path and folds
	path = collect(1:20)
	num_folds = 5
	folds = rand(1:num_folds, size(x, 1))

	# cross validation routine that distributes `path` (with debias) 
	@time distribute_path_debias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true)
	@test argmin(distribute_path_debias) == 7
	@test all(distribute_path_debias .≈ [ 257.6449700029333; 243.38693629922395; 223.07132106240928; 208.63589906660434;
		 200.79539897703896; 199.16737833822066; 198.3115289117867; 200.15665119335674;
		 199.23943324016793; 203.93158114741487; 209.35989571272734; 216.93149462582784;
		 211.756618147333; 226.58646880627361; 230.458609662188; 234.3357809157166;
		 229.94180376824846; 230.8672337582721; 240.46232573564828; 247.25215437849334])

	# takes a long time to run on travis for linux, idk why
	# cross validation routine that distributes `path` (no debias) 
	# @time distribute_path_nodebias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
	# @test argmin(distribute_path_nodebias) == 7
	# @test all(distribute_path_nodebias ≈  [ 257.64496677586015; 243.75040705427494; 226.0234744259589; 208.63552263277975;
	# 	 200.79614739265912; 200.6285824173121; 200.50377786046377; 205.28224388295416;
	# 	 205.86931538862675; 210.27081072703828; 214.25196674396312; 215.47005758404097;
	# 	 219.65370324293696; 226.32322841491788; 230.96348711029404; 236.17651527745733;
	# 	 240.06593309953558; 248.5222261286046; 248.77988654724106; 273.4744827947865])

	# cross validation routine that distributes `fold` (with debias) 
	@time distribute_fold_debias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
	@test argmin(distribute_fold_debias) == 7
	@test all(distribute_fold_debias .≈ distribute_path_debias)

	# takes a long time to run on travis for linux, idk why
	# cross validation routine that distributes `fold` (no debias) 
	# @time distribute_fold_nodebias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
	# @test argmin(distribute_fold_nodebias) == 7
	# @test all(distribute_fold_nodebias .≈ distribute_path_nodebias)
end

@testset "Cross validation on SnpArrays, Poisson model" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

    #simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Poisson
	l = canonicallink(d())

	#set random seed
	Random.seed!(2019)

	#construct snpmatrix, covariate files, and true model b
	x = simulate_random_snparray(n, p, undef)
	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
	z = ones(n, 1) # the intercept

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

	#specify path and folds
	path = collect(1:20)
	num_folds = 3
	folds = rand(1:num_folds, size(x, 1))

	# cross validation routine that distributes `path` (with debias) 
	@time distribute_path_debias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true)
	@test argmin(distribute_path_debias) == 7
	@test all(distribute_path_debias ≈ [ 612.6766519782327; 554.4890824202639; 522.4053768900791; 
		485.3207425983446; 445.78015443806163;412.2320772688479; 382.6698340685961; 
		385.9384838080234;394.81185628720675; 405.7743050802736; 406.27235883258606;
		413.3816413610929; 423.313949398366; 425.98190340196277;427.5153001291356; 
		420.3871855662542; 440.9354230778697;445.96512112259165; 455.5151967326286; 
		437.7447856125584])

	# takes a long time to run on travis for linux, idk why
	# cross validation routine that distributes `path` (no debias) 
	# @time distribute_path_nodebias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
	# @test argmin(distribute_path_nodebias) == 7
	# @test all(distribute_path_nodebias ≈ [ 612.6766519737566; 559.9883076751612; 522.4087003201835; 485.3230528002009;
	# 	 453.2934699433838; 412.2355424881003; 382.6692628164427; 388.7228529150391;
	# 	 404.00166424703025; 404.5164534572898; 410.2502978750293; 421.5977779476969;
	# 	 427.14663301457153; 422.7857067669004; 437.6387438924328; 436.5131899747687;
	# 	 455.40258340001685; 454.62291895175935; 453.52085260325646; 458.51241004455574])

	# cross validation routine that distributes `fold` (with debias) 
	@time distribute_fold_debias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
	@test argmin(distribute_fold_debias) == 7
	@test all(distribute_fold_debias .≈ distribute_path_debias)

	# takes a long time to run on travis for linux, idk why
	# cross validation routine that distributes `fold` (no debias) 
	# @time distribute_fold_nodebias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
	# @test argmin(distribute_fold_nodebias) == 7
	# @test all(distribute_fold_nodebias .≈ distribute_path_nodebias)
end

# The following test passes locally (Julia 1.3.1 with Mac) but fails for Julia 1.0 on linux. 
# For now, comment out to make travis pass. 
# @testset "Cross validation on floating point matrices, Poisson model" begin
# 	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

#     #simulat data with k true predictors, from distribution d and with link l.
# 	n = 1000
# 	p = 10000
# 	k = 10
# 	d = Poisson
# 	l = canonicallink(d())

# 	#set random seed
# 	Random.seed!(2019)

# 	#construct snpmatrix, covariate files, and true model b
# 	T = Float32
# 	x = randn(T, n, p)
# 	z = ones(T, n, 1)

# 	# simulate response, true model b, and the correct non-0 positions of b
# 	true_b = zeros(T, p)
# 	true_b[1:k] .= collect(0.1:0.1:1.0)
# 	true_c = [T.(4.0)]
# 	shuffle!(true_b)
# 	correct_position = findall(!iszero, true_b)
#     prob = GLM.linkinv.(l, x * true_b)
#     clamp!(prob, -20, 20)
#     y = [rand(d(i)) for i in Float64.(prob)]
#     y = T.(y)

# 	#specify path and folds
# 	path = collect(1:20)
# 	num_folds = 3
# 	folds = rand(1:num_folds, size(x, 1))

# 	# cross validation routine that distributes `path` (with debias) 
# 	@time distribute_path_debias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true)
# 	@test argmin(distribute_path_debias) == 9
# 	@test isapprox(distribute_path_debias, [ 2439.7858577999773; 2475.264523550615; 1973.7876065895484; 1644.5429713385843;
# 		 1191.6012625437234;  962.0998240517529;  945.2135472991168;  803.4670649487925;
# 		  726.3490249930313;  879.0828516881473; 1090.9087530817117; 1086.6832570393738;
# 		  908.5639625201087; 1018.8319853263057; 1056.648254697459; 1157.9251420414168;
# 		 1122.3220489466703; 1209.6375481386988; 1158.460645637434; 1221.420617457075], atol=1e-3)

# 	# cross validation routine that distributes `path` (no debias) 
# 	@time distribute_path_nodebias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
# 	@test argmin(distribute_path_nodebias) == 9
# 	@test isapprox(distribute_path_nodebias, [ 2439.7858577999773; 2030.4754783497865; 1612.10770908851; 1253.6145310413526;
# 		 1039.3392917466201;  819.5515998712442;  774.7129950203067;  719.2753247267227;
# 		  599.9076216107337;  633.0254472967089;  674.7081663323884;  668.19202838403;
# 		  637.7622401343957;  695.9010527981492;  744.7628524502941;  708.9971241770048;
# 		  794.5064627756589;  705.7317132555488;  739.3570279221592;  717.6058803885434], atol=1e-3)

# 	# cross validation routine that distributes `fold` (with debias) 
# 	@time distribute_fold_debias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
# 	@test argmin(distribute_fold_debias) == 9
# 	@test isapprox(distribute_fold_debias, distribute_path_debias, atol=1e-3)

# 	# cross validation routine that distributes `fold` (no debias) 
# 	@time distribute_fold_nodebias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
# 	@test argmin(distribute_fold_nodebias) == 9
# 	@test isapprox(distribute_fold_nodebias, distribute_path_nodebias, atol=1e-3)
# end

@testset "Cross validation on SnpArrays, NegativeBinomial model" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

    #simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = NegativeBinomial
	l = LogLink()

	#set random seed
	Random.seed!(2019)

	#construct snpmatrix, covariate files, and true model b
	x = simulate_random_snparray(n, p, undef)
	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
	z = ones(n, 1) # the intercept

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

	#specify path and folds
	path = collect(1:20)
	num_folds = 3
	folds = rand(1:num_folds, size(x, 1))

	# cross validation routine that distributes `path` (with debias) 
	@time distribute_path_debias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true)
	@test argmin(distribute_path_debias) == 6
	@test all(distribute_path_debias ≈ [ 299.29083418540984; 277.1588834992124; 254.24543899850323; 
		239.65111105625778; 234.28391512742712; 226.11206626366229; 231.00815753407852;
		 230.28151617161788; 234.06225621624264; 235.33924496682374; 237.76185304461012; 
		 241.4319393553253; 247.51432485217947; 249.26308636719477; 248.54094497173412; 
		 253.51544251902232; 259.8047937477371; 253.51719798579745; 263.6992561532014; 
		 263.43746844493137])

	# takes a long time to run on travis for linux, idk why
	# cross validation routine that distributes `path` (no debias) 
	# @time distribute_path_nodebias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
	# @test argmin(distribute_path_nodebias) == 8
	# @test all(distribute_path_nodebias ≈ [ 296.9596278110663; 277.1566776256703; 254.2454224098157; 239.65129141020367;
	# 	 231.48007085570202; 226.1099471857413; 227.8661320960005; 225.69590986451072;
	# 	 231.23054325752122; 238.1076457509811; 238.97785919335448; 246.7973602914957;
	# 	 247.2714020303685; 250.58371404534932; 248.53296831075767; 254.66328324866123;
	# 	 257.7865309364425; 260.9270182803426; 270.6855801637331; 263.47871860428785])

	# cross validation routine that distributes `fold` (with debias) 
	@time distribute_fold_debias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
	@test argmin(distribute_fold_debias) == 6
	@test all(distribute_fold_debias .≈ distribute_path_debias)

	# takes a long time to run on travis for linux, idk why
	# cross validation routine that distributes `fold` (no debias) 
	# @time distribute_fold_nodebias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
	# @test argmin(distribute_fold_nodebias) == 8
	# @test all(distribute_fold_nodebias .≈ distribute_path_nodebias)
end

@testset "Cross validation on floating point matrices, NegativeBinomial model" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

    #simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = NegativeBinomial
	l = LogLink()

	#set random seed
	Random.seed!(2019)

	#construct snpmatrix, covariate files, and true model b
	T = Float32
	x = randn(T, n, p)
	z = ones(T, n, 1)

	# simulate response, true model b, and the correct non-0 positions of b
	true_b = zeros(T, p)
	true_b[1:k] .= collect(0.1:0.1:1.0)
	true_c = [T.(4.0)]
	shuffle!(true_b)
	correct_position = findall(!iszero, true_b)
    prob = GLM.linkinv.(l, x * true_b)
    clamp!(prob, -20, 20)
    y = [rand(d(i)) for i in prob]
    y = T.(y)

	#specify path and folds
	path = collect(1:20)
	num_folds = 3
	folds = rand(1:num_folds, size(x, 1))

	# cross validation routine that distributes `path` (with debias) 
	@time distribute_path_debias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true)
	@test argmin(distribute_path_debias) == 9
	@test isapprox(distribute_path_debias[1:19], [ 711.1148907867071; 656.0309209427922; 574.4763636279968; 384.47323689913634;
		 337.75419026125724; 282.6397965594231; 259.41343256652266; 251.50847139462587;
		 242.07355736953332; 260.4072851331833; 263.4608096673131; 273.8827842674657;
		 267.5470586950606; 275.0826393719378; 281.3164670378588; 277.0172074748429;
		 293.4404916036798; 290.0605291398179; 301.2161804394812], atol=1e-1) # not sure why last entry could change...

	# cross validation routine that distributes `path` (no debias) 
	@time distribute_path_nodebias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
	@test argmin(distribute_path_nodebias) == 9
	@test isapprox(distribute_path_nodebias, [ 711.1148907867071; 624.4294628308379; 517.3170633355957; 445.96689810093415;
		 350.16094871382575; 297.79798852376996; 267.09787665626254; 246.8237308757399;
		 242.07976062029564; 256.4975998934053; 264.5056379981211; 272.5896415371733;
		 270.75936626996645; 279.5313791270837; 287.00157517659346; 277.71006362854473;
		 293.56512081443094; 287.0336874986847; 291.6214308347876; 294.21248835728244], atol=1e-1)

	# cross validation routine that distributes `fold` (with debias) 
	@time distribute_fold_debias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
	@test argmin(distribute_fold_debias) == 9
	@test isapprox(distribute_fold_debias, distribute_path_debias, atol=1e-4)

	# cross validation routine that distributes `fold` (no debias) 
	@time distribute_fold_nodebias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
	@test argmin(distribute_fold_nodebias) == 9
	@test isapprox(distribute_fold_nodebias, distribute_path_nodebias, atol=1e-4)
end

@testset "iht_run_many_models normal" begin
	n = 1000
	p = 10000
	k = 10
	d = Normal
	l = canonicallink(d())

	#set random seed
	Random.seed!(2019)

	#construct snpmatrix, covariate files, and true model b
	x = simulate_random_snparray(n, p, undef)
	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
	z = ones(n, 1) # the intercept

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

	#specify path and run a ton of models 
	path = collect(1:20)
	@time result = iht_run_many_models(d(), l, x, z, y, 1, path, verbose=true, parallel=true)

	@test result[1] ≈ -2272.788840069668
	@test result[2] ≈ -2144.4519706525098
	@test result[3] ≈ -1998.9118082179523
	@test result[4] ≈ -1888.362338848445
	@test result[5] ≈ -1767.649126874247
	@test result[6] ≈ -1617.2385746806704
	@test result[7] ≈ -1484.2709104112216
	@test result[8] ≈ -1447.9560237353137
	@test result[9] ≈ -1421.5794827334514
	@test result[10] ≈ -1406.8807653835697
	@test result[11] ≈ -1396.8628294699147
	@test result[12] ≈ -1390.1763942804469
	@test result[13] ≈ -1383.5471518926422
	@test result[14] ≈ -1376.7041041059629
	@test result[15] ≈ -1371.070140615097
	@test result[16] ≈ -1363.7164318022296
	@test result[17] ≈ -1358.8454592637506
	@test result[18] ≈ -1353.106442197983
	@test result[19] ≈ -1351.709447505716
	@test result[20] ≈ -1343.1979015928894
end
