using IHT, SnpArrays, MendelBase, CSV, DataFrames

srand(2018)

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

function test_data()
	# dataset with 2 SNP and 6 people. The SNP matrix is 6x3 (with column of intercept)
	x = SnpArray("test") 
	y = CSV.read("test.fam", delim = ' ', header = false)
	y = convert(Array{Float64,1}, y[:, 6])
	k = 2
	v = IHTVariables(x, y, k)
	return (x, y, k, v)
end

function gwas1_data()
	# dataset with 10000 SNP and 2200 people. The SNP matrix is 2200x10001 (with column of intercept)
	x = SnpArray("gwas 1 data") 
	y = CSV.read("gwas 1 data_kevin.fam", delim = ',', header = false) # same file, comma separated
	y = convert(Array{Float64,1}, y[:, 6])
	k = 10
	v = IHTVariables(x, y, k)
	return (x, y, k, v)
end


@testset "initilize IHTVariables" begin
	(x, y, k, v) = test_data()

	#k must be a positive integer
	@test_throws(AssertionError, IHTVariables(x, y, 0))
	@test_throws(AssertionError, IHTVariables(x, y, -1))
	@test_throws(MethodError, IHTVariables(x, y, 1.1))
	@test_throws(MethodError, IHTVariables(x, y, NaN))
	@test_throws(MethodError, IHTVariables(x, y, missing))
	@test_throws(MethodError, IHTVariables(x, y, Inf))

	#Different types of inputs for IHTVariables(x, y, k) is 
	@test typeof(v) == IHTVariable{eltype(y), typeof(y)}
	@test typeof(x) == SnpData || typeof(x) <: SnpArray

	@test size(v.b)    == (3,) 
	@test size(v.b0)   == (3,)
	@test size(v.xb)   == (6,)
	@test size(v.xb0)  == (6,)
	@test size(v.xk)   == (6, 2)
	@test size(v.gk)   == (2,)
	@test size(v.xgk)  == (6,)
	@test size(v.idx)  == (3,)
	@test size(v.idx0) == (3,)
	@test size(v.r)	   == (6,)
	@test size(v.df)   == (3,)

	@test typeof(v.b)    == Array{Float64, 1}
	@test typeof(v.b0)   == Array{Float64, 1}
	@test typeof(v.xb)   == Array{Float64, 1}
	@test typeof(v.xb0)  == Array{Float64, 1}
	@test typeof(v.xk)   == SnpArray{2}
	@test typeof(v.gk)   == Array{Float64, 1}
	@test typeof(v.xgk)  == Array{Float64, 1}
	@test typeof(v.idx)  == BitArray{1}
	@test typeof(v.idx0) == BitArray{1}
	@test typeof(v.r)	 == Array{Float64, 1}
	@test typeof(v.df)   == Array{Float64, 1}
end

@testset "_iht_indices" begin
	(x, y, k, v) = gwas1_data()

	# if v.idx is zero vector (i.e. first iteration of L0_reg), running _iht_indices should 
	# set v.idx = 1 for the k largest terms in v.df 
	v.df[1:10001] .= rand(10001)
	p = sortperm(v.df, rev = true)
	top_k_index = p[1:10]
	IHT._iht_indices(v, k)

	@test all(v.idx[top_k_index] .== 1)

	# if v.idx is not zero vector, then _iht_indices should find non-0 entries of b and 
	# set v.idx = 1 for those entries 
	v.b[1:k] .= rand(10)
	shuffle!(v.b)
	p = sortperm(v.b, rev = true)
	top_k_index = p[1:10]
	IHT._iht_indices(v, k)

	@test all(v.idx[top_k_index] .== 1)
end

@testset "project_k!" begin
    x = rand(100000)
    k = 100
    p = sortperm(x, rev = true)
    top_k_index = p[1:k]
	last_k_index = p[k+1:end]
	IHT.project_k!(x, k)

	@test all(x[top_k_index] .!= 0.0)
	@test all(x[last_k_index] .== 0.0)
end

@testset "use_A2_as_minor_allele" begin
	(x, y, k, v) = test_data()
    result = use_A2_as_minor_allele(x)
    answer = [[0.0 1.0]; [1.0 1.0]; [2.0 0.0]; [1.0 2.0]; [2.0 1.0]; [2.0 2.0]]

    @test all(result .== answer)
end

@testset "std_reciprocal" begin
	# first compute the correct answer by converting each column to floats and call std() directly
	x       = SnpArray("gwas 1 data")
	n, p    = size(x)
	storage = zeros(n)
	answer  = zeros(p)
    for i in 1:p
    	copy!(storage, view(x, :, i))
        answer[i] = std(storage)
    end
    answer .= 1.0 ./ answer

    # next compute the mean of snps since we need it for std_reciprocal()
    mean_vec = zeros(p) 
    maf, minor_allele, missings_per_snp, missings_per_person = summarize(x)
    for i in 1:p
        minor_allele[i] ? mean_vec[i] = 2.0 - 2.0maf[i] : mean_vec[i] = 2.0maf[i] 
    end
    std_vector = std_reciprocal(x, mean_vec)

    @test all(std_vector .≈ answer)
end

@testset "double_sparse_projection" begin
    m, n, k = 2, 3, 20
	y = randn(k);
	group = rand(1:5, k);
	x = doubly_sparse_projection(y, group, m, n);

	# view result easily:
	# for i = 1:length(x)
	#     println(i,"  ",group[i],"  ",y[i],"  ",x[i])
	# end

	non_zero_position = find(x)
	non_zero_entries = x[non_zero_position]
	@test all(non_zero_entries .== y[non_zero_position]) 
	@test all(non_zero_position .== [3; 5; 9; 10; 13; 19])
	@test all(non_zero_entries .≈ [1.4112006668065633; 0.43107790792018497; 
								   1.6657853439113885; 2.164461443215743;
								   2.1341501862143644; 1.7651975436622682])
end

# @testset "_iht_backtrack" begin
#    (x, y, k, v) = gwas1_data()
#    μ, ω = 1.0, 1.0
#    @test IHT._iht_backtrack(v, ω, μ) == false

#    μ, ω = 0.2, 0.3
#    @test IHT._iht_backtrack(v, ω, μ) == true

#    μ, ω = 0.8, 0.792
#    @test IHT._iht_backtrack(v, ω, μ) == false

#    μ, ω = 0.5, 0.2
#    @test IHT._iht_backtrack(v, ω, μ) == false

#    μ, ω = 0.98, 1.0
#    @test IHT._iht_backtrack(v, ω, μ) == true
# end

# @testset "_iht_omega" begin
# 
# end

#@testset "_iht_gradstep" begin
#    (x, y, k, v) = test_data()
#    v.b .= rand(3)
#    v.df .= rand(3)
#    b = copy(v.b)
#    df = copy(v.df)
#    k = 2
#    μ = 0.9
#
#    IHT._iht_gradstep(v, μ, k)
#    @test v.b[1] == 0.0 # because first entry is smallest, it should be set to 0
#    @test v.b[2] == (b + μ*df)[2]
#    @test v.b[3] == (b + μ*df)[3]
#end

