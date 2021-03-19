function test_data(d, l)
    x = simulate_random_snparray(undef, 1000, 1000)
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true)
    z = ones(1000, 1)
    y = rand(1000)
    v = IHTVariable(xla, z, y, 1, 10, d, l, Int[], Float64[], :none)

    return x, z, y, v
end

function make_IHTvar(d, μ, y)
    n = length(μ)
    v = IHTVariable(rand(n, 1), rand(n, 1), y, 1, 10, d, IdentityLink(),
        Int[], Float64[], :none) 
    v.μ = μ
    return v
end

@testset "loglikelihood" begin
    Random.seed!(2019)
    d = Normal()
    μ = rand(1000000)
    y = [rand(Normal(μi, 1)) for μi in μ]
    v = make_IHTvar(d, μ, y) # create IHTVariable for testing

    @test isapprox(MendelIHT.loglikelihood(v), sum(logpdf.(Normal.(μ, 1.0), y)), atol=1e-4)

    d = Bernoulli()
    p = rand(10000)
    y = rand(0.0:1.0, 10000)
    v = make_IHTvar(d, p, y) # create IHTVariable for testing

    @test isapprox(MendelIHT.loglikelihood(v), sum(logpdf.(Bernoulli.(p), y)), atol=1e-8)

    d = Poisson()
    λ = rand(10000)
    y = Float64.(rand(1:100, 10000))
    v = make_IHTvar(d, λ, y) # create IHTVariable for testing

    @test isapprox(MendelIHT.loglikelihood(v), sum(logpdf.(Poisson.(λ), y)), atol=1e-8)

    d = NegativeBinomial()
    p = rand(1000000)
    r = ones(1000000)
    y = Float64.([rand(NegativeBinomial(1.0, p_i)) for p_i in p])
    v = make_IHTvar(d, p, y) # create IHTVariable for testing

    @test isapprox(MendelIHT.loglikelihood(v), sum(logpdf.(NegativeBinomial.(r, r ./ (p .+ r)), y)), atol=1e-5)
end

@testset "deviance" begin
    Random.seed!(2019)
    # Need more test other than normal distribution!!
    d = Normal()
    y = randn(1000)
    μ = randn(1000)
    v = make_IHTvar(d, μ, y) # create IHTVariable for testing
    @test sum(abs2, y .- μ) ≈ MendelIHT.deviance(v)
end

@testset "update_μ!" begin
    Random.seed!(2019)
    d = Normal
    μ = zeros(1000)
    xb = rand(1000)
    MendelIHT.update_μ!(μ, xb, canonicallink(d()))

    @test all(isapprox(μ, xb, atol=1e-8))

    d = Bernoulli
    p = zeros(1000)
    xb = rand(1000)
    MendelIHT.update_μ!(p, xb, canonicallink(d()))

    @test all(isapprox(p, 1 ./ (1 .+ exp.(-1.0 .* xb)), atol=1e-8))

    d = Poisson
    λ = zeros(1000)
    xb = rand(1000)
    MendelIHT.update_μ!(λ, xb, canonicallink(d()))

    @test all(isapprox(λ, (exp.(xb)), atol=1e-8))

    d = NegativeBinomial
    p = zeros(1000)
    xb = rand(1000)
    MendelIHT.update_μ!(p, xb, LogLink()) #use loglink for NegativeBinomial

    @test all(isapprox(p, exp.(xb), atol=1e-8))
end

@testset "update_xb! and check_covariate_supp!" begin
   	Random.seed!(1111)
    x, z, y, v = test_data(Normal(), IdentityLink())
    v.idx[1:10] .= trues(10)
    v.b .= rand(1000)
    tmp_x = convert(Matrix{Float64}, @view(x[:, v.idx]), center=true, scale=true)

    @test size(v.xk, 2) == 9 #IHTVariable is initialized to have k - 1 columns

    MendelIHT.check_covariate_supp!(v)

    @test size(v.xk, 2) == 10 

    MendelIHT.update_xb!(v)

    @test all(v.xb .≈ tmp_x * v.b[1:10])
    @test all(v.zc .== 0.0)
    @test all(abs.(v.b) .<= 30)
end

@testset "score!" begin
    # Not sure how to "really" test this function, so I'm throwing in a lot of input/outputs
    # as they are calculated right now, because the code seems to work well

    Random.seed!(1993)
    d = Normal()
    l = IdentityLink()
    (x, z, y, v) = test_data(d, l)
    score!(v)

    @test all(v.df[1:3] .≈ [8.057330896198419; -2.229495636684423; 9.948528223937274])
    @test v.df2[1] ≈ 500.8665816573597

    (x, z, y, v) = test_data(d, l)
    score!(v)

    @test all(v.df[1:3] .≈ [-12.569757729532215; 8.237144382138629; 9.625154193038197])
    @test v.df2[1] ≈ 503.8433996263735

    (x, z, y, v) = test_data(d, l)
    score!(v)

    @test all(v.df[1:3] .≈ [-2.2634046444623226; -5.266454260596382; -1.5449038662162529])
    @test v.df2[1] ≈ 507.55935638764254
end

@testset "_iht_gradstep!" begin
    Random.seed!(1111)

    x, z, y, v = test_data(Normal(), IdentityLink())
    v.b[1:3] .= [1; 2; 3]
    v.df[1:3] .= [1; 2; 3]
    b = copy(v.b)
    df = copy(v.df)
    J = 1
    k = 2
    η = 0.9
    v.k = k

    MendelIHT._iht_gradstep!(v, η) # this should keep 1 * 2 = 2 elements

    @test v.b[1] ≈ 0.0 # because first entry is smallest, it should be set to 0
    @test v.b[2] ≈ (b + η*df)[2]
    @test v.b[3] ≈ (b + η*df)[3]
    @test all(v.b[4:end] .≈ 0.0)
    @test all(v.c .≈ 0.0)
    @test all(v.df .≈ df)
end

@testset "_choose!" begin
    J = 1
    k = 10

    Random.seed!(1111)
    x, z, y, v = test_data(Normal(), IdentityLink())
    v.idx[1:10] .= trues(10)
    MendelIHT._choose!(v)

    @test all(v.idx[1:10] .== trues(10))
    @test v.idc[1] == false

    v.idx[1:11] .= trues(11)
    MendelIHT._choose!(v)

    @test sum(v.idx[1:11]) == 10

    # set k = 1
    x, z, y, v = test_data(Normal(), IdentityLink())
    v.idc .= true
    v.k = 1
    MendelIHT._choose!(v)

    @test v.idc[1] == true
    @test sum(v.idc) == 1
    @test sum(v.idx) == 0

    x, z, y, v = test_data(Normal(), IdentityLink())
    v.idc .= true
    v.idx[1:10] .= trues(10)
    MendelIHT._choose!(v)

    @test sum(v.idx) + sum(v.idc) == 10
end

@testset "_iht_backtrack" begin
    @test MendelIHT._iht_backtrack_(-150.0, -100.0, 1, 10) == true
    @test MendelIHT._iht_backtrack_(-50.0, -100.0, 1, 10) == false
    @test MendelIHT._iht_backtrack_(150.0, 100.0, 1, 10) == false
    @test MendelIHT._iht_backtrack_(50.0, 100.0, 1, 10) == true
    @test MendelIHT._iht_backtrack_(50.0, 100.0, 0, 3) == true
    @test MendelIHT._iht_backtrack_(50.0, 100.0, 11, 10) == false
    @test MendelIHT._iht_backtrack_(50.0, 100.0, 11, 11) == false
end

@testset "standardize!" begin
	Random.seed!(1017)
    z = rand(1000, 1000)
    standardize!(z)
    @test isapprox(0.0, mean(z[:, 1]), atol=1e-8)
    @test isapprox(0.0, mean(z[:, 10]), atol=1e-8)
    @test isapprox(0.0, mean(z[:, 100]), atol=1e-8)
    @test isapprox(1.0, std(z[:, 1]), atol=1e-8)
    @test isapprox(1.0, std(z[:, 10]), atol=1e-8)
    @test isapprox(1.0, std(z[:, 100]), atol=1e-8)

    z = rand(1000, 1000)
    col1_μ = mean(z[:, 1])
    col1_σ = std(z[:, 1])
    standardize!(@view(z[:, 2:end]))
    @test isapprox(col1_μ, mean(z[:, 1]), atol=1e-8)
    @test isapprox(0.0, mean(z[:, 10]), atol=1e-8)
    @test isapprox(0.0, mean(z[:, 100]), atol=1e-8)
    @test isapprox(col1_σ, std(z[:, 1]), atol=1e-8)
    @test isapprox(1.0, std(z[:, 10]), atol=1e-8)
    @test isapprox(1.0, std(z[:, 100]), atol=1e-8)
end

@testset "project_k!" begin
    x = rand(100000)
    k = 100
    p = sortperm(x, rev = true)
    top_k_index = p[1:k]
    last_k_index = p[k+1:end]
    project_k!(x, k)

    @test all(x[top_k_index] .!= 0.0)
    @test all(x[last_k_index] .== 0.0)
end

# Note: since depending on RNG the groups and x,y vectors will be different,
#       we do not test specific values, but only test properties of result. 
@testset "project_group_sparse!" begin
    #2 active groups, 3 active predictors per group, 10 total predictors
    m, n, k = 2, 3, 10 
    y = randn(k);
    group = rand(1:5, k);
    x = copy(y)
    project_group_sparse!(x, group, m, n)

    non_zero_position = findall(!iszero, x)
    non_zero_entries = x[non_zero_position]
    @test all(non_zero_entries .== y[non_zero_position])
    @test length(non_zero_position) ≤ 6

    # 2 active groups, 50% of active predictor per group, variable group size, 9 total predictors
    J, n = 2, 15
    k = [1, 1, 2, 2, 3]
    y = 5rand(n)
    y_copy = copy(y)
    group = [1; 2; 2; 3; 3; 3; 4; 4; 4; 4; 5; 5; 5; 5; 5] 
    project_group_sparse!(y, group, J, k)
    non_zero_position = findall(!iszero, y)
    non_zero_entries = y[non_zero_position]
    @test all(non_zero_entries .== y_copy[non_zero_position])
    @test length(non_zero_position) ≤ 2 + 3

    # test project_group_sparse! is equivalent to project_k! when max group = 1
    m, n, k = 1, 10, 100000
    group = ones(Int, k) #everybody is in the same group
    y = randn(k)
    x = copy(y)
    project_k!(x, n)
    project_group_sparse!(y, group, m, n)
    @test all(x .== y)
end

@testset "maf_weights" begin
    Random.seed!(33)
    x = simulate_random_snparray(undef, 1000, 10000)
    m = maf(x)
    p = maf_weights(x)

    @test eltype(p) <: AbstractFloat
    @test all(p .>= 1.0)
    @test p[1] ≈ 1 / (2.0sqrt(m[1] * (1 - m[1])))
    @test p[2] ≈ 1 / (2.0sqrt(m[2] * (1 - m[2])))

    p = maf_weights(x, max_weight=2.0)
    @test all(1.0 .<= p .<= 2)
    @test p[1] ≈ 1 / (2.0sqrt(m[1] * (1 - m[1])))
    @test p[15] == 2.0
end

@testset "save_prev!" begin
    Random.seed!(1111)
    x, z, y, v = test_data(Normal(), IdentityLink())

    v.b .= rand(1000)
    v.idx .= bitrand(1000)
    v.idc .= true
    v.c .= rand()

    save_prev!(v)

    @test all(v.b .== v.b0)
    @test all(v.idx .== v.idx0)
    @test all(v.idc .== v.idc0)
    @test all(v.c .== v.c0)
end

@testset "iht_stepsize!" begin
    Random.seed!(1234)
    d = Normal()
    l = canonicallink(d)
    x, z, y, v = test_data(d, l)
    v.df .= rand(1000)
    v.xk .= rand(1000, 9)
    v.μ .= rand(1000)
    v.idx[1:9] .= trues(9)

    @test MendelIHT.iht_stepsize!(v) ≥ 0

    d = Poisson()
    l = canonicallink(d)
    x, z, y, v = test_data(d, l)
    v.df .= rand(1000)
    v.xk .= rand(1000, 9)
    v.μ .= rand(1000)
    v.idx[1:9] .= trues(9)

    @test MendelIHT.iht_stepsize!(v) ≥ 0

    d = NegativeBinomial()
    l = LogLink()
    x, z, y, v = test_data(d, l)
    v.df .= rand(1000)
    v.xk .= rand(1000, 9)
    v.μ .= rand(1000)
    v.idx[1:9] .= trues(9)

    @test MendelIHT.iht_stepsize!(v) ≥ 0

    d = Bernoulli()
    l = canonicallink(d)
    x, z, y, v = test_data(d, l)
    v.df .= rand(1000)
    v.xk .= rand(1000, 9)
    v.μ .= rand(1000)
    v.idx[1:9] .= trues(9)

    @test MendelIHT.iht_stepsize!(v) ≥ 0
end
