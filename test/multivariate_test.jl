function make_mIHTvar(r) # r = number of traits
    n = 1000  # number of samples
    p = 10000 # number of SNPs
    k = 10    # number of causal SNPs
    
    # set random seed for reproducibility
    Random.seed!(2021)
    
    # simulate `.bed` file with no missing data
    x = simulate_random_snparray(undef, n, p)
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
    
    # intercept is the only nongenetic covariate
    z = ones(n, 1)
    
    # simulate response y, true model b, and the correct non-0 positions of b
    Y, true_Σ, true_b, correct_position = simulate_random_response(xla, k, r);
    
    # everything is transposed for multivariate analysis!
    Yt = Matrix(Y')
    Zt = Matrix(z')
    v = MendelIHT.mIHTVariable(Transpose(xla), Zt, Yt, k)

    return v, Transpose(xla), Yt, Zt, k
end

@testset "update_support!" begin
    idx = BitVector([false, false, true, true, false])
    b = rand(5, 5)
    MendelIHT.update_support!(idx, b)
    @test all(idx)

    b[:, 1] .= 0.0
    MendelIHT.update_support!(idx, b)
    @test idx[1] == false
    @test all(idx[2:5] .== true)

    b[3, 3] = 0.0
    MendelIHT.update_support!(idx, b)
    @test idx[1] == false
    @test all(idx[2:5] .== true)

    b[:, 3] .= 0.0
    MendelIHT.update_support!(idx, b)
    @test idx[1] == false
    @test idx[2] == true
    @test idx[3] == false
    @test idx[4] == true
    @test idx[5] == true
end

@testset "benchmarks for multivariate function" begin

    # r = 2 traits
    v, xla, Yt, Zt, k = make_mIHTvar(2)
    MendelIHT.init_iht_indices!(v)
    @btime MendelIHT.score!($v)           # 120.270 ms (7 allocations: 352 bytes) -> this step takes so long other functions doesn't really matter
    @btime MendelIHT.loglikelihood($v)    # 3.377 μs (2 allocations: 208 bytes)
    @btime MendelIHT.update_xb!($v)       # 187.489 μs (8 allocations: 656 bytes)
    @btime MendelIHT.update_μ!($v)        # 2.337 μs (0 allocations: 0 bytes)
    @btime MendelIHT._choose!($v)         # 16.698 μs (3 allocations: 208 bytes)
    @btime MendelIHT.solve_Σ!($v)         # 2.889 μs (2 allocations: 48 bytes)
    # @btime MendelIHT.iht_stepsize(v) # cannot be done because BenchmarkTools require multiple evaluations of the same function call, which makes Γ not pd
    @btime MendelIHT.vectorize!($(v.full_b), $(v.B), $(v.C)) # 10.299 μs (0 allocations: 0 bytes)
    @btime MendelIHT.unvectorize!($(v.full_b), $(v.B), $(v.C)) # 10.548 μs (0 allocations: 0 bytes)
    @btime MendelIHT.update_support!($(v.idx), $(v.B)) # 23.204 μs (0 allocations: 0 bytes)

    # r = 10 traits
    v, xla, Yt, Zt, k = make_mIHTvar(10)
    MendelIHT.init_iht_indices!(v)
    @btime MendelIHT.score!($v)           # 165.774 ms (7 allocations: 352 bytes) -> this step takes so long other functions doesn't really matter
    @btime MendelIHT.loglikelihood($v)    # 12.208 μs (2 allocations: 1.03 KiB)
    @btime MendelIHT.update_xb!($v)       # 219.192 μs (8 allocations: 656 bytes)
    @btime MendelIHT.update_μ!($v)        # 1.963 μs (0 allocations: 0 bytes)
    @btime MendelIHT._choose!($v)         # 68.161 μs (5 allocations: 432 bytes)
    @btime MendelIHT.solve_Σ!($v)         # 12.623 μs (2 allocations: 48 bytes)
    # @btime MendelIHT.iht_stepsize(v) # cannot be done because BenchmarkTools require multiple evaluations of the same function call, which makes Γ not pd
    @btime MendelIHT.vectorize!($(v.full_b), $(v.B), $(v.C)) # 23.712 μs (0 allocations: 0 bytes)
    @btime MendelIHT.unvectorize!($(v.full_b), $(v.B), $(v.C)) # 23.761 μs (0 allocations: 0 bytes)
    @btime MendelIHT.update_support!($(v.idx), $(v.B)) # 50.047 μs (0 allocations: 0 bytes)
end