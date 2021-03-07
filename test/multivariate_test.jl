using Revise
using MendelIHT
using SnpArrays
using Random
using GLM
using DelimitedFiles
using Test
using Distributions
using LinearAlgebra
using CSV
using DataFrames
using BenchmarkTools

function make_mIHTvar()
    n = 1000  # number of samples
    p = 10000 # number of SNPs
    k = 10    # number of causal SNPs
    r = 2     # number of traits
    
    # set random seed for reproducibility
    Random.seed!(2021)
    
    # simulate `.bed` file with no missing data
    x = simulate_random_snparray(undef, n, p)
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
    
    # intercept is the only nongenetic covariate
    z = ones(n, 1)
    intercepts = [10.0 1.0] # each trait have different intercept
    
    # simulate response y, true model b, and the correct non-0 positions of b
    Y, true_Σ, true_b, correct_position = simulate_random_response(xla, k, r, Zu=z*intercepts, overlap=2);
    
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
    v, xla, Yt, Zt, k = make_mIHTvar()
    @btime MendelIHT.init_iht_indices!(v) # 73.642 ms (10 allocations: 156.89 KiB) -> need to call `check_covariate_supp!` so might be okay
    @btime MendelIHT.score!($v)           # 72.751 ms (7 allocations: 352 bytes)
    @btime MendelIHT.loglikelihood($v)    # 2.210 μs (2 allocations: 48 bytes)
    @btime MendelIHT.update_xb!($v)       # 85.013 μs (8 allocations: 656 bytes)
    @btime MendelIHT.update_μ!($v)        # 297.873 ns (0 allocations: 0 bytes)
    @btime MendelIHT._choose!($v)         # 13.436 μs (3 allocations: 208 bytes)
    @btime MendelIHT.solve_Σ!(v)          # 2.365 μs (2 allocations: 48 bytes)
    # @btime MendelIHT.iht_stepsize(v) # cannot be done because BenchmarkTools require multiple evaluations of the same function call, which makes Γ not pd

    @btime MendelIHT.vectorize!(v.full_b, v.B, v.C) # 4.003 μs (0 allocations: 0 bytes)
    @btime MendelIHT.unvectorize!(v.B, v.C, v.full_b) # 3.934 μs (0 allocations: 0 bytes)
    @btime MendelIHT.update_support!(v.idx, v.B) # 17.059 μs (0 allocations: 0 bytes)

end