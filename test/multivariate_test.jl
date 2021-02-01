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

@testset "multivariate project_k!" begin
    x = rand(5, 5)
    k = 2
    project_k!(x, k)
    for xi in eachrow(x)
        @test count(!iszero, xi) ≤ k
    end
end