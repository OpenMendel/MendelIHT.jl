@testset "wrappers" begin
    cd(normpath(MendelIHT.datadir()))
    result1 = iht("normal", 9) # 8 SNPs + 1 intercept
    result2 = iht("normal", "covariates.txt", 10) # include sex as covariate
    result3 = iht("phenotypes.txt", "normal", "covariates.txt", 10) # input phenotypes from separate file

    @test length(result1.c) == 1
    @test length(findall(!iszero, result1.beta)) == 8
    @test length(result2.c) == 2
    @test length(findall(!iszero, result2.beta)) == 8
    @test length(result3.c) == 2
    @test length(findall(!iszero, result3.beta)) == 8

    mses1 = cross_validate("normal", 1:20) # intercept is the only non-genetic covariate
    mses2 = cross_validate("normal", "covariates.txt", 1:20) # include sex as covariate
    mses3 = cross_validate("phenotypes.txt", "normal", "covariates.txt", 1:20) # input phenotypes from separate file
    
    @test length(mses1) == 20
    @test length(mses2) == 20
    @test length(mses3) == 20
    @test argmin(mses3) == argmin(mses2)
end
