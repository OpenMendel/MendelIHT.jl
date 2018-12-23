using IHT
using PLINK
using StatsBase

srand(2018)

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

@testset "Do MendelIHT and IHT output the same result?" begin
	# Note: For both test, there are only 1 group and covariates only contain 
	# a column of 1 for intercept

    # test on small data set: CURRENTLY THESE ANSWERS IS OFF BY A FEW DECIMAL POINTS BECAUSE THE OLD CODE DOESN'T UPDATE v.idx
 #    mendel_result = MendelIHT("test_control.txt")
 #    (xbed, ybed) = PLINK.read_plink_data("test")
	# iht_result = L0_reg(xbed, ybed, 2)
	# non_zero_mendel = find(mendel_result.beta)
	# non_zero_iht = find(iht_result.beta)
	# mendel_beta_val = mendel_result.beta[non_zero_mendel]
	# mendel_intercept = mendel_result.c
	# iht_beta_val = iht_result.beta[non_zero_iht]

	# #test overall loss and number of iterations agree
	# @test mendel_result.loss ≈ iht_result.loss
	# @test mendel_result.iter == iht_result.iter

	# #same number of variables were selected
	# @test length(mendel_beta_val) + 1 == length(iht_beta_val)

	# #test non-zero entries of beta are equal in value
	# @test mendel_beta_val[1] ≈ iht_beta_val[1]
	# @test mendel_intercept[1] ≈ iht_beta_val[2]

	# #test non-zero entries of beta are at the correct places
	# @test non_zero_mendel[1] == non_zero_iht[1]

    #test on bigger dataset
    mendel_result = MendelIHT(Pkg.dir() * "/IHT/test/gwas 1 Control.txt")
    (xbed, ybed) = PLINK.read_plink_data(Pkg.dir() * "/IHT/test/gwas 1 data_kevin", delim=',', header=false)
	iht_result = L0_reg(xbed, ybed, 10)
	non_zero_mendel = find(mendel_result.beta)
	non_zero_iht = find(iht_result.beta)
	mendel_beta_val = mendel_result.beta[non_zero_mendel]
	mendel_intercept = mendel_result.c
	iht_beta_val = iht_result.beta[non_zero_iht]

	@test round(mendel_result.loss, 3) == round(iht_result.loss, 3)
	@test mendel_result.iter == iht_result.iter

	#same number of variables were selected
	@test length(mendel_beta_val) + 1 == length(iht_beta_val)

	#test non-zero entries of beta are equal in absolute value
	@test all(isapprox.(mendel_beta_val, iht_beta_val[1:end-1], atol=0.0000001))
	@test isapprox(mendel_result.c[1], iht_beta_val[end], atol=0.0000001)

	#test non-zero entries of beta are at the correct places
	@test all(non_zero_mendel .== non_zero_iht[1:end-1]) 
end
