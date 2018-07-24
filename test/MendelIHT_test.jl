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
    
    #test on small data set
    mendel_result = MendelIHT("test_control.txt")
    (xbed, ybed) = PLINK.read_plink_data("test")
	iht_result = L0_reg(xbed, ybed, 2)
	non_zero_mendel = find(mendel_result.beta)
	non_zero_iht = find(iht_result.beta) 
	mendel_beta_val = mendel_result.beta[non_zero_mendel]
	iht_beta_val = iht_result.beta[non_zero_iht]

	@test mendel_result.loss ≈ iht_result.loss
	@test mendel_result.iter == iht_result.iter

	#test non-zero entries of beta are equal in value
	@test all(mendel_beta_val .≈ mendel_beta_val)

	#test non-zero entries of beta are at the correct places
	@test all(non_zero_mendel .== non_zero_iht) 


    #test on bigger dataset
    mendel_result = MendelIHT(Pkg.dir() * "/IHT/test/gwas 1 Control.txt")
    (xbed, ybed) = PLINK.read_plink_data(Pkg.dir() * "/IHT/test/gwas 1 data_kevin", delim=',', header=false)
	iht_result = L0_reg(xbed, ybed, 10)
	non_zero_mendel = find(mendel_result.beta)
	non_zero_iht = find(iht_result.beta) 
	mendel_beta_val = mendel_result.beta[non_zero_mendel]
	iht_beta_val = iht_result.beta[non_zero_iht]

	@test round(mendel_result.loss, 3) == round(iht_result.loss, 3)
	@test mendel_result.iter == iht_result.iter

	#test non-zero entries of beta are equal in absolute value
	@test all(mendel_beta_val .≈ mendel_beta_val)

	#test non-zero entries of beta are at the correct places
	@test all(non_zero_mendel .== non_zero_iht) 
end
