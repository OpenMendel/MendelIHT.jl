using IHT, PLINK, RegressionTools

srand(2018)

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

@testset "Is MendelIHT.jl and original IHT.jl outputting the same thing" begin
    
    #test on small data set
    mendel_result = MendelIHT("test_control.txt")
    (xbed, ybed) = read_plink_data("test")
	kevin_result = L0_reg(xbed, ybed, 2)
	non_zero_mendel = find(mendel_result.beta)
	non_zero_kevin = find(kevin_result.beta) 
	mendel_beta_val = mendel_result.beta[non_zero_mendel]
	kevin_beta_val = kevin_result.beta[non_zero_kevin]

	@test round(mendel_result.loss, 3) == round(kevin_result.loss, 3)
	@test mendel_result.iter == kevin_result.iter
	#test non-zero entries of beta are equal in absolute value
	@test all(abs(sort(mendel_beta_val, by=abs)) .== abs(sort(mendel_beta_val, by=abs)))
	#test non-zero entries of beta are at the correct places
	@test all(non_zero_mendel[2:end] .- 1 .== non_zero_kevin[1:end-1]) 


    #test on bigger dataset
    mendel_result = MendelIHT("gwas 1 Control.txt")
    (xbed, ybed) = read_plink_data("gwas 1 data_kevin", delim=',', header=false)
	kevin_result = L0_reg(xbed, ybed, 10)
	non_zero_mendel = find(mendel_result.beta)
	non_zero_kevin = find(kevin_result.beta) 
	mendel_beta_val = mendel_result.beta[non_zero_mendel]
	kevin_beta_val = kevin_result.beta[non_zero_kevin]

	@test round(mendel_result.loss, 3) == round(kevin_result.loss, 3)
	@test mendel_result.iter == kevin_result.iter
	#test non-zero entries of beta are equal in absolute value
	@test all(abs(sort(mendel_beta_val, by=abs)) .== abs(sort(mendel_beta_val, by=abs)))
	#test non-zero entries of beta are at the correct places
	@test all(non_zero_mendel[2:end] .- 1 .== non_zero_kevin[1:end-1]) 
end
