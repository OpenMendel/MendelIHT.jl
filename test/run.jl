using IHT, BenchmarkTools
@benchmark x = MendelIHT("gwas 1 Control.txt") seconds = 100

using MendelIterHardThreshold
x = IterHardThreshold("gwas 1 Control.txt")

x = MendelIHT("test_control.txt")


using IHT, BenchmarkTools
@benchmark MendelIHT("gwas 1 Control userpath.txt")
# x, y = MendelIHT("gwas 1 Control userpath.txt")


using IHT, PLINK
# MendelIHT("gwas 1 Control.txt")
(xbed, ybed) = read_plink_data("gwas 1 data_kevin", delim = ',')
output = L0_reg(xbed, ybed, 10)

MendelIHT("gwas 1 Control userpath.txt")
nmodels = 20
pathidx = collect(1:nmodels)        # the model sizes to test: 1, 2, ..., nmodels
# ihtbetas = iht_path(xbed, ybed, pathidx) # note that ihtpath is a sparse matrix...
# full(ihtbetas[bidx,:])



using IHT
MendelIHT("gwas 1 Control.txt")


using IHT
MendelIHT("gwas 1 Control cv.txt")



using IHT, BenchmarkTools
@benchmark MendelIHT("gwas 1 Control.txt")


using IHT, BenchmarkTools
@benchmark MendelIHT("gwas 1 Control cv.txt") seconds = 100




using MendelIterHardThreshold
IterHardThreshold("test_control.txt")

using IHT
MendelIHT("test_control.txt")






# inside PLINK/data
using IHT, PLINK
(xbed, ybed) = read_plink_data(Pkg.dir() * "/IHT/test/gwas 1 data_kevin", delim=',', header=false)
iht_result = L0_reg(xbed, ybed, 10)

using IHT, PLINK
cv_iht("gwas 1 data_kevin.bed", "gwas 1 data_kevin.cov", "gwas 1 data_kevin.ypath")

function test()
	c = zeros(100)
	x = rand(1000000)
	b = view(x, 1:100000)
	A = rand(100, 100000)
	A_mul_B!(c, A, b)
end

function test()
	x = zeros(1000, 1000)
	y = rand(1000, 1000)
	x[1:900, 1:900] .= view(y, 1:900, 1:900)
end

using SnpArrays, BenchmarkTools
x = SnpArray(1000, 10000)
function test()
	# maf, _, _, _ = summarize(x)
	# maf, a, b, c = summarize(x)
	maf, = summarize(x)
end
@benchmark test()







using MendelBase, MendelKinship, SnpArrays
keyword = set_keyword_defaults!(Dict{AbstractString, Any}())
keyword["repetitions"] = 1
keyword["xlinked_analysis"] = false
keyword["kinship_output_file"] = "kinship_file_output.txt"
keyword["compare_kinships"] = false
keyword["maf_threshold"] = 0.01
keyword["grm_method"] = "MoM" # MoM is less sensitive to rare snps
keyword["deviant_pairs"] = 0
keyword["kinship_plot"] = ""
keyword["z_score_plot"] = ""
process_keywords!(keyword, "control_just_theoretical_29a.txt", "")
(pedigree, person, nuclear_family, locus, snpdata, locus_frame, 
    phenotype_frame, pedigree_frame, snp_definition_frame) =
    read_external_data_files(keyword)







