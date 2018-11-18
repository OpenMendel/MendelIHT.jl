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


using SnpArrays, BenchmarkTools
x = SnpArray(rand(0:2, 5000, 30000))
maf, minor_allele, = summarize(x)
people, snps = size(x)
mean_vec = deepcopy(maf) # Gordon wants maf below
   
function test()
    for i in 1:snps
        minor_allele[i] ? mean_vec[i] = 2.0 - 2.0mean_vec[i] : mean_vec[i] = 2.0mean_vec[i]
    end
end

function test2()
    @inbounds @simd for i in 1:snps
        minor_allele[i] ? mean_vec[i] = 2.0 - 2.0mean_vec[i] : mean_vec[i] = 2.0mean_vec[i]
    end
end

function test3()
    @inbounds for i in 1:snps
        minor_allele[i] ? mean_vec[i] = 2.0 - 2.0mean_vec[i] : mean_vec[i] = 2.0mean_vec[i]
    end
end

function test4()
    @simd for i in 1:snps
        minor_allele[i] ? mean_vec[i] = 2.0 - 2.0mean_vec[i] : mean_vec[i] = 2.0mean_vec[i]
    end
end


#BELOW ARE LOGISTIC SIMUATIONS
#load packages
using IHT
using SnpArrays
using DataFrames
using Distributions

#set random seed
srand(1111) 

#specify dimension and noise of data
n = 1000                        # number of cases
p = 10000                       # number of predictors
k = 10                          # number of true predictors per group
s = 0.1                         # noise vector, from very little noise to a lot of noise

#construct snpmatrix, covariate files, and true model b
x           = SnpArray(rand(0:2, n, p))    # a random snpmatrix
z           = ones(n, 1)                   # non-genetic covariates, just the intercept
true_b      = zeros(p)                     # model vector
true_b[1:k] = randn(k)                     # Initialize k non-zero entries in the true model
shuffle!(true_b)                           # Shuffle the entries
correct_position = find(true_b)            # keep track of what the true entries are
noise = rand(Normal(0, s), n)              # noise vectors from N(0, s) where s ∈ S = {0.01, 0.1, 1, 10}s

#compute mean and std used to standardize data to mean 0 variance 1
mean_vec, minor_allele, = summarize(x)
for i in 1:p
    minor_allele[i] ? mean_vec[i] = 2.0 - 2.0mean_vec[i] : mean_vec[i] = 2.0mean_vec[i]
end
std_vec = std_reciprocal(x, mean_vec)

#simulate phenotypes under different noises by: y = Xb + noise
y_temp = zeros(n)
SnpArrays.A_mul_B!(y_temp, x, true_b, mean_vec, std_vec)
y_temp .+= noise #add some noise

# Apply inverse logit link to map y to {0, 1} 
y = 1 ./ (1 .+ exp.(-y_temp)) #inverse logit link
y .= round.(y)                     #map y to 0, 1

#compute logistic IHT result
scale = sum(y) / n
estimated_models = zeros(k)
v = IHTVariables(x, z, y, 1, k)
result = L0_logistic_reg(v, x, z, y, 1, k, glm = "logistic")

#check result
estimated_models .= result.beta[correct_position]






#BELOW ARE NORMAL SIMUATIONS
#load packages
using IHT
using SnpArrays
using DataFrames
using Distributions

#set random seed
srand(1111) 

#specify dimension and noise of data
n = 1000                        # number of cases
p = 10000                       # number of predictors
k = 10                          # number of true predictors per group
S = 0.1                         # noise vector, from very little noise to a lot of noise

#construct snpmatrix, covariate files, and true model b
x           = SnpArray(rand(0:2, n, p))    # a random snpmatrix
z           = ones(n, 1)                   # non-genetic covariates, just the intercept
true_b      = zeros(p)                     # model vector
true_b[1:k] = randn(k)                     # Initialize k non-zero entries in the true model
shuffle!(true_b)                           # Shuffle the entries
correct_position = find(true_b)            # keep track of what the true entries are
noise = rand(Normal(0, S), n)              # noise vectors from N(0, s) where s ∈ S = {0.01, 0.1, 1, 10}s

#compute mean and std used to standardize data to mean 0 variance 1
mean_vec, minor_allele, = summarize(x)
for i in 1:p
    minor_allele[i] ? mean_vec[i] = 2.0 - 2.0mean_vec[i] : mean_vec[i] = 2.0mean_vec[i]
end
std_vec = std_reciprocal(x, mean_vec)

#simulate phenotypes under different noises by: y = Xb + noise
y = zeros(n)
SnpArrays.A_mul_B!(y, x, true_b, mean_vec, std_vec)
y .+= noise

#compute IHT result for less noisy data
v = IHTVariables(x, z, y, 1, k)
result = L0_reg(v, x, z, y, 1, k)
estimated_models = result.beta[correct_position]



function test(
	p :: AbstractVector{Float64}, 
	x :: SnpLike{2},
	z :: AbstractMatrix{Float64},
	b :: AbstractVector{Float64},
	c :: AbstractVector{Float64}
)
	@inbounds @simd for i in eachindex(p)
        xβ = dot(view(x.A1, i, :), b) + dot(view(x.A2, i, :), b) + dot(view(z, i, :), c)
        p[i] = e^xβ / (1 + e^xβ)
    end
end

p = rand(100)
x = SnpArray(rand(0:2, 100, 100))
z = rand(100, 100)
b = rand(100)
c = rand(100)
test(p, x, z, b, c)

function test()
	p = 1000000
	k = 1000
	x = rand(0:2, p)
	β = zeros(p)
	β[1:100] = randn(100)
	shuffle!(β)
	e^dot(x, β) / (1 + e^dot(x, β))
end





