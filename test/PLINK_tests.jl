# will need to test with parallel capabilities enabled 
addprocs(3)

@everywhere using NumericExtensions: sumsq
@everywhere using PLINK
@everywhere using Base.Test

#############################
### test the constructors ###
#############################

# we rely on testing data bundled with standard PLINK 1.07 download
# it has been precompressed into binary format
# first initialize what the data *should* look like
const num_case = 6
const num_snps = 3
const block    = iceil(num_case/4)
const tblock   = iceil(num_snps/4)
const x        = convert(SharedArray{Int8,1}, [-36; 15; -25; 15; 107; 1])
const xt       = convert(SharedArray{Int8,1}, [60; 39; 41; 31; 31; 15])

# now use all four constructors
const x1 = BEDFile(x, xt, num_case, num_snps, block, tblock)
const x2 = BEDFile("../data/test_binary.bed", "../data/test_binary_t.bed")
const x3 = BEDFile("../data/test_binary.bed", "../data/test_binary_t.bed", num_case, num_snps)
const x4 = BEDFile("../data/test_binary.bed", "../data/test_binary_t.bed", num_case, num_snps, block, tblock)

# did inner constructor work?
@test isequal(x1.x, x)             
@test isequal(x1.p, num_snps)      
@test isequal(x1.n, num_case)      
@test isequal(x1.blocksize, block) 

# did outer constructors yield same result?
@test isequal(x1,x2) 
@test isequal(x1,x3) 
@test isequal(x1,x4) 


##################################
### test the utility functions ###
##################################

# test size functions
@test isequal(size(x1), (num_case, num_snps)) 
@test isequal(size(x1,1), num_case)           
@test isequal(size(x1,2), num_snps)           

###################################
### test decompression routines ###
###################################

# first initialize the correct results
Y = convert(SharedArray{Float64,2},
	[0.0 2.0 2.0
     2.0 NaN 1.0
 	 NaN 1.0 1.0
	 2.0 2.0 NaN  
	 2.0 2.0 NaN  
	 2.0 2.0 0.0])

const means = convert(SharedArray{Float64,1}, [1.6, 1.8, 1.0])
const invstds = convert(SharedArray{Float64,1}, [1.1180339887498947,2.2360679774997894,1.224744871391589])
for j = 1:num_snps
	for i = 1:num_case
		Y[i,j] = (Y[i,j] - means[j]) * invstds[j]
	end
end
Y[isnan(Y)] = 0.0
const Yt = Y'

# this tests the full decompression routine
Y1 = SharedArray(Float64, num_case, num_snps)
decompress_genotypes!(Y1,x1, means=means, invstds=invstds)
@test isequal(Y,Y1) 

Y2 = SharedArray(Float64,0,0)
colidx = trues(num_snps)
rowidx = trues(num_case)
# this tests the indexed decompression
for j = 0:num_snps
	colidx = trues(num_snps)
	colidx[1:j] = false
	Y2          = SharedArray(Float64, x1.n, sum(colidx))
	decompress_genotypes!(Y2,x1,colidx,means=means,invstds=invstds)
	@test isequal(Y2, Y[:, colidx])
end

#x5 = copy(x1)
## now test several permutations of subsetting and decompression
#for i = 0:num_case, j = 0:num_snps
#	println("testing subsetting for $i rows, $j columns")
#	rowidx      = trues(num_case)
#	colidx      = trues(num_snps)
#	rowidx[1:i] = false
#	colidx[1:j] = false
#	x5          = subset_bedfile(x1, rowidx, colidx)
#	Y2          = SharedArray(Float64, x5.n, x5.p)
#	
#	decompress_genotypes!(Y2,x5,colidx, means=means, invstds=invstds)
#	@test isequal(Y2, Y[rowidx, colidx])
#end

####################################
### test linear algebra routines ###
####################################

# create temporary arrays for linear algebra routines
y       = SharedArray(Float64, x1.n, init = S -> S[localindexes(S)] = 1.0)
b       = SharedArray(Float64, x1.p, init = S -> S[localindexes(S)] = 1.0 ./ localindexes(S))
r       = SharedArray(Float64, x1.n, init = S -> S[localindexes(S)] = 0.0)
indices = trues(num_snps)

const XtY_out   = Yt*y 
const Xb_out    = Y*b
const sumsq_out = convert(SharedArray, vec(NumericExtensions.sumsq(sdata(Y),1)))
const res_out   = y - Xb_out

XtY = xty(x1,y) 
Xb  = xb(x1, b, indices, x1.p) 
ssq = PLINK.sumsq(x1)
PLINK.update_partial_residuals!(r, y, x1, indices, b, x1.p, Xb=Xb)

my_eps = 4.0*eps()
@test_approx_eq_eps(XtY_out, XtY, my_eps)
@test_approx_eq_eps(Xb_out, Xb, my_eps)
@test_approx_eq_eps(sumsq_out, ssq, my_eps)
@test_approx_eq_eps(res_out, r, my_eps)

