module PLINK

using StatsBase: logistic, softplus 

import Base.size 
import Base.==
import Base.isequal
import Base.mean
import Base.copy
import NumericExtensions.sumsq

export BEDFile
export decompress_genotypes!
export decompress_genotypes
export subset_bedfile
export xb!
export xb
export xty!
export xty
export update_residuals!
export update_partial_residuals!
export sumsq!
export sumsq
export mean
export invstd
export maf

# constants used for decompression purposes
const THREE8 = convert(Int8,3)
const ZERO8  = convert(Int8,0)
const ONE8   = convert(Int8,1)
const TWO8   = convert(Int8,2)
const MNUM1  = convert(Int8,108)
#const MNUM1  = convert(Int8, -17)
const MNUM2  = convert(Int8,27)
#const MNUM2  = convert(Int8,-85)

##########################################
### CONSTRUCTORS AND UTILITY FUNCTIONS ###
##########################################

# The BEDFile type encodes the vector of compressed genotypes.
# It also encodes the number of samples and the number of predictors in the UNcompressed matrix.
# Finally, it encodes the blocksize for decompression purposes.
# These four features uniquely define any compressed genotype matrix. 
# Note that this BEDFile object, and the rest of this module for that matter, operate with the assumption
# that the compressed matrix is in column-major (SNP-major) format.
# Row-major (case-major) format is not supported.
immutable BEDFile
	x::DenseArray{Int8,1}   # compressed genotypes for genotype matrix X
	xt::DenseArray{Int8,1}  # compressed genotypes for TRANSPOSED genotype matrix X'
	n::Int                  # number of cases (people) in uncompressed genotype matrix 
	p::Int                  # number of predictors (SNPs) in uncompressed genotype matrix
	blocksize::Int          # number of bytes per compressed column of genotype matrix
	tblocksize::Int         # number of bytes per compressed column of TRANSPOSED genotype matrix

	BEDFile(x,xt,n,p,blocksize,tblocksize) = new(x,xt,n,p,blocksize,tblocksize)
end

# simple constructors for when n, p, and maybe blocksize are known and specified
# x must come from an actual BED file, so specify the path to the correct file
BEDFile(filename::ASCIIString, tfilename::ASCIIString, n::Int, p::Int, blocksize::Int, tblocksize::Int) = BEDFile(read_bedfile(filename),read_bedfile(tfilename),n,p,blocksize,tblocksize)
BEDFile(filename::ASCIIString, tfilename::ASCIIString, n::Int, p::Int) = BEDFile(read_bedfile(filename),read_bedfile(tfilename),n,p,iceil(n/4),iceil(p/4))

# a more complicated constructor that attempts to infer n, p, and blocksize based on the BED filepath
# it assumes that the BED, FAM, and BIM files are all in the same directory
function BEDFile(filename::ASCIIString, tfilename::ASCIIString; shared::Bool = true)

	# first load x, xt
	x  = read_bedfile(filename)
	xt = read_bedfile(tfilename)
	
	# if using SharedArrays, (the default), then convert x to a SharedArray
	if shared
		x  = convert(SharedArray, x)
		xt = convert(SharedArray, xt)
	end

	# find n from the corresponding FAM file 
	famfile = filename[1:(endof(filename)-3)] * "fam"
	n = count_cases(famfile)

	# find p from the corresponding BIM file
	bimfile = filename[1:(endof(filename)-3)] * "bim"
	p = count_predictors(bimfile)

	# blocksizes are easy to calculate
	blocksize  = iceil(n/4) 
	tblocksize = iceil(p/4) 

	return BEDFile(x,xt,n,p,blocksize,tblocksize)
end


###########################
###  UTILITY FUNCTIONS ###
###########################

# COUNT PREDICTORS FROM NUMBER OF LINES IN BIM FILE
function count_predictors(filename::ASCIIString)
	isequal(filename[(endof(filename)-3):endof(filename)], ".bim") || throw(ArgumentError("Filename must point to a PLINK BIM file."))
	return countlines(filename)
end

# COUNT CASES FROM NUMBER OF LINES IN FAM FILE
function count_cases(filename::ASCIIString)
	isequal(filename[(endof(filename)-3):endof(filename)], ".fam") || throw(ArgumentError("Filename must point to a PLINK FAM file."))
	return countlines(filename)
end

# OBTAIN SIZE OF UNCOMPRESSED MATRIX
size(x::BEDFile) = (x.n,x.p) 

function size(x::BEDFile, dim::Int)
	(dim == 1 || dim == 2) || throw(ArgumentError("Argument `dim` only accepts 1 or 2"))
	return ifelse(dim == 1, x.n, x.p)
end

# COPY A BEDFILE OBJECT
copy(x::BEDFile) = BEDFile(x.x, x.xt, x.n, x.p, x.blocksize, x.tblocksize)

# COMPARE DIFFERENT BEDFILE OBJECTS
==(x::BEDFile, y::BEDFile) = x.x  == y.x         && 
                             x.xt == y.xt        && 
							 x.n  == y.n         && 
							 x.p  == y.p         && 
					  x.blocksize == y.blocksize && 
					 x.tblocksize == y.tblocksize

isequal(x::BEDFile, y::BEDFile) = isequal(x.x, y.x)   && 
                                  isequal(x.xt, y.xt) && 
								  isequal(x.n, y.n)   && 
								  isequal(x.p, y.p)   && 
					isequal(x.blocksize, y.blocksize) && 
				  isequal(x.tblocksize, y.tblocksize)

# COMPUTE MINOR ALLELE FREQUENCIES
#
# This function calculates the MAF for each SNP of a compressed matrix X in a BEDFile.
#
# Arguments:
# -- x is the BEDFile object containing the compressed n x p design matrix.
#
# Optional Arguments:
# -- y is a temporary array to store a column of X.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function maf(x::BEDFile; y::DenseArray{Float64,1} = zeros(x.n))
	z = zeros(x.p)
	@inbounds for i = 1:x.p
		decompress_genotypes!(y,x,i)
		z[i] = (min( sum(y .== 1.0), sum(y .== -1.0)) + 0.5*sum(y .== 0.0)) / (x.n - sum(isnan(y)))
	end
	return z
end



# READ PLINK BINARY GENOTYPE FILES
#
# This function reads a PLINK binary file (BED) and returns an array of Int8 numbers.
# It discards the first three bytes ("magic numbers") since they are not needed here.
# 
# Arguments:
# -- filename is the path to the BED file
#
# Output:
# -- A vector of Int8 numbers. For a genotype file with n cases and p SNPs,
#    there should be AT LEAST (n*p/4) numbers. The scaling factor of 4 comes from the
#    compression of four genotypes into each byte. But PLINK stores each column in blocks
#    of bytes instead of a continuous bitstream, which sometimes entails extra unused bits 
#    at the end of each block.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function read_bedfile(filename::ASCIIString; transpose::Bool = false)

	# check that file is BED file
	contains(filename, ".bed") || throw(ArgumentError("Filename must point to a PLINK BED file."))

	# slurp file as bitstream into variable x
	# in process, reinterpret bitstream into Int8
	x = open(filename) do f
		reinterpret(Int8, readbytes(f))
	end

	# check magic number
#	isequal(x[1], MNUM1) || throw(error("Problem with first byte of magic number, is this a true BED file?"))
#	isequal(x[2], MNUM2) || throw(error("Problem with second byte of magic number, is this a true BED file?"))

	# check mode
	(transpose && isequal(x[3], ONE8)) && throw(error("For transposed matrix, third byte of BED file must indicate individual-major format."))

	# we can now safely assume that file is true BED file
	# return the genotypes
	return x[4:end]
end


# REMAP THE BITSHIFT IN GENOTYPE INTERPRETATION
#
# This function remaps the bitshifts for BEDFile subsetting purposes.
function map_bitshift(case::Int)
	k = 6 - 2 * mod(case,4)
	if k == 4
		return 0
	elseif k == 2
		return 2
	elseif k == 0
		return 4
	else
		return 6
	end
end


### NOT WORKING YET!! ###
# SUBSET A PLINK BEDFILE
#
# This function will subset a compressed matrix and contstruct a new BEDFile object from it.
#
# Arguments:
# -- x is the BEDfile object that contains the compressed n x p design matrix.
# -- rowidx is a BitArray that indexes the rows of the uncompressed x.
# -- colidx is a BitArray that indexes the columns of the uncompressed x.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function subset_bedfile(x::BEDFile, rowidx::BitArray{1}, colidx::BitArray{1})
	
	# set to "false" for debugging
	quiet = true 

	# get paramters of new BEDFile 
	yn      = sum(rowidx)
	yp      = sum(colidx)
	yblock  = iceil(yn / 4)
	ytblock = iceil(yp / 4)

	# preallocate space for new matrices
	y  = zeros(Int8, yp*yblock)
	yt = zeros(Int8, yn*ytblock)

	# if the subset is degenerate, then return a degenerate BEDFile
	(yn == 0 || yp == 0 || yblock == 0) && return BEDFile(y,yt,yn,yp,yblock,ytblock)

#	println("output compressed matrix will have ", yn, " cases.")
#	println("output compressed matrix will have ", yp, " predictors.")
#	println("output compressed matrix will have blocksize ", yblock, ".")
#	println("output compressed matrix will have ", yp*yblock, " compressed components.")

	# ensure that indices are of acceptable dimensions
	yn <= x.n || throw(ArgumentError("Argument rowidx indexes more rows than are available in x"))
	yp <= x.p || throw(ArgumentError("Argument colidx indexes more columns than are available in x"))


	# new begin to fill y
	# initialize an iterator to index y
	l = 0 

	# now loop over all columns in x 
	@inbounds for snp = 1:x.p

		# only consider the current column of X if it is indexed
		if colidx[snp]

			# initialize counters for walking down the column of X and the bytes
			case = 1
			j    = 1
			l   += 1

			# initialize a new block to fill
			new_block      = zero(Int8)
			num_genotypes  = 0

			# start looping over cases
			@inbounds while case <= x.n 

				# only consider the current row of X if it is indexed
				if rowidx[case]

					quiet || println("moving genotype for case = ", case, " and snp = ", snp)

#					# obtain the bitwise representation of the current number
#					genotype_block = x.x[(snp-1)*x.blocksize + iceil(case/4)]
#					
#					quiet || println("genotype block equals ", genotype_block)
#
#					# loop over the bit representation of the Int8 number
#					k = map_bitshift(case)
#
#					quiet || println("bitshift is k = ", k, " bits to the right")
#
#					# use bitshifting to push the two relevant bits to the right
#					# then mask using the Int8 number 3, with bit representation "00000011"
#					# performing the bitwise AND with 3 masks the first six bits and preserves the last two
#					# we can now interpret the "genotype" result as one of four possible Int8 numbers
#					genotype = (genotype_block >>> k) & THREE8
					genotype = get_index(x,case,snp)

					# new_block stores the Int8 that we will eventually put in y
					# add new genotypes to it from the right
					# to do this, apply bitwise OR to new_block with genotype bitshifted left to correct position 
					new_block = new_block | (genotype << 2*num_genotypes) 

					quiet || println("Added ", genotype, " to new_block, which now equals ", new_block)

					# keep track of how many genotypes have been compressed so far 
					num_genotypes += 1

					quiet || println("num_genotypes is now ", num_genotypes)

					# make sure to track the number of cases that we have covered so far 
					case += 1

					# as soon as we pack the byte completely, then move to the next byte
					if num_genotypes == 4 
						y[l]          = new_block	# add new block to matrix y
						new_block     = zero(Int8)	# reset new_block
						num_genotypes = 0			# reset num_genotypes
						quiet || println("filled byte at l = ", l)

						# if not at last case, then increment the index for y
						# we skip incrementing l at the last case to avoid double-incrementing
						# this only occurs at last case since l is incremented at start of new predictor 
						if sum(rowidx[1:min(case-1,x.n)]) !== yn
							quiet || println("currently at ", sum(rowidx[1:min(case-1,x.n)]), " cases of ", yn, " total.")
							l += 1		
							quiet || println("Incrementing l to l = ", l)
						end
					elseif case > x.n 
						# at this point, we haven't filled the byte
						# quit if we exceed the total number of cases
						# this will cause function to move to new genotype block
						quiet || println("Reached total number of cases, filling byte at l = ", l)
						y[l]          = new_block	# add new block to matrix y
						new_block     = zero(Int8)	# reset new_block
						num_genotypes = 0			# reset num_genotypes
						break
					end
				else
					# if current case is not indexed, then we merely add it to the counter
					# this ensures that its correspnding genotype is not compressed,
					# but it is also important for correctly indexing all of the cases in a column 
					case += 1
				end # end if/else over current case	
			end # end loop over cases
		end	# end if statement for current SNP 
	end	# end loop over SNPs

	# did we fill all of y?
	l == length(y) || warn("subsetted matrix x has $(length(y)) indices but we filled $l of them")
	l = 0

	quiet || println("filling x'")
	# now loop over columns of x'
	@inbounds for case = 1:x.n

#		quiet || println("selecting case $case? ", rowidx[case])

		# only consider the current column of X if it is indexed
		if rowidx[case]

			# initialize counters for walking down the column of X and the bytes
			snp  = 1
			j    = 1
			l   += 1

			# initialize a new block to fill
			new_block      = zero(Int8)
			num_genotypes  = 0

			# start looping over snps 
			@inbounds while snp <= x.p 

				# only consider the current row of X if it is indexed
				if colidx[snp]

					quiet || println("moving genotype for snp = ", snp , " and case = ", case)
					# obtain the bitwise representation of the current number
					genotype_block = x.xt[(case-1)*x.tblocksize + iceil(snp/4)]
					
					quiet || println("genotype block equals ", genotype_block)

					# loop over the bit representation of the Int8 number
					k = map_bitshift(case)

					quiet || println("bitshift is k = ", k, " bits to the right")

					# use bitshifting to push the two relevant bits to the right
					# then mask using the Int8 number 3, with bit representation "00000011"
					# performing the bitwise AND with 3 masks the first six bits and preserves the last two
					# we can now interpret the "genotype" result as one of four possible Int8 numbers
					genotype = (genotype_block >>> k) & THREE8

					# new_block stores the Int8 that we will eventually put in y
					# add new genotypes to it from the right
					# to do this, apply bitwise OR to new_block with genotype bitshifted left to correct position 
					new_block = new_block | (genotype << 2*num_genotypes) 

					quiet || println("Added ", genotype, " to new_block, which now equals ", new_block)

					# keep track of how many genotypes have been compressed so far 
					num_genotypes += 1

					quiet || println("num_genotypes is now ", num_genotypes)

					# make sure to track the number of cases that we have covered so far 
					snp += 1

					# as soon as we pack the byte completely, then move to the next byte
					if num_genotypes == 4 
						yt[l]         = new_block	# add new block to matrix y
						new_block     = zero(Int8)	# reset new_block
						num_genotypes = 0			# reset num_genotypes
						quiet || println("filled byte at l = ", l)

						# if not at last case, then increment the index for y
						# we skip incrementing l at the last case to avoid double-incrementing
						# this only occurs at last case since l is incremented at start of new predictor 
						if sum(colidx[1:min(snp-1,x.p)]) !== yp
							quiet || println("currently at ", sum(colidx[1:min(snp-1,x.p)]), " SNPs of ", yp, " total.")
							l += 1		
							quiet || println("Incrementing l to l = ", l)
						end
					elseif snp > x.p 
						# at this point, we haven't filled the byte
						# quit if we exceed the total number of cases
						# this will cause function to move to new genotype block
						quiet || println("Reached total number of cases, filling byte at l = ", l)
						yt[l]         = new_block	# add new block to matrix y
						new_block     = zero(Int8)	# reset new_block
						num_genotypes = 0			# reset num_genotypes
						break
					end
				else
					# if current case is not indexed, then we merely add it to the counter
					# this ensures that its correspnding genotype is not compressed,
					# but it is also important for correctly indexing all of the cases in a column 
					snp += 1
				end # end if/else over current snp 
			end # end loop over snps 
		end	# end if statement for current case 
	end	# end loop over cases 

	# did we fill all of yt?
	l == length(yt) || warn("subsetted matrix x' has $(length(yt)) indices but we filled $l of them")

	# construct new BEDFile object that contains subset of matrix
	# then return new object
	z = BEDFile(y, yt, yn, yp, yblock, ytblock)
	return z
end

##############################
### DECOMPRESSION ROUTINES ###
##############################



# INDEX A COMPRESSED BEDFILE MATRIX
#
# This subroutine succinctly extracts the dosage at the given case and SNP.
#
# Arguments:
# -- x is the BEDfile object that contains the compressed n x p design matrix X.
# -- case is the index of the current case.
# -- snp is the index of the current SNP.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function get_index(x::BEDFile, case::Int, snp::Int)
	genotype_block = x.x[(snp-1)*x.blocksize + iceil(case/4)]
	k = map_bitshift(case)
	genotype = (genotype_block >>> k) & THREE8
	return interpret_genotype(genotype)
end

# INTERPRET BIT-REPRESENTATION OF GENOTYPES
#
# This subroutine encodes the following PLINK format for genotypes:
#
# -- 00 is homozygous for allele 1
# -- 01 is heterozygous
# -- 10 is missing
# -- 11 is homozygous for allele 2
# 
# The idea is to map 00 to -1, 11 to 1, and 01 to 0.
# Since we cannot use 01, we will map it to NaN.
# Further note that the bytes are read from right to left.
# That is, if we label each of the 8 position as A to H, we would label backwards:
#
#     01101100
#     HGFEDCBA
#
# and so the first four genotypes are read as follows:
#
#     01101100
#     HGFEDCBA
#
#           AB   00  -- homozygote (first)
#         CD     11  -- other homozygote (second)
#       EF       01  -- heterozygote (third)
#     GH         10  -- missing genotype (fourth)
#
# Finally, when we reach the end of a SNP (or if in individual-mode, the end of an individual),
# then we skip to the start of a new byte (i.e. skip any remaining bits in that byte). 
# For a precise desceiption of PLINK BED files, ee the file type reference at
# http://pngu.mgh.harvard.edu/~purcell/plink/binary.shtml
#
# Arguments:
# -- a is the component of the *compressed matrix* to read.
#    It should be an 8-bit integer result of the aforementioned process. 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function interpret_genotype(a::Int8)
	if isequal(a,ZERO8)
		return 0.0
	elseif isequal(a,TWO8)
		return 1.0
	elseif isequal(a,THREE8)
		return 2.0
	else
		return NaN
	end
end


# DECOMPRESS GENOTYPES FROM INT8 BINARY FORMAT
#
# This function decompresses a column (SNP) of a SNP binary file x. 
# In Julia, these binary files are represented as arrays of Int8 numbers.
# Each SNP genotype is stored in two bits, and all people are assumed to be typed. 
# The genotypes output in y take the centered floating point values -1, 0, and 1.
#
# Arguments:
# -- y is the matrix to fill with (centered) dosages.
# -- x is the BEDfile object that contains the compressed n x p design matrix X.
# -- snp is the current SNP (predictor) to extract.
# -- means is an array of column means for X.
# -- invstds is an array of reciprocal column standard deviations for X.
#
# coded by Kevin L. Keys and Kenneth Lange (2015)
# klkeys@g.ucla.edu
function decompress_genotypes!(y::DenseArray{Float64,1}, x::BEDFile, snp::Int, means::DenseArray{Float64,1}, invstds::DenseArray{Float64,1})
	m = means[snp]
	d = invstds[snp]
	t = 0.0
	@inbounds for case = 1:x.n
		t       = get_index(x,case,snp) 
		y[case] = ifelse(isnan(t), 0.0, (t - m)*d)
	end
	return y 
end



# WRAPPER FOR DECOMPRESS_GENOTYPES!
#
# This function decompresses from x a column of genotypes corresponding to a single SNP.
# It returns the column of decompressed SNPs.
#
# Arguments:
# -- x is the BEDfile object that contains the compressed n x p design matrix.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function decompress_genotypes(x::BEDFile, snp::Int, means::DenseArray{Float64,1}, invstds::DenseArray{Float64,1}; shared::Bool = true)
	y = ifelse(shared, SharedArray(Float64, x.n, init = S -> S[localindexes(S)] = 0.0), zeros(x.n))
	decompress_genotypes!(y,x,snp,means,invstds)
	return y
end



# DECOMPRESS GENOTYPES FROM PLINK BINARY FORMAT
#
# This function decompresses PLINK BED files into a matrix.
# Use this function to test the accuracy of the linear algebra routines in this module.
# Be VERY careful with this function, since the memory demands from decompressing large portions of x
# can grow quite large.
#
# Arguments:
# -- Y is the matrix to fill with decompressed genotypes.
# -- x is the BEDfile object that contains the compressed n x p design matrix.
#
# Optional Arguments:
# -- y is temporary array for storing a column of genotypes. Defaults to zeros(n).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function decompress_genotypes!(Y::DenseArray{Float64,2}, x::BEDFile; y::DenseArray{Float64,1} = SharedArray(Float64, x.n), means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x, y=means)) 

	# extract size of Y
	const (n,p) = size(Y)

	# ensure dimension compatibility
	n == x.n || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
	p <= x.p || throw(DimensionMismatch("Y has more columns than x"))

	@inbounds for i = 1:p

		# decompress the genotypes into y
		decompress_genotypes!(y, x, i, means, invstds) 

		# copy y into Y
		@sync @inbounds @parallel for j = 1:n
			Y[j,i] = y[j]
		end
	end 
	return nothing
end


# DECOMPRESS GENOTYPES FROM PLINK BINARY FORMAT USING INDEX VECTOR
#
# This function decompresses PLINK BED files into a matrix.
# Use this function to test the accuracy of the linear algebra routines in this module.
# Be VERY careful with this function, since the memory demands from decompressing large portions of x
# can grow quite large.
#
# Arguments:
# -- Y is the matrix to fill with decompressed genotypes.
# -- x is the BEDfile object that contains the compressed n x p design matrix.
# -- indices is a BitArray that indexes the columns to use in filling Y.
#
# Optional Arguments:
# -- y is temporary array for storing a column of genotypes. Defaults to zeros(n).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function decompress_genotypes!(Y::DenseArray{Float64,2}, x::BEDFile, indices::BitArray{1}; means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x, y=means))

	# get dimensions of matrix to fill 
	const (n,p) = size(Y)

	# ensure dimension compatibility
	n == x.n          || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
	p <= x.p          || throw(DimensionMismatch("Y has more columns than x"))
	sum(indices) <= p || throw(DimensionMismatch("Vector 'indices' indexes more columns than available in Y"))

	# counter to ensure that we do not attempt to overfill Y
	current_col = 0

	@inbounds for snp = 1:x.p

		# use this column?
		if indices[snp]

			# add to counter
			current_col += 1

			# extract column mean, inv std
			m = means[snp]
			d = invstds[snp]

			@inbounds for case = 1:n
				t = get_index(x,case,snp)
				Y[case,current_col] = ifelse(isnan(t), 0.0, (t - m)*d)
			end

			# quit when Y is filled
			current_col == p && return Y
		end
	end 

	return Y 
end



###############################
### LINEAR ALGEBRA ROUTINES ###
###############################




 
# SQUARED EUCLIDEAN NORM OF A COLUMN OF A COMPRESSED MATRIX
#
# This function computes the squared L2 (Euclidean) norm of a matrix.
# The squared L2 norm of a vector y = [y1 y2 ... yn] is equal to the sum of its squared components:
#
#     || y ||_2^2 = y1^2 + y2^2 + ... + yn^2.
#
# sumsq() operates on a vector x of compressed genotypes from a PLINK BED file.
# The argument snp chooses the column of genotypes from the uncompressed matrix.
#
# Arguments:
# -- x is the BEDFile object containing the compressed genotypes.
# -- snp is the current SNP (column) to decompress.
# -- n is the number of cases in the uncompressed matrix.
# -- p is the number of predictors in the uncompressed matrix.
# -- blocksize is the number of bytes per column of the uncompressed genotype matrix.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function sumsq(x::BEDFile, snp::Int, means::DenseArray{Float64,1}, invstds::DenseArray{Float64,1}) 
	s = 0.0	# accumulation variable, will eventually equal dot(y,z)
	t = 0.0 # temp variable, output of interpret_genotype
	m = means[snp]
	d = invstds[snp]

	# loop over all n individuals
	@inbounds for case = 1:x.n
		t = get_index(x,case,snp)
		t = ifelse(isnan(t), 0.0, (t - m)*d)
		s += t*t 
	end

	return s
end

# SQUARED EUCLIDEAN NORM OF COLUMNS OF A COMPRESSED MATRIX 
#
# Compute the squared L2 norm of each column of a compressed matrix x.
#
# Arguments:
# -- y is the vector to fill with the squared norms.
# -- x is the BEDfile object that contains the compressed n x p design matrix from which to draw the columns.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function sumsq!(y::DenseArray{Float64,1}, x::BEDFile, means::DenseArray{Float64,1}, invstds::DenseArray{Float64,1})
	x.p == length(y) || throw(DimensionMismatch("y must have one row for every column of x"))
	@sync @inbounds @parallel for snp = 1:x.p
		y[snp] = sumsq(x,snp,means,invstds)
	end
	return y
end


# WRAPPER FOR SUMSQ!
#
# Return a vector with the squared L2 norms of the compressed matrix x from a BEDFile.
#
# Arguments:
# -- x is the BEDFile object to use for computing squared L2 norms
#
# Optional Arguments:
# -- shared is a Bool to indicate whether or not to return a SharedArray. Defaults to true (return SharedArray)
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function sumsq(x::BEDFile; shared::Bool = true, means::DenseArray{Float64,1} = mean(x, shared=shared), invstds::DenseArray{Float64,1} = invstd(x, y=means, shared=shared)) 
	y = ifelse(shared, SharedArray(Float64, x.p, init= S -> S[localindexes(S)] = 0.0), zeros(x.p))
	sumsq!(y,x,means,invstds)
	return y
end


# MEAN OF COLUMNS OF A COMPRESSED MATRIX
#
# Compute the arithmetic means of the columns of a compressed matrix x from a BEDFile. 
# Note that this function will ignore NaNs, unlike the normal Julia function Base.mean.
#
# Arguments:
# -- x is the BEDFile object to use for computing column means
#
# Optional Arguments
# -- shared is a Bool to indicate whether or not to return a SharedArray. Defaults to true (return SharedArray)
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function mean(x::BEDFile; shared::Bool = true)

	# initialize return vector
	y = ifelse(shared, SharedArray(Float64, x.p, init= S -> S[localindexes(S)] = 0.0), zeros(x.p))

	if shared
		@sync @inbounds @parallel for snp = 1:x.p
			y[snp] = mean_col(x, snp)
		end
	else
		@inbounds @simd for snp = 1:x.p
			y[snp] = mean_col(x, snp)
		end
	end

	return y
end

function mean_col(x::BEDFile, snp::Int)
	i = 1	# count number of people
	j = 1	# count number of bytes 
	s = 0.0	# accumulation variable, will eventually equal mean(x,col) for current col 
	t = 0.0 # temp variable, output of interpret_genotype
	u = 0.0	# count the number of people

	# loop over all n individuals
	@inbounds for case = 1:x.n
		t = get_index(x,case,snp)

		# ensure that we do not count NaNs
		if isfinite(t)
			s += t 
			u += 1.0
		end
	end
		
	# now divide s by u to get column mean and return
	return s /= u
end


# INVERSE STANDARD DEVIATION OF COLUMNS OF A COMPRESSED MATRIX
#
# Compute the inverse or reciprocal standard deviations (1 / std) of the columns of a compressed matrix x from a BEDFile. 
# Note that this function will ignore NaNs, unlike the normal Julia function Base.std.
#
# Arguments:
# -- x is the BEDFile object to use for computing column standard deviations 
#
# Optional Arguments
# -- shared is a Bool to indicate whether or not to return a SharedArray. Defaults to true (return SharedArray)
# -- y is a vector that contains the column means of x. Defaults to PLINK.mean(x).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function invstd(x::BEDFile; shared::Bool = true, y::DenseArray = mean(x))

	# initialize return vector
	z = ifelse(shared, SharedArray(Float64, x.p, init= S -> S[localindexes(S)] = 0.0), zeros(x.p))

	if shared
		@sync @inbounds @parallel for snp = 1:x.p
			z[snp] = invstd_col(x, snp, y)
		end
	else
		@inbounds @simd for snp = 1:x.p
			z[snp] = invstd_col(x, snp, y)
		end
	end

	return z
end


function invstd_col(x::BEDFile, snp::Int, means::DenseArray{Float64,1})
	s = 0.0			# accumulation variable, will eventually equal mean(x,col) for current col 
	t = 0.0 		# temp variable, output of interpret_genotype
	u = 0.0			# count the number of people
	m = means[snp]	# mean of current column

	# loop over all n individuals
	@inbounds for case = 1:x.n
		t = get_index(x,case,snp)

		# ensure that we do not count NaNs
		if isfinite(t) 
			s        += (t - m)^2 
			u        += 1.0
		end
	end

	# now compute the std = sqrt(s) / (u - 1))   
	# save inv std in y
	s    = ifelse(s <= 0.0, 0.0, sqrt((u - 1.0) / s)) 
	return s
end

# UPDATE PARTIAL RESIDUALS BASED ON PERMUTATION VECTOR FOR COMPRESSED X
# 
# This function computes the residual sum of squares || Y - XB ||_2^2 using the compressed matrix in a BEDFile object.
#
# Arguments:
# -- r is the vector to overwrite with the residuals.
# -- Y is the response vector.
# -- x is the BEDFile object with the compressed n x p design matrix.
# -- perm is a vector of integers that indexes the nonzeroes in b. 
# -- b is the parameter vector
# -- k is the desired number of components of b to use in the calculation. It should equal the number of nonzeroes in b.
#
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_partial_residuals!(r::DenseArray{Float64,1}, y::DenseArray{Float64,1}, x::BEDFile, perm::DenseArray{Int,1}, b::DenseArray{Float64,1}, k::Int; Xb::DenseArray{Float64,1} = xb!(Xb,X,b,support,k)) 
	k <= length(b)   || throw(ArgumentError("k cannot exceed the length of b!"))
	length(r) == x.n || throw(DimensionMismatch("r must have length $(x.n)!"))
	length(y) == x.n || throw(DimensionMismatch("y must have length $(x.n)!"))

	@sync @inbounds @parallel for i = 1:x.n
		r[i] = y[i] - Xb[i]
	end

	return r
end


# UPDATE PARTIAL RESIDUALS BASED ON PERMUTATION VECTOR FOR COMPRESSED X
# 
# This function computes the residual sum of squares || Y - XB ||_2^2 using the compressed matrix in a BEDFile object.
#
# Arguments:
# -- r is the vector to overwrite with the residuals.
# -- Y is the response vector.
# -- x is the BEDFile object with the compressed n x p design matrix.
# -- indices is a BitArray that indexes the nonzeroes in b. 
# -- b is the parameter vector.
# -- k is the desired number of components of b to use in the calculation. It should equal the number of nonzeroes in b.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_partial_residuals!(r::DenseArray{Float64,1}, y::DenseArray{Float64,1}, x::BEDFile, indices::BitArray{1}, b::DenseArray{Float64,1}, k::Int; means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x, y=means), Xb::DenseArray{Float64,1} = xb!(Xb,X,b,indices,k, means=means, invstds=invstds) ) 
	k <= length(b)   || throw(ArgumentError("k cannot exceed the length of b!"))
	length(r) == x.n || throw(DimensionMismatch("r must have length $(x.n)!"))
	length(y) == x.n || throw(DimensionMismatch("y must have length $(x.n)!"))

	if typeof(r) == SharedArray{Float64,1}
		@sync @inbounds @parallel for i = 1:x.n
			r[i] = y[i] - Xb[i]
		end
	else
		@inbounds @simd for i = 1:x.n
			r[i] - y[i] - Xb[i]
		end
	end

	return r
end



# COMPUTE WEIGHTED RESIDUALS (Y - 1/2 - diag(W)XB) IN LOGISTIC REGRESSION
# This subroutine, in contrast to the previous update_residuals!() function, 
# will compute WEIGHTED residuals in ONE pass.
# For optimal performance, feed it a precomputed vector of x*b.
# This variant accepts a BEDFile object for the argument x.
# 
# Arguments:
# -- r is the preallocated vector of n residuals to overwrite.
# -- x is the BEDFile that contains the compressed n x p design matrix.
# -- y is the n-vector of responses.
# -- b is the p-vector of effect sizes.
# -- perm is the p-vector that indexes b.
# -- w is the n-vector of residual weights.
#
# Optional Arguments:
# -- Xb is the n-vector x*b of predicted responses.
#    If X*b is precomputed, then this function will compute the residuals much more quickly.
# -- n is the number of samples. Defaults to length(y).
# -- p is the number of predictors. Defaults to length(b).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_residuals!(r::DenseArray{Float64,1}, x::BEDFile, y::DenseArray{Float64,1}, b::DenseArray{Float64,1}, perm::DenseArray{Int,1}, w::DenseArray{Float64,1}, k::Int; Xb::DenseArray{Float64,1} = xb(x,b,perm,k), n::Int = length(y), p::Int = length(b))
    (n,p) == size(x) || throw(DimensionMismatch("update_residuals!: nonconformable arguments!"))

    @sync @inbounds @parallel for i = 1:n 
        r[i] = y[i] - 0.5 - w[i] * Xb[i] 
    end 

    return r
end


# UPDATE WEIGHTS FOR SURROGATE FUNCTION IN LOGISTIC REGRESSION FOR ENTIRE GWAS
#
# This function calculates a vector of weights
#
#     w = 0.5*diag( tanh(0.5 * x * b) ./ x*b )
#
# for the logistic loglikelihood surrogate function. 
# Note that w is actually defined as 0.25 for each component of x*b that equals zero,
# even though the formula above would yield an undefined quantity.
#
# Arguments:
# -- w is the n-vector of weights for the predicted responses.
# -- x is the BEDFile that contains the compressed n x p design matrix.
# -- b is the p-vector of effect sizes.
#
# Optional Arguments:
# -- xb is the n-vector x*b of predicted responses. 
#    If x*b is precomputed, then this function will compute the weights much more quickly.
# -- n is the number of samples. Defaults to length(w).
# -- p is the number of predictors. Defaults to length(b).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_weights!(w::DenseArray{Float64,1}, x::BEDFile, b::DenseArray{Float64,1}, perm::DenseArray{Int,1}, k::Int; Xb::DenseArray{Float64,1} = xb(x,b,perm,k), n::Int = length(w), p::Int = length(b))
    (n,p) == size(x) || throw(DimensionMismatch("update_weights!: nonconformable arguments!"))

    @sync @inbounds @parallel for i = 1:n 
        w[i] = ifelse(xb[i] == 0.0, 0.25, 0.5*tanh(0.5*xb[i]) / xb[i]) 
    end 

    return w
end

# COMPUTE THE LOGISTIC LOGLIKELIHOOD (Y - 0.5)'XB - LOG(COSH(0.5*XB)) FOR GWAS DATA
# This subroutine computes the logistic likelihood in one pass.
# For optimal performance, supply this function with a precomputed x*b. 
# 
# Arguments:
# -- y is the n-vector of responses
# -- x is the BEDFile that contains the compressed n x p design matrix.
# -- b is the p-vector of effect sizes.
#
# Optional Arguments:
# -- n is the number of samples. Defaults to length(y).
# -- xb is the n-vector x*b of predicted responses
#    If x*b is precomputed, then this function will compute the loglikelihood more quickly. 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function compute_loglik(y::DenseArray{Float64,1}, x::BEDFile, b::DenseArray{Float64,1}, perm::DenseArray{Int,1}, k::Int; n::Int = length(y), Xb::DenseArray{Float64,1} = xb(x,b,perm,k), p::Int = length(b))
    n == length(xb) || throw(DimensionMismatch("compute_loglik: y and X*b must have same length!"))

    # each part accumulates sum s
    s = @sync @inbounds @parallel (+) for i = 1:n
        y[i]*xb[i] - softplus(xb[i])
    end

    return s
end


# COMPUTE THE DOT PRODUCT OF A COLUMN OF X AGAINST Y
#
# This function computes the dot product of a column from the compressed PLINK BED file against a vector y
# of floating point values.
#
# Arguments:
# -- x is the BEDFile object with the compressed n x p design matrix.
# -- y is the vector on which to perform the dot product.
# -- snp is the desired SNP (column) of the decompressed matrix to use for the dot product.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function dot(x::BEDFile, y::DenseArray{Float64,1}, snp::Int, means::DenseArray{Float64,1}, invstds::DenseArray{Float64,1}) 
	s = 0.0		# accumulation variable, will eventually equal dot(y,z)
	t = 0.0		# store interpreted genotype
	m = means[snp]
	d = invstds[snp]

	# loop over all individuals
	@inbounds for case = 1:x.n
		t = get_index(x,case,snp)
		# handle exceptions on t
		if isnan(t)
			t = 0.0
		else
			t  = (t - m)
			s += y[case] * t 
		end
	end

	# return the (normalized) dot product 
	return s*d 
end

# DOT PRODUCT ALONG ROWS OF X
#
# This function calculates the dot product of a vector against the rows of a matrix X of genotypes.
# It respects memory stride for column-major array ordering by employing X' in lieu of X.
# Using X' allows us to compute dot(X[i,:], b) as dot(X'[:,i], b) which respects the memory stride.
#
# Arguments:
# -- x is the BEDFile object with the compressed n x p design matrix.
# -- b is the vector to use in the dot product
# -- case is the index of the row of X (column of X') to use in the dot product
# -- indices is a BitArray that indexes the nonzero elements of b
#
# Optional Arguments:
# -- means is a vector of column means for X.
# -- invstds is a vector of reciprocal column standard deviations for X. 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function dott(x::BEDFile, b::DenseArray{Float64,1}, case::Int, indices::BitArray{1}, means::DenseArray{Float64,1}, invstds::DenseArray{Float64,1}) 
	snp = 1
	j = 1
	k = 0
	s = 0.0		# accumulation variable, will eventually equal dot(y,z)
	t = 0.0		# store interpreted genotype
	@inbounds while snp <= x.p 

		# if current index of b is FALSE, then skip it since it does not contribute to Xb
		if indices[snp] 
			genotype_block = x.xt[(case-1)*x.tblocksize + j]
			genotype       = (genotype_block >>> k) & THREE8
			t              = interpret_genotype(genotype)

			# handle exceptions on t
			if isnan(t)
				t = 0.0
			else
				t  = (t - means[snp]) * invstds[snp] 
				s += b[snp] * t 
			end

			snp += 1 
			snp > x.p && return s 
			k += 2
			if k > 6
				k  = 0
				j += 1
			end
		else
			snp += 1
			snp > x.p && return s 
			k += 2
			if k > 6
				k  = 0
				j += 1
			end
		end
	end

	# return the dot product 
	return s 
end

# PERFORM X * BETA
#
# This function computes the operation X*b for the compressed n x p design matrix X in a BEDFile object in a manner that respects memory stride for column-major arrays.
# It also assumes a sparse b, for which we have an index vector to select the nonzeroes.
#
# Arguments:
# -- Xb is the n-dimensional output vector.
# -- x is the BEDFile object with the compressed n x p design matrix.
# -- b is the p-dimensional vector against which we multiply X.
# -- indices is a BitArray that indexes the nonzeroes in b.
# -- k is the number of nonzeroes to use in computing X*b.
#
# Optional Arguments:
# -- means is a vector of column means for X.
# -- invstds is a vector of reciprocal column standard deviations for X. 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function xb!(Xb::DenseArray{Float64,1}, x::BEDFile, b::DenseArray{Float64,1}, indices::BitArray{1}, k::Int; means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x, y=means))
    # error checking
    0 <= k <= x.p     || throw(ArgumentError("Number of active predictors must be nonnegative and less than p"))
	k == sum(indices) || throw(ArgumentError("k != sum(indices)"))

	# loop over the desired number of predictors 
	@sync @inbounds @parallel for case = 1:x.n
		Xb[case] = dott(x, b, case, indices, means, invstds)	
	end

	return Xb
end 


# WRAPPER FOR XB!
#
# This function allocates, computes, and returns the matrix-vector product X*b using a matrix X of compressed genotypes.
#  
# Arguments:
# -- x is the BEDFile object with the compressed n x p design matrix.
# -- b is the p-dimensional vector against which we multiply X.
# -- k is the number of nonzeroes.
# -- indices is a BitArray that indexes the nonzeroes in b.
#
# Optional Arguments:
# -- shared is a Bool to indicate whether or not to return a SharedArray. Defaults to true (return SharedArray).
# -- means is a vector of column means for X.
# -- invstds is a vector of reciprocal column standard deviations for X. 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function xb(x::BEDFile, b::DenseArray{Float64,1}, indices::BitArray{1}, k::Int; shared::Bool = true, means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x, y=means)) 
	Xb = ifelse(shared, SharedArray(Float64, x.n, init = S -> S[localindexes(S)] = 0.0), zeros(x.n))
	xb!(Xb,x,b,indices,k, means=means, invstds=invstds)
	return Xb
end


# PERFORM X'Y, OR A TRANSPOSED MATRIX-VECTOR PRODUCT
#
# This function performs a general matrix-vector multiply with a transposed matrix.
# Compare the results to BLAS.gemv('T', 1.0, X, y, 0.0, Xty).
#
# Arguments:
# -- Xty is the output vector to overwrite
# -- x is the BEDfile object that contains the compressed n x p design matrix.
# -- y is the vector used in the matrix-vector multiply
#
# Optional Arguments:
# -- means is a vector of column means for X.
# -- invstds is a vector of reciprocal column standard deviations for X. 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function xty!(Xty::DenseArray{Float64,1}, x::BEDFile, y::DenseArray{Float64,1}; means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x, y=means)) 

	# error checking
	x.p == length(Xty) || throw(ArgumentError("Attempting to fill argument Xty of length $(length(Xty)) with $(x.p) elements!"))
	x.n == length(y)   || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.n) of them!"))

	# loop over the desired number of predictors 
	@sync @inbounds @parallel for snp = 1:x.p
		Xty[snp] = dot(x,y,snp,means,invstds)
	end

	return Xty
end 


# WRAPPER FOR XTY!
#
# This function initializes an output vector and computes x'*y with a BEDFile object.
# Compare output to BLAS.gemv('T', 1.0, x, y).
#
# Arguments:
# -- x is the BEDFile object whose compressed matrix we will decompress in order to compute x'*y.
# -- y is the vector against which we will multiply.
#
# Optional Arguments:
# -- shared is a Bool to indicate whether or not to return a SharedArray. Defaults to true (return SharedArray).
# -- means is a vector of column means for X.
# -- invstds is a vector of reciprocal column standard deviations for X. 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function xty(x::BEDFile, y::DenseArray{Float64,1}; shared::Bool = true, means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x, y=means)) 
	Xty = ifelse(shared, SharedArray(Float64, x.p, init = S -> S[localindexes(S)] = 0.0), zeros(x.p))
	xty!(Xty,x,y, means=means, invstds=invstds) 
	return Xty
end

end	# end module PLINK
