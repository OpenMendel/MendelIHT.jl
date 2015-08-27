module PLINK

using StatsBase: logistic, softplus 

import Base.size 
import Base.==
import Base.isequal
import Base.mean
import Base.copy
import Base.getindex
import Base.length
import Base.ndims
import NumericExtensions.sumsq
import Base.display

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
export getindex
export addx2!

# constants used for decompression purposes
const ZERO8  = convert(Int8,0)
const ONE8   = convert(Int8,1)
const TWO8   = convert(Int8,2)
const THREE8 = convert(Int8,3)
const MNUM1  = convert(Int8,108)
#const MNUM1  = convert(Int8, -17)
const MNUM2  = convert(Int8,27)
#const MNUM2  = convert(Int8,-85)

# SWITCH TO INTERPRET BIT-REPRESENTATION OF GENOTYPES
#
# This lookup table encodes the following PLINK format for genotypes:
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
# For a precise desceiption of PLINK BED files, see the file type reference at
# http://pngu.mgh.harvard.edu/~purcell/plink/binary.shtml
const geno     = [0.0, NaN, 1.0, 2.0]

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
type BEDFile
	x::DenseArray{Int8,1}   	# compressed genotypes for genotype matrix X
	xt::DenseArray{Int8,1}  	# compressed genotypes for TRANSPOSED genotype matrix X'
	n::Int                  	# number of cases (people) in uncompressed genotype matrix 
	p::Int                  	# number of predictors (SNPs) in uncompressed genotype matrix
	blocksize::Int          	# number of bytes per compressed column of genotype matrix
	tblocksize::Int         	# number of bytes per compressed column of TRANSPOSED genotype matrix
	x2::DenseArray{Float64,2}	# nongenetic covariantes, if any exist
	p2::Int						# number of nongenetic covariates
	x2t::DenseArray{Float64,2}	# transpose of nongenetic covariantes, used in matrix algebra 

	BEDFile(x,xt,n,p,blocksize,tblocksize,x2,p2,x2t) = new(x,xt,n,p,blocksize,tblocksize,x2,p2,x2t)
end

# simple constructors for when n, p, and maybe blocksize are known and specified
# x must come from an actual BED file, so specify the path to the correct file
function BEDFile(filename::ASCIIString, tfilename::ASCIIString, n::Int, p::Int, blocksize::Int, tblocksize::Int, x2filename::ASCIIString)
	x  = BEDFile(read_bedfile(filename),read_bedfile(tfilename),n,p,blocksize,tblocksize,SharedArray(Float64,n,0),0)
	x2 = convert(SharedArray{Float64,2}, readdlm(x2filename))
	p2 = size(x2,2)
	x.x2 = x2
	x.x2t = x2'
	x.p2 = p2
	return x
end

function BEDFile(filename::ASCIIString, tfilename::ASCIIString, n::Int, p::Int, x2filename::ASCIIString)
	x = BEDFile(read_bedfile(filename),read_bedfile(tfilename),n,p,((n-1)>>>2)+1,((p-1)>>>2)+1,SharedArray(Float64,n,0),0)
	x2 = convert(SharedArray{Float64,2}, readdlm(x2filename))
	p2 = size(x2,2)
	x.x2 = x2
	x.x2t = x2'
	x.p2 = p2
	return x
end

# a more complicated constructor that attempts to infer n, p, and blocksize based on the BED filepath
# it assumes that the BED, FAM, and BIM files are all in the same directory
function BEDFile(filename::ASCIIString, tfilename::ASCIIString; shared::Bool = true)

	# find n from the corresponding FAM file 
	famfile = filename[1:(endof(filename)-3)] * "fam"
	n = count_cases(famfile)

	# find p from the corresponding BIM file
	bimfile = filename[1:(endof(filename)-3)] * "bim"
	p = count_predictors(bimfile)

	# blocksizes are easy to calculate
	blocksize  = ((n-1) >>> 2) + 1
	tblocksize = ((p-1) >>> 2) + 1

	# now load x, xt
	x   = read_bedfile(filename)
	xt  = read_bedfile(tfilename)
	x2  = zeros(n,0) 
	x2t = x2' 

	# if using SharedArrays, (the default), then convert x to a SharedArray
	if shared
		x   = convert(SharedArray, x)
		xt  = convert(SharedArray, xt)
		x2  = convert(SharedArray, x2)
		x2t = convert(SharedArray, x2t)
	end

	return BEDFile(x,xt,n,p,blocksize,tblocksize,x2,0,x2t)
end


# an extra constructor based on previous one 
# this one admits a third file path for the nongenetic covariates 
# it uncreatively creates a BEDFile using previous constructor with two file paths,
# and then fills the nongenetic covariates with the third file path 
function BEDFile(filename::ASCIIString, tfilename::ASCIIString, x2filename::ASCIIString; shared::Bool = true, header::Bool = true)

	x    = BEDFile(filename, tfilename, shared=shared)
	x2   = readdlm(x2filename, header=header)
	if shared
		x2 = convert(SharedArray, x2)
	end
	x.n   == size(x2,1) || throw(DimensionMismatch("Nongenetic covariates have more rows than genotype matrix"))
	x.x2  = x2
	x.p2  = size(x2,2)
	x.x2t = x2'
	return x 
end


###########################
###  UTILITY FUNCTIONS ###
###########################

# COUNT PREDICTORS FROM NUMBER OF LINES IN BIM FILE
function count_predictors(f::ASCIIString)
	isequal(f[(endof(f)-3):endof(f)], ".bim") || throw(ArgumentError("Filename must point to a PLINK BIM file."))
	return countlines(f)
end

# COUNT CASES FROM NUMBER OF LINES IN FAM FILE
function count_cases(f::ASCIIString)
	isequal(f[(endof(f)-3):endof(f)], ".fam") || throw(ArgumentError("Filename must point to a PLINK FAM file."))
	return countlines(f)
end

# OBTAIN SIZE OF UNCOMPRESSED MATRIX
size(x::BEDFile) = (x.n, x.p + x.p2) 

function size(x::BEDFile, dim::Int)
	(dim == 1 || dim == 2) || throw(ArgumentError("Argument `dim` only accepts 1 or 2"))
	return ifelse(dim == 1, x.n, x.p + x.p2)
end

function size(x::BEDFile; submatrix::ASCIIString = "genotype")
	(isequal(submatrix, "genotype") || isequal(submatrix, "nongenetic")) || throw(ArgumentError("Argument `submatrix` only accepts `genotype` or `nongenetic`"))
	return ifelse(isequal(submatrix,"genotype"), (x.n, x.p), (x.n, x.p2))
end


# OBTAIN LENGTH OF UNCOMPRESSED MATRIX
length(x::BEDFile) = x.n*(x.p + x.p2)

# OBTAIN NUMBER OF DIMENSIONS OF UNCOMPRESSED MATRIX
ndims(x::BEDFile) = 2

# COPY A BEDFILE OBJECT
copy(x::BEDFile) = BEDFile(x.x, x.xt, x.n, x.p, x.blocksize, x.tblocksize, x.x2, x.p2, x.x2t)

# COMPARE DIFFERENT BEDFILE OBJECTS
==(x::BEDFile, y::BEDFile) = x.x   == y.x  &&
                             x.xt  == y.xt &&
                             x.n   == y.n  &&
                             x.p   == y.p  &&
                      x.blocksize  == y.blocksize &&
                     x.tblocksize  == y.tblocksize &&
					         x.x2  == y.x2 &&
					         x.p2  == y.p2 &&
							 x.x2t == y.x2t

isequal(x::BEDFile, y::BEDFile) = isequal(x.x, y.x)                   && 
                                  isequal(x.xt, y.xt)                 && 
                                  isequal(x.n, y.n)                   && 
                                  isequal(x.p, y.p)                   && 
                                  isequal(x.blocksize, y.blocksize)   && 
                                  isequal(x.tblocksize, y.tblocksize) &&
								  isequal(x.x2, y.x2)                 &&
								  isequal(x.p2, y.p2)                 &&
								  isequal(x.x2t, y.y2t)


function addx2!(x::BEDFile, x2::DenseArray{Float64,2}; shared::Bool = true)
	(n,p2) = size(x2)
	n == x.n || throw(DimensionMismatch("x2 has $n rows but should have $(x.n) of them"))
	x.p2 = p2
	x.x2 = ifelse(shared, SharedArray(Float64, n, p2), zeros(n,p2)) 
#	x.x2t = x.x2' 
	for j = 1:p2
		for i = 1:x.n
			@inbounds x.x2[i,j] = x2[i,j]
		end
	end
	x.x2t = x.x2'
	return nothing
end

function display(x::BEDFile)
	println("A BEDFile object with the following features:")
	println("\tnumber of cases        = $(x.n)")
	println("\tgenetic covariates     = $(x.p)")
	println("\tnongenetic covariates  = $(x.p2)")
	println("\tcovariate type         = $(typeof(x.x2))")
end

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


# SUBSET A COMPRESSED GENOTYPE MATRIX
#
# This subroutine will subset a stream of Int8 numbers representing a compressed genotype matrix.
# Argument X is vacuous; it simply ensures no ambiguity with current Array implementations
function subset_genotype_matrix(X::BEDFile, x::DenseArray{Int8,1}, rowidx::BitArray{1}, colidx::BitArray{1}, n::Int, p::Int, blocksize::Int; yn::Int = sum(rowidx), yp::Int = sum(colidx), yblock::Int = ((yn-1) >>> 2) + 1, ytblock::Int = ((yp-1) >>> 2) + 1)

	quiet = true 

	yn <= n || throw(ArgumentError("rowidx indexes more rows than available in uncompressed matrix."))
	yp <= p || throw(ArgumentError("colidx indexes more columns than available in uncompressed matrix."))

	y = zeros(Int8, yp*yblock)
	(yn == 0 || yblock == 0) && return y

	l = 0
	# now loop over all columns in x 
	@inbounds for col = 1:p

		# only consider the current column of X if it is indexed
		if colidx[col]

			# count bytes in y
			l += 1

			# initialize a new block to fill
			new_block      = zero(Int8)
			num_genotypes  = 0
			current_row    = 0

			# start looping over cases
			@inbounds for row = 1:n

				# only consider the current row of X if it is indexed
				if rowidx[row]

					quiet || println("moving genotype for row = ", row, " and col = ", col)

					genotype = getindex(X,x,row,col,blocksize, interpret=false)

					# new_block stores the Int8 that we will eventually put in y
					# add new genotypes to it from the right
					# to do this, apply bitwise OR to new_block with genotype bitshifted left to correct position 
					new_block = new_block | (genotype << 2*num_genotypes) 

					quiet || println("Added ", genotype, " to new_block, which now equals ", new_block)

					# keep track of how many genotypes have been compressed so far 
					num_genotypes += 1

					quiet || println("num_genotypes is now ", num_genotypes)

					# make sure to track the number of cases that we have covered so far 
					current_row += 1
					quiet || println("current_row = ", current_row)

					# as soon as we pack the byte completely, then move to the next byte
					if num_genotypes == 4 && current_row < yn 
						y[l]          = new_block	# add new block to matrix y
						new_block     = zero(Int8)	# reset new_block
						num_genotypes = 0			# reset num_genotypes
						quiet || println("filled byte at l = ", l)

						# if not at last row, then increment the index for y
						# we skip incrementing l at the last row to avoid double-incrementing l at start of new predictor
##						if sum(rowidx[1:min(row-1,n)]) !== yn
						if sum(rowidx[1:min(current_row-1,n)]) !== yn
##							quiet || println("currently at ", sum(rowidx[1:min(row-1,n)]), " rows of ", yn, " total.")
							quiet || println("currently at ", sum(rowidx[1:min(current_row-1,n)]), " rows of ", yn, " total.")
							l += 1		
							quiet || println("Incrementing l to l = ", l)
						end
					elseif current_row >= yn 
						# at this point, we haven't filled the byte
						# quit if we exceed the total number of cases
						# this will cause function to move to new genotype block
						quiet || println("Reached total number of rows, filling byte at l = ", l)
						y[l]          = new_block	# add new block to matrix y
						new_block     = zero(Int8)	# reset new_block
						num_genotypes = 0			# reset num_genotypes
						break
					end
				else
					# if current row is not indexed, then we merely add it to the counter
					# this not only ensures that its correspnding genotype is not compressed,
					# but it also ensures correct indexing for all of the rows in a column 
#					row += 1
				end # end if/else over current row 
			end # end loop over rows 
		end	# end if statement for current col 
	end	# end loop over cols 

	# did we fill all of y?
	l == length(y) || warn("subsetted matrix x has $(length(y)) indices but we filled $l of them")
	return y

end


function subset_genotype_matrix(X::BEDFile, x::DenseArray{Int8,1}, rowidx::UnitRange{Int64}, colidx::BitArray{1}, n::Int, p::Int, blocksize::Int; yn::Int = sum(rowidx), yp::Int = sum(colidx), yblock::Int = ((yn-1) >>> 2) + 1, ytblock::Int = ((yp-1) >>> 2) + 1)

	quiet = true 

	yn <= n || throw(ArgumentError("rowidx indexes more rows than available in uncompressed matrix."))
	yp <= p || throw(ArgumentError("colidx indexes more columns than available in uncompressed matrix."))

	y = zeros(Int8, yp*yblock)
	(yn == 0 || yblock == 0) && return y

	l = 0
	# now loop over all columns in x 
	@inbounds for col = 1:p

		# only consider the current column of X if it is indexed
		if colidx[col]

			# count bytes in y
			l += 1

			# initialize a new block to fill
			new_block      = zero(Int8)
			num_genotypes  = 0
			current_row    = 0

			# start looping over cases
			@inbounds for row in rowidx 

				quiet || println("moving genotype for row = ", row, " and col = ", col)

				genotype = getindex(X,x,row,col,blocksize, interpret=false)

				# new_block stores the Int8 that we will eventually put in y
				# add new genotypes to it from the right
				# to do this, apply bitwise OR to new_block with genotype bitshifted left to correct position 
				new_block = new_block | (genotype << 2*num_genotypes) 

				quiet || println("Added ", genotype, " to new_block, which now equals ", new_block)

				# keep track of how many genotypes have been compressed so far 
				num_genotypes += 1

				quiet || println("num_genotypes is now ", num_genotypes)

				# make sure to track the number of cases that we have covered so far 
				current_row += 1
				quiet || println("current_row = ", current_row)

				# as soon as we pack the byte completely, then move to the next byte
				if num_genotypes == 4 && current_row < yn 
					y[l]          = new_block	# add new block to matrix y
					new_block     = zero(Int8)	# reset new_block
					num_genotypes = 0			# reset num_genotypes
					quiet || println("filled byte at l = ", l)

					# if not at last row, then increment the index for y
					# we skip incrementing l at the last row to avoid double-incrementing l at start of new predictor
##						if sum(rowidx[1:min(row-1,n)]) !== yn
					if sum(rowidx[1:min(current_row-1,n)]) !== yn
##							quiet || println("currently at ", sum(rowidx[1:min(row-1,n)]), " rows of ", yn, " total.")
						quiet || println("currently at ", sum(rowidx[1:min(current_row-1,n)]), " rows of ", yn, " total.")
						l += 1		
						quiet || println("Incrementing l to l = ", l)
					end
				elseif current_row >= yn 
					# at this point, we haven't filled the byte
					# quit if we exceed the total number of cases
					# this will cause function to move to new genotype block
					quiet || println("Reached total number of rows, filling byte at l = ", l)
					y[l]          = new_block	# add new block to matrix y
					new_block     = zero(Int8)	# reset new_block
					num_genotypes = 0			# reset num_genotypes
					break
				end
			end # end loop over rows 
		end	# end if statement for current col 
	end	# end loop over cols 

	# did we fill all of y?
	l == length(y) || warn("subsetted matrix x has $(length(y)) indices but we filled $l of them")
	return y

end



function subset_genotype_matrix(X::BEDFile, x::DenseArray{Int8,1}, rowidx::BitArray{1}, colidx::UnitRange{Int64}, n::Int, p::Int, blocksize::Int; yn::Int = sum(rowidx), yp::Int = sum(colidx), yblock::Int = ((yn-1) >>> 2) + 1, ytblock::Int = ((yp-1) >>> 2) + 1)

	quiet = true 

	yn <= n || throw(ArgumentError("rowidx indexes more rows than available in uncompressed matrix."))
	yp <= p || throw(ArgumentError("colidx indexes more columns than available in uncompressed matrix."))

	y = zeros(Int8, yp*yblock)
	(yn == 0 || yblock == 0) && return y

	l = 0
	# now loop over all columns in x 
	@inbounds for col in colidx 

		# count bytes in y
		l += 1

		# initialize a new block to fill
		new_block      = zero(Int8)
		num_genotypes  = 0
		current_row    = 0

		# start looping over cases
		@inbounds for row = 1:n

			# only consider the current row of X if it is indexed
			if rowidx[row]

				quiet || println("moving genotype for row = ", row, " and col = ", col)

				genotype = getindex(X,x,row,col,blocksize, interpret=false)

				# new_block stores the Int8 that we will eventually put in y
				# add new genotypes to it from the right
				# to do this, apply bitwise OR to new_block with genotype bitshifted left to correct position 
				new_block = new_block | (genotype << 2*num_genotypes) 

				quiet || println("Added ", genotype, " to new_block, which now equals ", new_block)

				# keep track of how many genotypes have been compressed so far 
				num_genotypes += 1

				quiet || println("num_genotypes is now ", num_genotypes)

				# make sure to track the number of cases that we have covered so far 
				current_row += 1
				quiet || println("current_row = ", current_row)

				# as soon as we pack the byte completely, then move to the next byte
				if num_genotypes == 4 && current_row < yn 
					y[l]          = new_block	# add new block to matrix y
					new_block     = zero(Int8)	# reset new_block
					num_genotypes = 0			# reset num_genotypes
					quiet || println("filled byte at l = ", l)

					# if not at last row, then increment the index for y
					# we skip incrementing l at the last row to avoid double-incrementing l at start of new predictor
##						if sum(rowidx[1:min(row-1,n)]) !== yn
					if sum(rowidx[1:min(current_row-1,n)]) !== yn
##							quiet || println("currently at ", sum(rowidx[1:min(row-1,n)]), " rows of ", yn, " total.")
						quiet || println("currently at ", sum(rowidx[1:min(current_row-1,n)]), " rows of ", yn, " total.")
						l += 1		
						quiet || println("Incrementing l to l = ", l)
					end
				elseif current_row >= yn 
					# at this point, we haven't filled the byte
					# quit if we exceed the total number of cases
					# this will cause function to move to new genotype block
					quiet || println("Reached total number of rows, filling byte at l = ", l)
					y[l]          = new_block	# add new block to matrix y
					new_block     = zero(Int8)	# reset new_block
					num_genotypes = 0			# reset num_genotypes
					break
				end
			else
				# if current row is not indexed, then we merely add it to the counter
				# this not only ensures that its correspnding genotype is not compressed,
				# but it also ensures correct indexing for all of the rows in a column 
#					row += 1
			end # end if/else over current row 
		end # end loop over rows 
	end	# end loop over cols 

	# did we fill all of y?
	l == length(y) || warn("subsetted matrix x has $(length(y)) indices but we filled $l of them")
	return y

end


function subset_genotype_matrix(X::BEDFile, x::DenseArray{Int8,1}, rowidx::UnitRange{Int64}, colidx::UnitRange{Int64}, n::Int, p::Int, blocksize::Int; yn::Int = sum(rowidx), yp::Int = sum(colidx), yblock::Int = ((yn-1) >>> 2) + 1, ytblock::Int = ((yp-1) >>> 2) + 1) 

	quiet = true 

	yn <= n || throw(ArgumentError("rowidx indexes more rows than available in uncompressed matrix."))
	yp <= p || throw(ArgumentError("colidx indexes more columns than available in uncompressed matrix."))

	y = zeros(Int8, yp*yblock)
	(yn == 0 || yblock == 0) && return y

	l = 0
	# now loop over all columns in x 
	@inbounds for col in colidx 

		# count bytes in y
		l += 1

		# initialize a new block to fill
		new_block      = zero(Int8)
		num_genotypes  = 0
		current_row    = 0

		# start looping over cases
		@inbounds for row in rowidx 

			# only consider the current row of X if it is indexed
			if rowidx[row]

				quiet || println("moving genotype for row = ", row, " and col = ", col)

				genotype = getindex(X,x,row,col,blocksize, interpret=false)

				# new_block stores the Int8 that we will eventually put in y
				# add new genotypes to it from the right
				# to do this, apply bitwise OR to new_block with genotype bitshifted left to correct position 
				new_block = new_block | (genotype << 2*num_genotypes) 

				quiet || println("Added ", genotype, " to new_block, which now equals ", new_block)

				# keep track of how many genotypes have been compressed so far 
				num_genotypes += 1

				quiet || println("num_genotypes is now ", num_genotypes)

				# make sure to track the number of cases that we have covered so far 
				current_row += 1
				quiet || println("current_row = ", current_row)

				# as soon as we pack the byte completely, then move to the next byte
				if num_genotypes == 4 && current_row < yn 
					y[l]          = new_block	# add new block to matrix y
					new_block     = zero(Int8)	# reset new_block
					num_genotypes = 0			# reset num_genotypes
					quiet || println("filled byte at l = ", l)

					# if not at last row, then increment the index for y
					# we skip incrementing l at the last row to avoid double-incrementing l at start of new predictor
##						if sum(rowidx[1:min(row-1,n)]) !== yn
					if sum(rowidx[1:min(current_row-1,n)]) !== yn
##							quiet || println("currently at ", sum(rowidx[1:min(row-1,n)]), " rows of ", yn, " total.")
						quiet || println("currently at ", sum(rowidx[1:min(current_row-1,n)]), " rows of ", yn, " total.")
						l += 1		
						quiet || println("Incrementing l to l = ", l)
					end
				elseif current_row >= yn 
					# at this point, we haven't filled the byte
					# quit if we exceed the total number of cases
					# this will cause function to move to new genotype block
					quiet || println("Reached total number of rows, filling byte at l = ", l)
					y[l]          = new_block	# add new block to matrix y
					new_block     = zero(Int8)	# reset new_block
					num_genotypes = 0			# reset num_genotypes
					break
				end
			else
				# if current row is not indexed, then we merely add it to the counter
				# this not only ensures that its correspnding genotype is not compressed,
				# but it also ensures correct indexing for all of the rows in a column 
#					row += 1
			end # end if/else over current row 
		end # end loop over rows 
	end	# end loop over cols 

	# did we fill all of y?
	l == length(y) || warn("subsetted matrix x has $(length(y)) indices but we filled $l of them")
	return y

end

##############################
### DECOMPRESSION ROUTINES ###
##############################



## INDEX A COMPRESSED BEDFILE MATRIX
##
## This subroutine succinctly extracts the dosage at the given case and SNP.
##
## Arguments:
## -- x is the BEDfile object that contains the compressed n x p design matrix X.
## -- case is the index of the current case.
## -- snp is the index of the current SNP.
##
## coded by Kevin L. Keys (2015)
## klkeys@g.ucla.edu
#function getindex(x::BEDFile, case::Int, snp::Int; interpret::Bool = true)
#	genotype_block = x.x[(snp-1)*x.blocksize + iceil(case/4)]
#	k = map_bitshift(case)
#	genotype = (genotype_block >>> k) & THREE8
#	interpret && return interpret_genotype(genotype)
#	return genotype
#end


# GET THE VALUE OF A GENOTYPE IN A COMPRESSED MATRIX
# argument X is almost vacuous because it ensures no conflict with current Array implementations
# it becomes useful for accessing nongenetic covariates
function getindex(X::BEDFile, x::DenseArray{Int8,1}, row::Int, col::Int, blocksize::Int; interpret::Bool = true)
	if col <= X.p
#		genotype_block = x[(col-1)*blocksize + iceil(row/4)]
		genotype_block = x[(col-1)*blocksize + ((row - 1) >>> 2) + 1]
		k = 2*((row-1) & 3) 
		genotype = (genotype_block >>> k) & THREE8
        interpret && return geno[genotype + ONE8] 

		return genotype
	else
		return X.x2[row,(col-X.p)]
	end
end

function getindex(x::BEDFile, rowidx::BitArray{1}, colidx::BitArray{1})

	yn = sum(rowidx)
	yp = sum(colidx)
	yblock  = ((yn-1) >>> 2) + 1
	ytblock = ((yp-1) >>> 2) + 1

	y  = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock) 
	yt = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock) 
	y2 = x.x2[rowidx,colidx]
	y2t = y2'
	p2 = size(y2,2)

	return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2,y2')
end

function getindex(x::BEDFile, rowidx::UnitRange{Int64}, colidx::BitArray{1})

	yn = length(rowidx)
	yp = sum(colidx)
	yblock  = ((yn-1) >>> 2) + 1
	ytblock = ((yp-1) >>> 2) + 1

	y  = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock) 
	yt = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock) 
	y2 = x.x2[rowidx,colidx]
	p2 = size(y2,2)

	return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2)
end

function getindex(x::BEDFile, rowidx::BitArray{1}, colidx::UnitRange{Int64})

	yn = sum(rowidx)
	yp = length(colidx)
	yblock  = ((yn-1) >>> 2) + 1
	ytblock = ((yp-1) >>> 2) + 1

	y   = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock) 
	yt  = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock) 
	y2  = x.x2[rowidx,colidx]
	p2  = size(y2,2)
	y2t = y2'

	return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2,y2t)
end


function getindex(x::BEDFile, rowidx::UnitRange{Int64}, colidx::UnitRange{Int64})

	yn = length(rowidx)
	yp = length(colidx)
	yblock  = ((yn-1) >>> 2) + 1
	ytblock = ((yp-1) >>> 2) + 1

	y   = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock) 
	yt  = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock) 
	y2  = x.x2[rowidx,colidx]
	p2  = size(y2,2)
	y2t = y2'

	return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2,y2t)
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
	if snp <= x.p
		@inbounds for case = 1:x.n
			t       = getindex(x,x.x,case,snp,x.blocksize) 
			y[case] = ifelse(isnan(t), 0.0, (t - m)*d)
		end
	else
		@inbounds for case = 1:x.n
			y[case] = x.x2[case,(snp-x.p)]
		end
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
#function decompress_genotypes!(Y::DenseArray{Float64,2}, x::BEDFile; means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x, y=means)) 

	# extract size of Y
	const (n,p) = size(Y)

	quiet = true 

	# ensure dimension compatibility
	n == x.n || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
	p <= x.p || throw(DimensionMismatch("Y has more columns than x"))

#	@inbounds for i = 1:p
#
#		quiet || println("decompressing col $i")
#		# decompress the genotypes into y
#		decompress_genotypes!(y, x, i, means, invstds) 
#
#		# copy y into Y
#		@sync @inbounds @parallel for j = 1:n
#			Y[j,i] = y[j]
#			quiet || println("Y[$j,$i] = ", y[j])
#		end
#	end 

	@inbounds for j = 1:p
		decompress_genotypes!(y,x,j,means,invstds)	
		@inbounds for i = 1:n
			Y[i,j] = y[i]
		end
	end 

	return Y 
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
	sum(indices) <= p || throw(DimensionMismatch("Vector 'indices' indexes more columns than are available in Y"))

	# counter to ensure that we do not attempt to overfill Y
	current_col = 0


	quiet = true 
	@inbounds for snp = 1:(x.p + x.p2)

		# use this column?
		if indices[snp]

			# add to counter
			current_col += 1
			quiet || println("filling current column $current_col with snp $snp")

			if snp <= x.p

				# extract column mean, inv std
				m = means[snp]
				d = invstds[snp]

				@inbounds for case = 1:n
					t = getindex(x,x.x,case,snp,x.blocksize)
					Y[case,current_col] = ifelse(isnan(t), 0.0, (t - m)*d)
					quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
				end
			else
				@inbounds for case = 1:n
					Y[case,current_col] = x.x2[case,(snp-x.p)]
				end
			end

			# quit when Y is filled
			current_col == p && return Y
		end
	end 
	return Y 
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
# -- indices is an Integer vector that indexes the columns to use in filling Y.
#
# Optional Arguments:
# -- y is temporary array for storing a column of genotypes. Defaults to zeros(n).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function decompress_genotypes!(Y::DenseArray{Float64,2}, x::BEDFile, indices::DenseArray{Int,1}; means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x, y=means))

	# get dimensions of matrix to fill 
	const (n,p) = size(Y)

	# ensure dimension compatibility
	n == x.n          || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
	p <= x.p          || throw(DimensionMismatch("Y has more columns than x"))
	length(indices) <= p || throw(DimensionMismatch("Vector 'indices' indexes more columns than are available in Y"))

	# counter to ensure that we do not attempt to overfill Y
	current_col = 0


	quiet = true 
	@inbounds for snp in indices 

		# add to counter
		current_col += 1
		quiet || println("filling current column $current_col with snp $snp")

		if snp <= x.p
			# extract column mean, inv std
			m = means[snp]
			d = invstds[snp]

			@inbounds for case = 1:n
				t = getindex(x,x.x,case,snp,x.blocksize)
				Y[case,current_col] = ifelse(isnan(t), 0.0, (t - m)*d)
				quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
			end
		else
			@inbounds for case = 1:n
				Y[case,current_col] = x.x2[case,(snp-x.p)]
			end
		end

		# quit when Y is filled
		current_col == p && return Y
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
function sumsq_snp(x::BEDFile, snp::Int, means::DenseArray{Float64,1}, invstds::DenseArray{Float64,1}) 
	s = 0.0	# accumulation variable, will eventually equal dot(y,z)
	t = 0.0 # temp variable, output of interpret_genotype
	m = means[snp]
	d = invstds[snp]

	# loop over all n individuals
	@inbounds for case = 1:x.n
		t = getindex(x,x.x,case,snp,x.blocksize)
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
function sumsq!(y::SharedArray{Float64,1}, x::BEDFile, means::SharedArray{Float64,1}, invstds::SharedArray{Float64,1})
	(x.p + x.p2) == length(y) || throw(DimensionMismatch("y must have one row for every column of x"))
	@sync @inbounds @parallel for snp = 1:x.p
		y[snp] = sumsq_snp(x,snp,means,invstds)
	end
	@inbounds for covariate = 1:x.p2
		y[covariate] = 0.0
		@inbounds for row = 1:x.n
			y[covariate] += x.x2[row,covariate]
		end
	end

	return y
end


function sumsq!(y::Array{Float64,1}, x::BEDFile, means::Array{Float64,1}, invstds::Array{Float64,1})
	(x.p + x.p2) == length(y) || throw(DimensionMismatch("y must have one row for every column of x"))
	@inbounds for snp = 1:x.p
		y[snp] = sumsq_snp(x,snp,means,invstds)
	end
	@inbounds for covariate = 1:x.p2
		y[covariate] = 0.0
		@inbounds for row = 1:x.n
			y[covariate] += x.x2[row,covariate]
		end
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
	y = ifelse(shared, SharedArray(Float64, x.p + x.p2, init= S -> S[localindexes(S)] = 0.0), zeros(x.p + x.p2))
	sumsq!(y,x,means,invstds)
#	for i = (x.p+1):(x.p+x.p2)
#		@inbounds y[i] *= y[i]
#	end
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
	y = ifelse(shared, SharedArray(Float64, x.p + x.p2, init= S -> S[localindexes(S)] = 0.0), zeros(x.p + x.p2))

	if shared
		@sync @inbounds @parallel for snp = 1:x.p
			y[snp] = mean_col(x, snp)
		end
	else
		@inbounds @simd for snp = 1:x.p
			y[snp] = mean_col(x, snp)
		end
	end
	for i = 1:x.p2 
		for j = 1:x.n
			@inbounds y[x.p + i] += x.x2[j,i]
		end
		y[i] /= x.n
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
		t = getindex(x,x.x,case,snp,x.blocksize)

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
	z = ifelse(shared, SharedArray(Float64, x.p + x.p2, init= S -> S[localindexes(S)] = 0.0), zeros(x.p + x.p2))

	if shared
		@sync @inbounds @parallel for snp = 1:x.p
			z[snp] = invstd_col(x, snp, y)
		end
	else
		@inbounds @simd for snp = 1:x.p
			z[snp] = invstd_col(x, snp, y)
		end
	end
	for i = 1:x.p2 
		for j = 1:x.n
			@inbounds z[x.p + i] += (x.x2[j,i] - y[x.p + i])^2
		end
		z[i] /= x.n
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
		t = getindex(x,x.x,case,snp,x.blocksize)

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
#function update_partial_residuals!(r::DenseArray{Float64,1}, y::DenseArray{Float64,1}, x::BEDFile, perm::DenseArray{Int,1}, b::DenseArray{Float64,1}, k::Int; Xb::DenseArray{Float64,1} = xb!(Xb,X,b,support,k)) 
function update_partial_residuals!(r::SharedArray{Float64,1}, y::SharedArray{Float64,1}, x::BEDFile, perm::SharedArray{Int,1}, b::DenseArray{Float64,1}, k::Int; means=mean(x), invstds=invstd(x, y=means), Xb::SharedArray{Float64,1} = xb(X,b,support,k, means=means, invstds=invstds)) 
	k <= length(b)   || throw(ArgumentError("k cannot exceed the length of b!"))
	length(r) == x.n || throw(DimensionMismatch("r must have length $(x.n)!"))
	length(y) == x.n || throw(DimensionMismatch("y must have length $(x.n)!"))

	@sync @inbounds @parallel for i = 1:x.n
		r[i] = y[i] - Xb[i]
	end

	return r
end

function update_partial_residuals!(r::Array{Float64,1}, y::Array{Float64,1}, x::BEDFile, perm::Array{Int,1}, b::Array{Float64,1}, k::Int; means=mean(x), invstds=invstd(x, y=means), Xb::Array{Float64,1} = xb(X,b,support,k, means=means, invstds=invstds)) 
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
#function update_partial_residuals!(r::DenseArray{Float64,1}, y::DenseArray{Float64,1}, x::BEDFile, indices::BitArray{1}, b::DenseArray{Float64,1}, k::Int; means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x, y=means), Xb::DenseArray{Float64,1} = xb!(Xb,X,b,indices,k, means=means, invstds=invstds) ) 
function update_partial_residuals!(r::SharedArray{Float64,1}, y::SharedArray{Float64,1}, x::BEDFile, indices::BitArray{1}, b::SharedArray{Float64,1}, k::Int; means::SharedArray{Float64,1} = mean(x), invstds::SharedArray{Float64,1} = invstd(x, y=means), Xb::SharedArray{Float64,1} = xb(X,b,indices,k, means=means, invstds=invstds) ) 
	k <= length(b)    || throw(ArgumentError("k cannot exceed the length of b!"))
#	k <= sum(indices) || throw(ArgumentError("k cannot exceed the number of true values in indices!"))
	length(r) == x.n  || throw(DimensionMismatch("r must have length $(x.n)!"))
	length(y) == x.n  || throw(DimensionMismatch("y must have length $(x.n)!"))

	@sync @inbounds @parallel for i = 1:x.n
		r[i] = y[i] - Xb[i]
	end

	return r
end


function update_partial_residuals!(r::Array{Float64,1}, y::Array{Float64,1}, x::BEDFile, indices::BitArray{1}, b::Array{Float64,1}, k::Int; means::Array{Float64,1} = mean(x), invstds::Array{Float64,1} = invstd(x, y=means), Xb::Array{Float64,1} = xb(X,b,indices,k, means=means, invstds=invstds) ) 
	k <= length(b)    || throw(ArgumentError("k cannot exceed the length of b!"))
#	k <= sum(indices) || throw(ArgumentError("k cannot exceed the number of true values in indices!"))
	length(r) == x.n  || throw(DimensionMismatch("r must have length $(x.n)!"))
	length(y) == x.n  || throw(DimensionMismatch("y must have length $(x.n)!"))

	@inbounds @simd for i = 1:x.n
		r[i] = y[i] - Xb[i]
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

	if snp <= x.p
#		t = 0.0				# store interpreted genotype
		m = means[snp]		# mean of SNP predictor
		d = invstds[snp]	# 1/std of SNP predictor

		# loop over all individuals
		@inbounds for case = 1:x.n
			t = getindex(x,x.x,case,snp,x.blocksize)
			# handle exceptions on t
#			if isnan(t)
#				t = 0.0
#			else
#				t  = (t - m)
#				s += y[case] * t 
				s += y[case] * (t - m) 
#			end
		end

		# return the (normalized) dot product 
		return s*d 
	else
		@inbounds for case = 1:x.n
			s += x.x2[case,snp] * y[case]
		end
		return s
	end
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
#			t              = interpret_genotype(genotype)
			t              = geno[genotype + ONE8]

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
	@inbounds for snp = (x.p+1):(x.p+x.p2)
		s += b[snp] * x.x2t[snp,case]
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
    0 <= k <= size(x,2) || throw(ArgumentError("Number of active predictors must be nonnegative and less than p"))
#	k <= sum(indices)   || throw(ArgumentError("k != sum(indices)"))
	k >= sum(indices)   || throw(ArgumentError("Must have k >= sum(indices) or X*b will not compute correctly"))

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
#function xb(x::BEDFile, b::DenseArray{Float64,1}, indices::BitArray{1}, k::Int; shared::Bool = true, means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x, y=means)) 
#	Xb = ifelse(shared, SharedArray(Float64, x.n, init = S -> S[localindexes(S)] = 0.0), zeros(x.n))
#	xb!(Xb,x,b,indices,k, means=means, invstds=invstds)
#	return Xb
#end


function xb(x::BEDFile, b::Array{Float64,1}, indices::BitArray{1}, k::Int; means::Array{Float64,1} = mean(x, shared=false), invstds::Array{Float64,1} = invstd(x, y=means, shared=false)) 
	Xb = zeros(x.n)
	xb!(Xb,x,b,indices,k, means=means, invstds=invstds)
	return Xb
end

function xb(x::BEDFile, b::SharedArray{Float64,1}, indices::BitArray{1}, k::Int; means::SharedArray{Float64,1} = mean(x, shared=true), invstds::SharedArray{Float64,1} = invstd(x, y=means, shared=true)) 
	Xb = SharedArray(Float64, x.n, init = S -> S[localindexes(S)] = 0.0)
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
function xty!(Xty::SharedArray{Float64,1}, x::BEDFile, y::SharedArray{Float64,1}; means::SharedArray{Float64,1} = mean(x, shared=true), invstds::SharedArray{Float64,1} = invstd(x, y=means, shared=true)) 

	# error checking
	x.p <= length(Xty) || throw(ArgumentError("Attempting to fill argument Xty of length $(length(Xty)) with $(x.p) elements!"))
	x.n == length(y)   || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.n) of them!"))

	# loop over the desired number of predictors 
	@sync @inbounds @parallel for snp = 1:x.p
		Xty[snp] = dot(x,y,snp,means,invstds)
	end

	return Xty
end 

function xty!(Xty::Array{Float64,1}, x::BEDFile, y::Array{Float64,1}; means::Array{Float64,1} = mean(x, shared=false), invstds::Array{Float64,1} = invstd(x, y=means, shared=false)) 

	# error checking
	x.p <= length(Xty) || throw(ArgumentError("Attempting to fill argument Xty of length $(length(Xty)) with $(x.p) elements!"))
	x.n == length(y)   || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.n) of them!"))

	# loop over the desired number of predictors 
	@inbounds for snp = 1:x.p
		println("snp = ", snp)
		@time Xty[snp] = dot(x,y,snp,means,invstds)
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
function xty(x::BEDFile, y::SharedArray{Float64,1}; means::SharedArray{Float64,1} = mean(x, shared=true), invstds::SharedArray{Float64,1} = invstd(x, y=means, shared=true)) 
	Xty = SharedArray(Float64, x.p + x.p2, init = S -> S[localindexes(S)] = 0.0)
	xty!(Xty,x,y, means=means, invstds=invstds) 
	return Xty
end

function xty(x::BEDFile, y::Array{Float64,1}; means::Array{Float64,1} = mean(x, shared=false), invstds::Array{Float64,1} = invstd(x, y=means, shared=false)) 
	Xty = zeros(x.p + x.p2)
	xty!(Xty,x,y, means=means, invstds=invstds) 
	return Xty
end

end	# end module PLINK
