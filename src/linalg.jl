###############################
### LINEAR ALGEBRA ROUTINES ###
###############################


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
function maf{T <: Union(Float32, Float64)}(x::BEDFile; y::DenseArray{T,1} = zeros(T,x.n))
	z = zeros(T,x.p)
	@inbounds for i = 1:x.p
		decompress_genotypes!(y,x,i)
		z[i] = (min( sum(y .== 1.0), sum(y .== -1.0)) + 0.5*sum(y .== 0.0)) / (x.n - sum(isnan(y)))
	end
	return z
end


 
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
function sumsq_snp{T <: Union(Float32, Float64)}(x::BEDFile, snp::Integer, means::DenseArray{T,1}, invstds::DenseArray{T,1}) 
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
function sumsq!{T <: Union(Float32, Float64)}(y::SharedArray{T,1}, x::BEDFile, means::SharedArray{T,1}, invstds::SharedArray{T,1})
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


function sumsq!{T <: Union(Float32, Float64)}(y::Array{T,1}, x::BEDFile, means::Array{T,1}, invstds::Array{T,1})
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
function sumsq{T <: Union(Float32, Float64)}(x::BEDFile; shared::Bool = true, means::DenseArray{T,1} = mean(x, shared=shared), invstds::DenseArray{T,1} = invstd(x, y=means, shared=shared)) 
	y = ifelse(shared, SharedArray(T, x.p + x.p2, init= S -> S[localindexes(S)] = 0.0), zeros(T, x.p + x.p2))
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
function mean(T::Type, x::BEDFile; shared::Bool = true)

	# enforce floating point type
	T <: Union(Float32, Float64) || throw(ArgumentError("Type must be Float32 or Float64"))

	# initialize return vector
	y = ifelse(shared, SharedArray(T, x.p + x.p2, init= S -> S[localindexes(S)] = 0.0), zeros(T, x.p + x.p2))

	if shared
		@sync @inbounds @parallel for snp = 1:x.p
			y[snp] = mean_col(x, snp)
		end
	else
		@inbounds  for snp = 1:x.p
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

# for mean function, set default type to Float64
mean(x::BEDFile; shared::Bool = true) = mean(Float64, x, shared=shared)

function mean_col(x::BEDFile, snp::Integer)
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
#
### WARNING: need to fix type assertions here!
#function invstd(T::Type, x::BEDFile; shared::Bool = true, y::DenseArray{T,1} = mean(T,x, shared=shared))
function invstd(T::Type, x::BEDFile; shared::Bool = true, y::DenseArray{Float64,1} = mean(T,x, shared=shared))

	# enforce floating point type
	T <: Union(Float32, Float64) || throw(ArgumentError("Type must be Float32 or Float64"))

	# initialize return vector
	z = ifelse(shared, SharedArray(T, x.p + x.p2, init= S -> S[localindexes(S)] = 0.0), zeros(T, x.p + x.p2))

	if shared
		@sync @inbounds @parallel for snp = 1:x.p
			z[snp] = invstd_col(x, snp, y)
		end
	else
		@inbounds  for snp = 1:x.p
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

# for invstd function, set default type to Float64
invstd(x::BEDFile; shared::Bool = true, y::DenseArray{Float64,1} = mean(Float64, x, shared=shared)) = invstd(Float64, x, shared=shared, y=y)


function invstd_col{T <: Union(Float32, Float64)}(x::BEDFile, snp::Integer, means::DenseArray{T,1})
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
function update_partial_residuals!{T <: Union(Float32, Float64)}(r::SharedArray{T,1}, y::SharedArray{T,1}, x::BEDFile, perm::SharedArray{Int,1}, b::DenseArray{T,1}, k::Integer; means=mean(x), invstds=invstd(x, y=means), Xb::SharedArray{T,1} = xb(X,b,support,k, means=means, invstds=invstds)) 
	k <= length(b)   || throw(ArgumentError("k cannot exceed the length of b!"))
	length(r) == x.n || throw(DimensionMismatch("r must have length $(x.n)!"))
	length(y) == x.n || throw(DimensionMismatch("y must have length $(x.n)!"))

	@sync @inbounds @parallel for i = 1:x.n
		r[i] = y[i] - Xb[i]
	end

	return r
end

function update_partial_residuals!{T <: Union(Float32, Float64)}(r::Array{T,1}, y::Array{T,1}, x::BEDFile, perm::Array{Int,1}, b::Array{T,1}, k::Integer; means=mean(x), invstds=invstd(x, y=means), Xb::Array{T,1} = xb(X,b,support,k, means=means, invstds=invstds)) 
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
function update_partial_residuals!{T <: Union(Float32, Float64)}(r::SharedArray{T,1}, y::SharedArray{T,1}, x::BEDFile, indices::BitArray{1}, b::SharedArray{T,1}, k::Integer; means::SharedArray{T,1} = mean(x), invstds::SharedArray{T,1} = invstd(x, y=means), Xb::SharedArray{T,1} = xb(X,b,indices,k, means=means, invstds=invstds) ) 
	k <= length(b)    || throw(ArgumentError("k cannot exceed the length of b!"))
#	k <= sum(indices) || throw(ArgumentError("k cannot exceed the number of true values in indices!"))
	length(r) == x.n  || throw(DimensionMismatch("r must have length $(x.n)!"))
	length(y) == x.n  || throw(DimensionMismatch("y must have length $(x.n)!"))

	@sync @inbounds @parallel for i = 1:x.n
		r[i] = y[i] - Xb[i]
	end

	return r
end


function update_partial_residuals!{T <: Union(Float32, Float64)}(r::Array{T,1}, y::Array{T,1}, x::BEDFile, indices::BitArray{1}, b::Array{T,1}, k::Integer; means::Array{T,1} = mean(x), invstds::Array{T,1} = invstd(x, y=means), Xb::Array{T,1} = xb(X,b,indices,k, means=means, invstds=invstds) ) 
	k <= length(b)    || throw(ArgumentError("k cannot exceed the length of b!"))
#	k <= sum(indices) || throw(ArgumentError("k cannot exceed the number of true values in indices!"))
	length(r) == x.n  || throw(DimensionMismatch("r must have length $(x.n)!"))
	length(y) == x.n  || throw(DimensionMismatch("y must have length $(x.n)!"))

	@inbounds for i = 1:x.n
		r[i] = y[i] - Xb[i]
	end

	return r
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
function dot{T <: Union(Float32, Float64)}(x::BEDFile, y::DenseArray{T,1}, snp::Integer, means::DenseArray{T,1}, invstds::DenseArray{T,1}) 
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
function dott{T <: Union(Float32, Float64)}(x::BEDFile, b::DenseArray{T,1}, case::Integer, indices::BitArray{1}, means::DenseArray{T,1}, invstds::DenseArray{T,1}) 
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
			t              = typeof(t) == Float64 ? geno64[genotype + ONE8] : geno32[genotype + ONE8]

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
function xb!{T <: Union(Float32, Float64)}(Xb::DenseArray{T,1}, x::BEDFile, b::DenseArray{T,1}, indices::BitArray{1}, k::Integer; means::DenseArray{T,1} = mean(x), invstds::DenseArray{T,1} = invstd(x, y=means))
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


function xb{T <: Union(Float32, Float64)}(x::BEDFile, b::Array{T,1}, indices::BitArray{1}, k::Integer; means::Array{T,1} = mean(x, shared=false), invstds::Array{T,1} = invstd(x, y=means, shared=false)) 
	Xb = zeros(T,x.n)
	xb!(Xb,x,b,indices,k, means=means, invstds=invstds)
	return Xb
end

function xb{T <: Union(Float32, Float64)}(x::BEDFile, b::SharedArray{T,1}, indices::BitArray{1}, k::Integer; means::SharedArray{T,1} = mean(x, shared=true), invstds::SharedArray{T,1} = invstd(x, y=means, shared=true)) 
	Xb = SharedArray(T, x.n, init = S -> S[localindexes(S)] = 0.0)
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
function xty!{T <: Union(Float32, Float64)}(Xty::SharedArray{T,1}, x::BEDFile, y::SharedArray{T,1}; means::SharedArray{T,1} = mean(x, shared=true), invstds::SharedArray{T,1} = invstd(x, y=means, shared=true)) 

	# error checking
	x.p <= length(Xty) || throw(ArgumentError("Attempting to fill argument Xty of length $(length(Xty)) with $(x.p) elements!"))
	x.n == length(y)   || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.n) of them!"))

	# loop over the desired number of predictors 
	@sync @inbounds @parallel for snp = 1:x.p
		Xty[snp] = dot(x,y,snp,means,invstds)
	end

	return Xty
end 

function xty!{T <: Union(Float32, Float64)}(Xty::Array{T,1}, x::BEDFile, y::Array{T,1}; means::Array{T,1} = mean(x, shared=false), invstds::Array{T,1} = invstd(x, y=means, shared=false)) 

	# error checking
	x.p <= length(Xty) || throw(ArgumentError("Attempting to fill argument Xty of length $(length(Xty)) with $(x.p) elements!"))
	x.n == length(y)   || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.n) of them!"))

	# loop over the desired number of predictors 
	@inbounds for snp = 1:x.p
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
function xty{T <: Union(Float32, Float64)}(x::BEDFile, y::SharedArray{T,1}; means::SharedArray{T,1} = mean(x, shared=true), invstds::SharedArray{T,1} = invstd(x, y=means, shared=true)) 
	Xty = SharedArray(T, x.p + x.p2, init = S -> S[localindexes(S)] = 0.0)
	xty!(Xty,x,y, means=means, invstds=invstds) 
	return Xty
end

function xty{T <: Union(Float32, Float64)}(x::BEDFile, y::Array{T,1}; means::Array{T,1} = mean(x, shared=false), invstds::Array{T,1} = invstd(x, y=means, shared=false)) 
	Xty = zeros(T, x.p + x.p2)
	xty!(Xty,x,y, means=means, invstds=invstds) 
	return Xty
end
