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
const geno32 = [0f0, NaN32, 1.f0, 2f0]
const geno64 = [0.0, NaN, 1.0, 2.0]

include("bedfile.jl")
include("decompression.jl")
include("linalg.jl")
include("logistic.jl")








end	# end module PLINK
