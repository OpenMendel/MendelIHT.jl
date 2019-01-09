module IHT

using Distances: euclidean, chebyshev, sqeuclidean
# using PLINK
#using StatsFuns: logistic, logit, softplus  ## only for logistic IHT, not working right no
using DataFrames
using Gadfly
# using MendelBase
using SnpArrays
using StatsBase
using StatsFuns: logistic
using Random 
using LinearAlgebra
using SpecialFunctions

### idea from Julio Hoffimann Mendes to conditionally load OpenCL module
# only load if Julia can find OpenCL module
# otherwise warn and set "cl" variable to Void
# will load GPU code based on value of "cl"
# try
#     using OpenCL
# catch e
#     warn("IHT.jl cannot find an OpenCL library and will not load GPU functions correctly.")
#     global cl = nothing
# end

# since PLINK module also loads OpenCL, the previous test can fail
# check that PLINK actually loaded GPU code
# try
#     PLINK.PlinkGPUVariables
# catch e
#     warn("PLINK.jl could not find an OpenCL library, so IHT.jl will not load GPU functions correctly.")
#     global cl = nothing
# end


# used for pretty printing of IHTResults, IHTCrossvalidationResults
import Base.show

# used to plot MSEs v. models from IHTCrossvalidationResults
import Gadfly.plot

export L0_reg
export L0_log
export iht_path
export iht_path_log
export cv_iht
export cv_get_folds
export cv_log
export MendelIHT
export L0_logistic_reg
export L0_poisson_reg

export IHTVariables, use_A2_as_minor_allele
export project_k!, std_reciprocal, project_group_sparse!
export update_df!, At_mul_B!, A_mul_B!, check_y_content
export save_prev!, update_mean!, normalize!, simulate_random_snparray

# IHT will only work on single/double precision floats!
const Float = Union{Float64,Float32}

#include("numeric_IHT/common.jl")
# try
#     include("gpu.jl") # conditional load of GPU code
# catch e
#     warn("IHT.jl failed to load GPU functions!")
# end
# include("gwas.jl")
#include("numeric_IHT/cv.jl")
#include("numeric_IHT/hardthreshold.jl")
#include("aiht.jl")
#include("log.jl")
include("data_structures.jl")
include("MendelIHT_utilities.jl")
include("MendelIHT.jl")
include("gwas_normal.jl")
include("gwas_logistic.jl")
include("gwas_poisson.jl")
include("cross_validation.jl")

end # end module IHT
