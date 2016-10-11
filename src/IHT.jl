module IHT

using Distances: euclidean, chebyshev, sqeuclidean
using PLINK
using RegressionTools
using StatsBase
using DataFrames
using Gadfly

# conditional load of OpenCL module
# only load if Julia can find OpenCL module
# this code is copied from OpenCL module
const paths = is_apple() ? String["/System/Library/Frameworks/OpenCL.framework"] : String[]
const libopencl = Libdl.find_library(["libOpenCL", "OpenCL"], paths)
if libopencl == ""
    warn("IHT.jl does not see an OpenCL library and will not load GPU functions")
else
    using OpenCL
end

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

# IHT will only work on single/double precision floats!
typealias Float Union{Float64, Float32}

include("common.jl")
if libopencl != ""
    include("gpu.jl") # conditional load of GPU code
end
include("gwas.jl")
include("cv.jl")
include("hardthreshold.jl")
include("aiht.jl")
include("log.jl")

end # end module IHT
