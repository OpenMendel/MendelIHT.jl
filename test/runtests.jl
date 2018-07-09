using IHT
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

# write your own tests here
include("MendelIHT_test.jl")
include("MendelIHT_utilities_test.jl")


# To see code coverage, cd to the IHT folder and run the following
# julia -e 'Pkg.test("IHT",coverage=true)'
# @show get_summary(process_file("src/MendelIHT.jl"))
# @show get_summary(process_file("src/MendelIHT_utilities_test.jl"))