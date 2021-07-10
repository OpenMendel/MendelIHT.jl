
# FAQ

If you do not find your problem here, or the provided solution does not solve your problem, please file an issue on [GitHub](https://github.com/OpenMendel/MendelIHT.jl/issues). 

## Precompilation error

+ The first time one runs `using MendelIHT`, please do so without doing `using Distributed`. Sometimes precompilation can fail in a distributed environment. 
+ On cluster environments, sometimes Julia crashes randomly causing core dumps. If this happens, try running Julia on intel nodes. 

## Parallel computing
+ [How to start Julia with multiple threads](https://docs.julialang.org/en/v1/manual/multi-threading/#Starting-Julia-with-multiple-threads).
+ Execute `Threads.nthreads()` to check if multiple thread is enabled
