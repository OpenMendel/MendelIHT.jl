
# Examples

Here we give numerous example analysis of GWAS data with `MendelIHT.jl`. For exact function input/output descriptions, see the manuel's API.


```julia
# machine information for reproducibility
versioninfo()
```

    Julia Version 1.6.0
    Commit f9720dc2eb (2021-03-24 12:55 UTC)
    Platform Info:
      OS: macOS (x86_64-apple-darwin19.6.0)
      CPU: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz
      WORD_SIZE: 64
      LIBM: libopenlibm
      LLVM: libLLVM-11.0.1 (ORCJIT, skylake)
    Environment:
      JULIA_NUM_THREADS = 8



```julia
# load necessary packages for running all examples below
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using Random
using LinearAlgebra
using GLM
using DelimitedFiles
using Statistics
using BenchmarkTools

BLAS.set_num_threads(1) # prevent over subscription with multithreading & BLAS
Random.seed!(1111) # set seed for reproducibility
```




    MersenneTwister(1111)



## Using MendelIHT.jl

Users are exposed to 2 levels of interface:
+ Wrapper functions [iht()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.iht) and [cross_validate()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.cross_validate). These functions are simple scripts that import data, runs IHT, and writes result to output automatically. Since they are very simplistic, they might fail for whatever reason (please file an issue on GitHub). If so, please use:
+ Core functions [fit_iht()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.fit_iht) and [cv_iht()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.cv_iht). Input arguments for these functions must be first imported into Julia by the user manually.

Below we use numerous examples to illustrate how to use these functions separately. 

## Parallel computing


To exploit `MendelIHT.jl`'s parallel processing, [start Julia with multiple threads](https://docs.julialang.org/en/v1/manual/multi-threading/#Starting-Julia-with-multiple-threads). Two levels of shared-memory parallelism is supported.
+ (genotype-matrix)-(vector or matrix) multiplication
+ cross validation

**Note**: If one is running IHT on `Matrix{Float64}`, BLAS should NOT run with multiple threads (execute `BLAS.set_num_threads(1)` before running IHT). This prevents [oversubscription](https://ieeexplore.ieee.org/document/5470434). 


```julia
Threads.nthreads() # show number of threads
```




    8



## Example 1: GWAS with PLINK files

In this example, our data are stored in binary PLINK files:

+ `normal.bed`
+ `normal.bim`
+ `normal.fam`

which contains simulated (Gaussian) phenotypes for $n=1000$ samples and $p=10,000$ SNPs. There are $8$ causal variants and 2 causal non-genetic covariates (intercept and sex). 

These data are present under `MendelIHT/data` directory.


```julia
# change directory to where example data is located
cd(normpath(MendelIHT.datadir()))

# show working directory
@show pwd() 

# show files in current directory
readdir()
```

    pwd() = "/Users/biona001/.julia/dev/MendelIHT/data"





    17-element Vector{String}:
     ".DS_Store"
     "README.md"
     "covariates.txt"
     "cviht.summary.txt"
     "iht.beta.txt"
     "iht.cov.txt"
     "iht.summary.txt"
     "multivariate.bed"
     "multivariate.bim"
     "multivariate.fam"
     "multivariate.phen"
     "normal.bed"
     "normal.bim"
     "normal.fam"
     "normal_true_beta.txt"
     "phenotypes.txt"
     "simulate.jl"



Here `covariates.txt` contains non-genetic covariates (intercept + sex), `normal.bed/bim/fam` are the PLINK files storing genetic covariates, `phenotypes.txt` are phenotypes for each sample, `normal_true_beta.txt` is the true statistical model used to generate the phenotypes, and `simulate.jl` is the script used to generate all the files. 

### Step 1: Run cross validation to determine best model size

Here phenotypes are stored in the 6th column of `.fam` file. Other covariates are stored separately (which includes a column of 1 as intercept). Here we cross validate $k = 1,2,...20$. 

Note the first run might take awhile because Julia needs to compile the code. 


```julia
mses = cross_validate("normal", Normal, covariates="covariates.txt", phenotypes=6, path=1:20,);

# Alternative syntax
# mses = cross_validate("normal", Normal, covariates="covariates.txt", phenotypes=6, path=[1, 5, 10, 15, 20]) # test k = 1, 5, 10, 15, 20
# mses = cross_validate("normal", Normal, covariates="covariates.txt", phenotypes="phenotypes.txt", path=1:20) # when phenotypes are stored separately
```

    ****                   MendelIHT Version 1.4.1                  ****
    ****     Benjamin Chu, Kevin Keys, Chris German, Hua Zhou       ****
    ****   Jin Zhou, Eric Sobel, Janet Sinsheimer, Kenneth Lange    ****
    ****                                                            ****
    ****                 Please cite our paper!                     ****
    ****         https://doi.org/10.1093/gigascience/giaa044        ****
    


    [32mCross validating...100%|████████████████████████████████| Time: 0:00:17[39m


    
    
    Crossvalidation Results:
    	k	MSE
    	1	1383.3887888333788
    	2	714.3108200438915
    	3	630.5562933019421
    	4	599.6952113619866
    	5	438.42328589185536
    	6	327.08547676403236
    	7	304.1177395732532
    	8	297.41346319671766
    	9	227.5681066123959
    	10	202.69232180491358
    	11	206.35218981854467
    	12	209.76930769385618
    	13	213.48278746710366
    	14	215.43361702847125
    	15	215.6684475575618
    	16	223.24291123616155
    	17	221.4060678342564
    	18	276.6464367814522
    	19	230.976587123502
    	20	268.32185480128885
    
    Best k = 10
    


Do not be alarmed if you get slightly different numbers, because cross validation breaks data into training/testing randomly. Set a seed by `Random.seed!(1234)` if you want reproducibility.

### Step 2: Run IHT on best k

According to cross validation, `k = 10` achieves the minimum MSE. Thus we run IHT on the full dataset.


```julia
result = iht("normal", 10, Normal, covariates="covariates.txt", phenotypes=6);
```

    ****                   MendelIHT Version 1.4.1                  ****
    ****     Benjamin Chu, Kevin Keys, Chris German, Hua Zhou       ****
    ****   Jin Zhou, Eric Sobel, Janet Sinsheimer, Kenneth Lange    ****
    ****                                                            ****
    ****                 Please cite our paper!                     ****
    ****         https://doi.org/10.1093/gigascience/giaa044        ****
    
    Running sparse linear regression
    Number of threads = 8
    Link functin = IdentityLink()
    Sparsity parameter (k) = 10
    Prior weight scaling = off
    Doubly sparse projection = off
    Debias = off
    Max IHT iterations = 200
    Converging when tol < 0.0001 and iteration ≥ 5:
    
    Iteration 1: loglikelihood = -1457.8431531776794, backtracks = 0, tol = 0.7683976715709564
    Iteration 2: loglikelihood = -1391.9907038720667, backtracks = 0, tol = 0.11226800885813881
    Iteration 3: loglikelihood = -1390.7113887489165, backtracks = 0, tol = 0.011581483394882052
    Iteration 4: loglikelihood = -1390.6980736000305, backtracks = 0, tol = 0.0010566984279122389
    Iteration 5: loglikelihood = -1390.6978716629346, backtracks = 0, tol = 0.00010752965216198347
    Iteration 6: loglikelihood = -1390.4000085563305, backtracks = 0, tol = 0.03716090961691858
    Iteration 7: loglikelihood = -1390.3012534352063, backtracks = 0, tol = 0.0032563210092611864
    Iteration 8: loglikelihood = -1390.300372706346, backtracks = 0, tol = 0.0003148875004358726
    Iteration 9: loglikelihood = -1390.3003586111483, backtracks = 0, tol = 3.673233513687601e-5


The convergence criteria can be tuned by keywords `tol` and `min_iter`. 

### Step 3: Examine results

IHT picked 8 SNPs and 2 non-genetic predictors: intercept and sex. The `Position` argument corresponds to the order in which the SNP appeared in the PLINK file, and the `Estimated_β` argument is the estimated effect size for the selected SNPs. To extract more information (for instance to extract `rs` IDs), we can do


```julia
snpdata = SnpData("normal")                   # import PLINK information
snps_idx = findall(!iszero, result.beta)      # indices of SNPs selected by IHT
selected_snps = snpdata.snp_info[snps_idx, :] # see which SNPs are selected
@show selected_snps;
```

    selected_snps = 8×6 DataFrame
     Row │ chromosome  snpid    genetic_distance  position  allele1  allele2
         │ String      String   Float64           Int64     String   String
    ─────┼───────────────────────────────────────────────────────────────────
       1 │ 1           snp3136               0.0         1  1        2
       2 │ 1           snp3137               0.0         1  1        2
       3 │ 1           snp4246               0.0         1  1        2
       4 │ 1           snp4717               0.0         1  1        2
       5 │ 1           snp6290               0.0         1  1        2
       6 │ 1           snp7755               0.0         1  1        2
       7 │ 1           snp8375               0.0         1  1        2
       8 │ 1           snp9415               0.0         1  1        2


The table above displays the SNP information for the selected SNPs. Because there's only 7 causal SNPs, we have 1 false positive. 

Since data is simulated, the fields `chromosome`, `snpid`, `genetic_distance`, `position`, `allele1`, and `allele2` are fake. 

## Example 2: How to simulate data

Here we demonstrate how to use `MendelIHT.jl` and [SnpArrays.jl](https://github.com/OpenMendel/SnpArrays.jl) to simulate data, allowing you to design your own genetic studies. Note:
+ For more complex simulation, please use the module [TraitSimulations.jl](https://github.com/OpenMendel/TraitSimulation.jl).  
+ All linear algebra routines involving PLINK files are handled by [SnpArrays.jl](https://github.com/OpenMendel/SnpArrays.jl). 

First we simulate an example PLINK trio (`.bim`, `.bed`, `.fam`) and non-genetic covariates, then we illustrate how to import them. For simplicity, let us simulated indepent SNPs with binary phenotypes. Explicitly, our model is:

$$y_i \sim \rm Bernoulli(\mathbf{x}_i^T\boldsymbol\beta)$$
$$x_{ij} \sim \rm Binomial(2, \rho_j)$$
$$\rho_j \sim \rm Uniform(0, 0.5)$$
$$\beta_i \sim \rm N(0, 1)$$
$$\beta_{\rm intercept} = 1$$
$$\beta_{\rm sex} = 1.5$$


```julia
n = 1000            # number of samples
p = 10000           # number of SNPs
k = 10              # 8 causal SNPs and 2 causal covariates (intercept + sex)
d = Bernoulli       # Binary (continuous) phenotypes
l = LogitLink()     # canonical link function

# set random seed
Random.seed!(0)

# simulate `sim.bed` file with no missing data
x = simulate_random_snparray("sim.bed", n, p)
xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true, impute=true) 

# nongenetic covariate: first column is the intercept, second column is sex: 0 = male 1 = female
z = ones(n, 2) 
z[:, 2] .= rand(0:1, n)
standardize!(@view(z[:, 2:end])) 

# randomly set genetic predictors where causal βᵢ ~ N(0, 1)
true_b = zeros(p) 
true_b[1:k-2] = randn(k-2)
shuffle!(true_b)

# find correct position of genetic predictors
correct_position = findall(!iszero, true_b)

# define effect size of non-genetic predictors: intercept & sex
true_c = [1.0; 1.5] 

# simulate phenotype using genetic and nongenetic predictors
prob = GLM.linkinv.(l, xla * true_b .+ z * true_c) # note genotype-vector multiplication is done with `xla`
y = [rand(d(i)) for i in prob]
y = Float64.(y); # turn y into floating point numbers

# create `sim.bim` and `sim.bam` files using phenotype
make_bim_fam_files(x, y, "sim")

#save covariates and phenotypes (without header)
writedlm("sim.covariates.txt", z, ',')
writedlm("sim.phenotypes.txt", y)
```

!!! note

    Please **standardize** (or at least center) your non-genetic covariates. If you use our `iht()` or `cross_validation()` functions, standardization is automatic. For genotype matrix, `SnpLinAlg` efficiently achieves this standardization. For non-genetic covariates, please use the built-in function `standardize!`. 

## Example 3: Logistic/Poisson/Negative-binomial GWAS

In Example 2, we simulated binary phenotypes, genotypes, non-genetic covariates, and we know true $k = 10$. Let's try running a logistic regression (i.e. phenotype follows the Bernoulli distribution) on this data. 


```julia
result = iht("sim", 10, Bernoulli, covariates="sim.covariates.txt")
```

    ****                   MendelIHT Version 1.4.1                  ****
    ****     Benjamin Chu, Kevin Keys, Chris German, Hua Zhou       ****
    ****   Jin Zhou, Eric Sobel, Janet Sinsheimer, Kenneth Lange    ****
    ****                                                            ****
    ****                 Please cite our paper!                     ****
    ****         https://doi.org/10.1093/gigascience/giaa044        ****
    
    Running sparse logistic regression
    Number of threads = 8
    Link functin = LogitLink()
    Sparsity parameter (k) = 10
    Prior weight scaling = off
    Doubly sparse projection = off
    Debias = off
    Max IHT iterations = 200
    Converging when tol < 0.0001 and iteration ≥ 5:
    
    Iteration 1: loglikelihood = -388.8910083252736, backtracks = 0, tol = 0.6418034752978493
    Iteration 2: loglikelihood = -334.5278475827583, backtracks = 0, tol = 0.3170396015883161
    Iteration 3: loglikelihood = -327.9633012046245, backtracks = 0, tol = 0.2025866533020331
    Iteration 4: loglikelihood = -321.78128260006685, backtracks = 0, tol = 0.1860359951551827
    Iteration 5: loglikelihood = -320.82291695265127, backtracks = 1, tol = 0.019411649026600777
    Iteration 6: loglikelihood = -320.4507210246635, backtracks = 2, tol = 0.009984009889427738
    Iteration 7: loglikelihood = -317.5178129459779, backtracks = 2, tol = 0.10991749703912314
    Iteration 8: loglikelihood = -317.02211222442554, backtracks = 1, tol = 0.013593158611626565
    Iteration 9: loglikelihood = -316.84609828412295, backtracks = 2, tol = 0.007169021739514612
    Iteration 10: loglikelihood = -316.70442301799153, backtracks = 2, tol = 0.007932900581304757
    Iteration 11: loglikelihood = -316.6123017144071, backtracks = 2, tol = 0.006753634000554562
    Iteration 12: loglikelihood = -316.55859748055383, backtracks = 2, tol = 0.005166907901710646
    Iteration 13: loglikelihood = -316.5276537804478, backtracks = 2, tol = 0.00394060866454871
    Iteration 14: loglikelihood = -316.50992605296796, backtracks = 2, tol = 0.0029918249852994103
    Iteration 15: loglikelihood = -316.4998143291673, backtracks = 2, tol = 0.002265270160216051
    Iteration 16: loglikelihood = -316.49406613214404, backtracks = 2, tol = 0.0017110615239077592
    Iteration 17: loglikelihood = -316.49080687297084, backtracks = 2, tol = 0.0012902536156348425
    Iteration 18: loglikelihood = -316.4889624723853, backtracks = 2, tol = 0.0009716277317779244
    Iteration 19: loglikelihood = -316.4879202825298, backtracks = 2, tol = 0.0007309600514247552
    Iteration 20: loglikelihood = -316.48733204721407, backtracks = 2, tol = 0.000549484718269684
    Iteration 21: loglikelihood = -316.4870003149547, backtracks = 2, tol = 0.0004128291474231582
    Iteration 22: loglikelihood = -316.4868133555257, backtracks = 2, tol = 0.00031002536936769156
    Iteration 23: loglikelihood = -316.4867080384853, backtracks = 2, tol = 0.00023274674474752565
    Iteration 24: loglikelihood = -316.4866487332225, backtracks = 2, tol = 0.00017468833019519464
    Iteration 25: loglikelihood = -316.486615346784, backtracks = 2, tol = 0.0001310885323519644
    Iteration 26: loglikelihood = -316.48659655541053, backtracks = 2, tol = 9.835709203914658e-5





    
    IHT estimated 8 nonzero SNP predictors and 2 non-genetic predictors.
    
    Compute time (sec):     0.5662021636962891
    Final loglikelihood:    -316.48659655541053
    SNP PVE:                0.5592583156030205
    Iterations:             26
    
    Selected genetic predictors:
    [1m8×2 DataFrame[0m
    [1m Row [0m│[1m Position [0m[1m Estimated_β [0m
    [1m     [0m│[90m Int64    [0m[90m Float64     [0m
    ─────┼───────────────────────
       1 │      714    -0.7238
       2 │      777    -0.625216
       3 │     1356     1.01887
       4 │     2426     0.392727
       5 │     5080     0.395473
       6 │     5490    -0.720477
       7 │     6299    -2.09706
       8 │     7057     0.661431
    
    Selected nongenetic predictors:
    [1m2×2 DataFrame[0m
    [1m Row [0m│[1m Position [0m[1m Estimated_β [0m
    [1m     [0m│[90m Int64    [0m[90m Float64     [0m
    ─────┼───────────────────────
       1 │        1     0.919708
       2 │        2     1.48015



Since data is simulated, we can compare IHT's estimated effect size with the truth. 


```julia
[true_b[correct_position] result.beta[correct_position]]
```




    8×2 Matrix{Float64}:
     -0.787272  -0.7238
     -0.456783  -0.625216
      1.12735    1.01887
      0.185925   0.0
     -0.891023  -0.720477
     -2.15515   -2.09706
      0.166931   0.0
      0.82265    0.661431



The 1st column are the true beta values, and the 2nd column is the estimated values. IHT found 6/8 genetic predictors, and estimates are reasonably close to truth. IHT missed SNPs with small effect size. With increased sample size, these small effects can be detected. The estimated non-genetic effect size is also very close to the truth (1.0 and 1.5). 


```julia
# remove simulated data once they are no longer needed
rm("sim.bed", force=true)
rm("sim.bim", force=true)
rm("sim.fam", force=true)
rm("sim.covariates.txt", force=true)
rm("sim.phenotypes.txt", force=true)
rm("iht.beta.txt", force=true)
rm("iht.summary.txt", force=true)
rm("cviht.summary.txt", force=true)
```

## Example 4: Running IHT on general matrices

To run IHT on genotypes in VCF files, or other general data, one must call `fit_iht` and `cv_iht` directly. These functions are designed to work on `AbstractArray{T, 2}` type where `T` is a `Float64` or `Float32`. Thus, one must first import the data, and then call `fit_iht` and `cv_iht` on it. Note the vector of 1s (intercept) shouldn't be included in the design matrix itself, as it will be automatically included.

!!! tip

    Check out [VCFTools.jl](https://github.com/OpenMendel/VCFTools.jl) to learn how to import VCF data.

First we simulate some count response using the model:

$$y_i \sim \rm Poisson(\mathbf{x}_i^T \boldsymbol\beta)$$
$$x_{ij} \sim \rm Normal(0, 1)$$
$$\beta_i \sim \rm N(0, 0.3)$$


```julia
n = 1000             # number of samples
p = 10000            # number of SNPs
k = 10               # 9 causal predictors + intercept
d = Poisson          # Response distribution (count data)
l = LogLink()        # canonical link

# set random seed for reproducibility
Random.seed!(2020)

# simulate design matrix
x = randn(n, p)

# simulate response, true model b, and the correct non-0 positions of b
true_b = zeros(p)
true_b[1:k] .= rand(Normal(0, 0.5), k)
shuffle!(true_b)
intercept = 1.0
correct_position = findall(!iszero, true_b)
prob = GLM.linkinv.(l, intercept .+ x * true_b)
clamp!(prob, -20, 20) # prevents overflow
y = [rand(d(i)) for i in prob]
y = Float64.(y); # convert phenotypes to double precision
```

Now we have the response $y$, design matrix $x$. Let's run IHT and compare with truth.


```julia
# first run cross validation 
mses = cv_iht(y, x, path=1:20, d=Poisson(), l=LogLink());
```

    ****                   MendelIHT Version 1.4.1                  ****
    ****     Benjamin Chu, Kevin Keys, Chris German, Hua Zhou       ****
    ****   Jin Zhou, Eric Sobel, Janet Sinsheimer, Kenneth Lange    ****
    ****                                                            ****
    ****                 Please cite our paper!                     ****
    ****         https://doi.org/10.1093/gigascience/giaa044        ****
    


    [32mCross validating...100%|████████████████████████████████| Time: 0:00:25[39m


    
    
    Crossvalidation Results:
    	k	MSE
    	1	1294.4117918373186
    	2	669.4092214451908
    	3	543.3989940052342
    	4	473.50239409691363
    	5	449.5280013487634
    	6	469.726620661539
    	7	476.40540372697865
    	8	541.741342787916
    	9	539.3863798082896
    	10	523.5971500146672
    	11	517.4445374386203
    	12	558.3509823150954
    	13	600.125510865708
    	14	597.7008166652079
    	15	569.2744078914006
    	16	603.7206875133627
    	17	639.488936399919
    	18	643.1761683767324
    	19	646.566741795878
    	20	642.7739157309286
    
    Best k = 5
    


Now run IHT on the full dataset using the best k (achieved at k = 5)


```julia
result = fit_iht(y, x, k=argmin(mses), d=Poisson(), l=LogLink())
```

    ****                   MendelIHT Version 1.4.1                  ****
    ****     Benjamin Chu, Kevin Keys, Chris German, Hua Zhou       ****
    ****   Jin Zhou, Eric Sobel, Janet Sinsheimer, Kenneth Lange    ****
    ****                                                            ****
    ****                 Please cite our paper!                     ****
    ****         https://doi.org/10.1093/gigascience/giaa044        ****
    
    Running sparse Poisson regression
    Number of threads = 8
    Link functin = LogLink()
    Sparsity parameter (k) = 5
    Prior weight scaling = off
    Doubly sparse projection = off
    Debias = off
    Max IHT iterations = 200
    Converging when tol < 0.0001 and iteration ≥ 5:
    
    Iteration 1: loglikelihood = -2847.082501924986, backtracks = 0, tol = 0.2928574304111579
    Iteration 2: loglikelihood = -2465.401434829008, backtracks = 0, tol = 0.052309994098756654
    Iteration 3: loglikelihood = -2376.6519599956146, backtracks = 0, tol = 0.07104164424891495
    Iteration 4: loglikelihood = -2351.583313350411, backtracks = 0, tol = 0.02620817684956463
    Iteration 5: loglikelihood = -2343.1107828225104, backtracks = 0, tol = 0.020230016134234835
    Iteration 6: loglikelihood = -2339.0692529047383, backtracks = 0, tol = 0.01108030835180364
    Iteration 7: loglikelihood = -2337.149479249759, backtracks = 0, tol = 0.009309142365782066
    Iteration 8: loglikelihood = -2336.178140295875, backtracks = 0, tol = 0.005564169436184072
    Iteration 9: loglikelihood = -2335.6908426969235, backtracks = 0, tol = 0.004609772037770407
    Iteration 10: loglikelihood = -2335.440388139586, backtracks = 0, tol = 0.0028617995124748164
    Iteration 11: loglikelihood = -2335.3124548737906, backtracks = 0, tol = 0.002340881298705853
    Iteration 12: loglikelihood = -2335.2463824561282, backtracks = 0, tol = 0.0014798329877975507
    Iteration 13: loglikelihood = -2335.2123851939327, backtracks = 0, tol = 0.0012011066579049358
    Iteration 14: loglikelihood = -2335.1947979604856, backtracks = 0, tol = 0.0007661746047595179
    Iteration 15: loglikelihood = -2335.1857186752313, backtracks = 0, tol = 0.0006191995770663082
    Iteration 16: loglikelihood = -2335.181018905229, backtracks = 0, tol = 0.00039678836835178535
    Iteration 17: loglikelihood = -2335.1785888400173, backtracks = 0, tol = 0.0003199381492395469
    Iteration 18: loglikelihood = -2335.1773306203627, backtracks = 0, tol = 0.00020549845888248242
    Iteration 19: loglikelihood = -2335.1766795324024, backtracks = 0, tol = 0.00016549800033336165
    Iteration 20: loglikelihood = -2335.176342377269, backtracks = 0, tol = 0.0001064283022001086
    Iteration 21: loglikelihood = -2335.176167840737, backtracks = 0, tol = 8.565816720858893e-5





    
    IHT estimated 4 nonzero SNP predictors and 1 non-genetic predictors.
    
    Compute time (sec):     0.2328169345855713
    Final loglikelihood:    -2335.176167840737
    SNP PVE:                0.09113449276174614
    Iterations:             21
    
    Selected genetic predictors:
    [1m4×2 DataFrame[0m
    [1m Row [0m│[1m Position [0m[1m Estimated_β [0m
    [1m     [0m│[90m Int64    [0m[90m Float64     [0m
    ─────┼───────────────────────
       1 │       83    -0.809284
       2 │      989     0.378376
       3 │     4294    -0.274544
       4 │     4459     0.169417
    
    Selected nongenetic predictors:
    [1m1×2 DataFrame[0m
    [1m Row [0m│[1m Position [0m[1m Estimated_β [0m
    [1m     [0m│[90m Int64    [0m[90m Float64     [0m
    ─────┼───────────────────────
       1 │        1      1.26918




```julia
# compare IHT result with truth
[true_b[correct_position] result.beta[correct_position]]
```




    10×2 Matrix{Float64}:
     -1.303      -0.809284
      0.585809    0.378376
     -0.0700563   0.0
     -0.0901341   0.0
     -0.0620201   0.0
     -0.441452   -0.274544
      0.271429    0.169417
     -0.164888    0.0
     -0.0790484   0.0
      0.0829054   0.0



Since many of the true $\beta$ are small, we were only able to find 5 true signals (4 predictors + intercept). 

**Conclusion:** In this example, we ran IHT on count response with a general `Matrix{Float64}` design matrix. Since we used simulated data, we could compare IHT's estimates with the truth. 

## Example 5: Group IHT 

In this example, we show how to include group information to perform doubly sparse projections. Here the final model would contain at most $J = 5$ groups where each group contains limited number of (prespecified) SNPs. For simplicity, we assume the sparsity parameter $k$ is known. 

### Data simulation
To illustrate the effect of group information and prior weights, we generated correlated genotype matrix according to the procedure outlined in [our paper](https://www.biorxiv.org/content/biorxiv/early/2019/11/19/697755.full.pdf). In this example, each SNP belongs to 1 of 500 disjoint groups containing 20 SNPs each; $j = 5$ distinct groups are each assigned $1,2,...,5$ causal SNPs with effect sizes randomly chosen from $\{−0.2,0.2\}$. In all there 15 causal SNPs.  For grouped-IHT, we assume perfect group information. That is, the selected groups containing 1∼5 causative SNPs are assigned maximum within-group sparsity $\lambda_g = 1,2,...,5$. The remaining groups are assigned $\lambda_g = 1$ (i.e. only 1 active predictor are allowed).


```julia
# define problem size
d = NegativeBinomial
l = LogLink()
n = 1000
p = 10000
block_size = 20                  #simulation parameter
num_blocks = Int(p / block_size) #simulation parameter

# set seed
Random.seed!(2019)

# assign group membership
membership = collect(1:num_blocks)
g = zeros(Int64, p + 1)
for i in 1:length(membership)
    for j in 1:block_size
        cur_row = block_size * (i - 1) + j
        g[block_size*(i - 1) + j] = membership[i]
    end
end
g[end] = membership[end]

#simulate correlated snparray
x = simulate_correlated_snparray("tmp.bed", n, p)
intercept = 0.5
x_float = convert(Matrix{Float64}, x, model=ADDITIVE_MODEL, center=true, scale=true)

#simulate true model, where 5 groups each with 1~5 snps contribute
true_b = zeros(p)
true_groups = randperm(num_blocks)[1:5]
sort!(true_groups)
within_group = [randperm(block_size)[1:1], randperm(block_size)[1:2], 
                randperm(block_size)[1:3], randperm(block_size)[1:4], 
                randperm(block_size)[1:5]]
correct_position = zeros(Int64, 15)
for i in 1:5
    cur_group = block_size * (true_groups[i] - 1)
    cur_group_snps = cur_group .+ within_group[i]
    start, last = Int(i*(i-1)/2 + 1), Int(i*(i+1)/2)
    correct_position[start:last] .= cur_group_snps
end
for i in 1:15
    true_b[correct_position[i]] = rand(-1:2:1) * 0.2
end
sort!(correct_position)

# simulate phenotype
r = 10 #nuisance parameter
μ = GLM.linkinv.(l, intercept .+ x_float * true_b)
clamp!(μ, -20, 20)
prob = 1 ./ (1 .+ μ ./ r)
y = [rand(d(r, i)) for i in prob] #number of failures before r success occurs
y = Float64.(y);
```


```julia
#run IHT without groups
ungrouped = fit_iht(y, x_float, k=15, d=NegativeBinomial(), l=LogLink(), verbose=false)

#run doubly sparse (group) IHT by specifying maximum number of SNPs for each group (in order)
max_group_snps = ones(Int, num_blocks)
max_group_snps[true_groups] .= collect(1:5)
variable_group = fit_iht(y, x_float, d=NegativeBinomial(), l=LogLink(), k=max_group_snps, J=5, group=g, verbose=false);
```


```julia
#check result
correct_position = findall(!iszero, true_b)
compare_model = DataFrame(
    position = correct_position,
    correct_β = true_b[correct_position],
    ungrouped_IHT_β = ungrouped.beta[correct_position], 
    grouped_IHT_β = variable_group.beta[correct_position])
@show compare_model
println("\n")

#clean up. Windows user must do this step manually (outside notebook/REPL)
rm("tmp.bed", force=true)
```

    compare_model = 15×4 DataFrame
     Row │ position  correct_β  ungrouped_IHT_β  grouped_IHT_β
         │ Int64     Float64    Float64          Float64
    ─────┼─────────────────────────────────────────────────────
       1 │      963       -0.2        -0.21403        0.0
       2 │     3485       -0.2         0.0           -0.13032
       3 │     3487       -0.2        -0.323509      -0.225267
       4 │     7405        0.2         0.254196       0.260726
       5 │     7407       -0.2        -0.186084      -0.202747
       6 │     7417       -0.2        -0.190491      -0.201521
       7 │     9104       -0.2        -0.189195      -0.201113
       8 │     9110        0.2         0.192222       0.177787
       9 │     9118       -0.2        -0.196494      -0.189983
      10 │     9120        0.2         0.253254       0.248832
      11 │     9206       -0.2        -0.236861      -0.217945
      12 │     9209       -0.2        -0.198633      -0.177085
      13 │     9210       -0.2        -0.172682      -0.186602
      14 │     9211       -0.2        -0.234481      -0.23977
      15 │     9217        0.2         0.227397       0.217969
    
    


**Conclusion:** Ungroup and grouped IHT each found 1 SNP that the other didn't find.  

## Example 6: Linear Regression with prior weights

In this example, we show how to include (predetermined) prior weights for each SNP. You can check out [our paper](https://www.biorxiv.org/content/biorxiv/early/2019/11/19/697755.full.pdf) for references of why/how to choose these weights. In this case, we mimic our paper and randomly set $10\%$ of all SNPs to have a weight of $2.0$. Other predictors have weight of $1.0$. All causal SNPs have weights of $2.0$. Under this scenario, SNPs with weight $2.0$ is twice as likely to enter the model identified by IHT. 

Our model is simulated as:

$$y_i \sim \mathbf{x}_i^T\mathbf{\beta} + \epsilon_i$$
$$x_{ij} \sim \rm Binomial(2, \rho_j)$$
$$\rho_j \sim \rm Uniform(0, 0.5)$$
$$\epsilon_i \sim \rm N(0, 1)$$
$$\beta_i \sim \rm N(0, 0.25)$$


```julia
d = Normal
l = IdentityLink()
n = 1000
p = 10000
k = 10

#random seed
Random.seed!(4)

# construct snpmatrix, covariate files, and true model b
x = simulate_random_snparray("tmp.bed", n, p)
X = convert(Matrix{Float64}, x, center=true, scale=true)
intercept = 1.0
    
#define true_b 
true_b = zeros(p)
true_b[1:10] .= rand(Normal(0, 0.25), k)
shuffle!(true_b)
correct_position = findall(!iszero, true_b)

#simulate phenotypes (e.g. vector y)
prob = GLM.linkinv.(l, intercept .+ X * true_b)
clamp!(prob, -20, 20)
y = [rand(d(i)) for i in prob]
y = Float64.(y);

# construct weight vector
w = ones(p + 1)
w[correct_position] .= 2.0
one_tenth = round(Int, p/10)
idx = rand(1:p, one_tenth)
w[idx] .= 2.0; #randomly set ~1/10 of all predictors to 2
```


```julia
#run weighted and unweighted IHT
unweighted = fit_iht(y, X, k=10, d=Normal(), l=IdentityLink(), verbose=false)
weighted   = fit_iht(y, X, k=10, d=Normal(), l=IdentityLink(), verbose=false, weight=w)

#check result
compare_model = DataFrame(
    position    = correct_position,
    correct     = true_b[correct_position],
    unweighted  = unweighted.beta[correct_position], 
    weighted    = weighted.beta[correct_position])
@show compare_model
println("\n")

#clean up. Windows user must do this step manually (outside notebook/REPL)
rm("tmp.bed", force=true)
```

    compare_model = 10×4 DataFrame
     Row │ position  correct     unweighted  weighted
         │ Int64     Float64     Float64     Float64
    ─────┼─────────────────────────────────────────────
       1 │     1264   0.252886     0.270233   0.280652
       2 │     1506  -0.0939841    0.0       -0.118611
       3 │     4866  -0.227394    -0.233703  -0.232989
       4 │     5778  -0.510488    -0.507114  -0.501733
       5 │     5833  -0.311969    -0.324309  -0.319763
       6 │     5956  -0.0548168    0.0        0.0
       7 │     6378  -0.0155173    0.0        0.0
       8 │     7007  -0.123301     0.0        0.0
       9 │     7063   0.0183886    0.0        0.0
      10 │     7995  -0.102122     0.0       -0.134898
    
    


**Conclusion**: weighted IHT found 2 extra predictor than non-weighted IHT.

## Example 7: Multivariate IHT

When there is multiple quantitative traits, analyzing them jointly is known to be superior than conducting multiple univariate-GWAS ([ref1](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0095923), [ref2](https://www.nature.com/articles/srep38837)). When `MendelIHT.jl` performs a multivariate analysis, 

+ IHT estimates effect of every SNP (covariate) conditioned on every other SNP across traits
+ IHT outputs an estimated covariate matrix among traits
+ IHT estimates proportion of trait variance explained by the genetic predictors


### First simulate data

With $r$ traits, each sample's phenotype $\mathbf{y}_{i} \in \mathbb{R}^{n \times 1}$ is simulated under

$$\mathbf{y}_{i}^{r \times 1} \sim N(\mathbf{B}^{r \times p}\mathbf{x}_{i}^{p \times 1}, \ \ \Sigma_{r \times r})$$

This model assumes each sample is independent. The covariance among traits is specified by $\Sigma$.


```julia
n = 1000  # number of samples
p = 10000 # number of SNPs
k = 10    # number of causal SNPs
r = 2     # number of traits

# set random seed for reproducibility
Random.seed!(2021)

# simulate `.bed` file with no missing data
x = simulate_random_snparray("multivariate.bed", n, p)
xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, impute=false, center=true, scale=true) 

# intercept is the only nongenetic covariate
z = ones(n, 1)
intercepts = randn(r)' # each trait have different intercept

# simulate response y, true model b, and the correct non-0 positions of b
Y, true_Σ, true_b, correct_position = simulate_random_response(xla, k, r, Zu=z*intercepts, overlap=0)
writedlm("multivariate.trait.cov", true_Σ, ',')

# create `.bim` and `.bam` files using phenotype
make_bim_fam_files(x, Y, "multivariate")

# also save phenotypes in separate file
open("multivariate.phen", "w") do io
    for i in 1:n
        println(io, Y[i, 1], ",", Y[i, 2])
    end
end
```

For multivariate IHT, one can store multiple phenotpyes as extra columns in the `.fam` file. The first 10 rows of such a file is visualized below:


```julia
;head multivariate.fam
```

    1	1	0	0	1	-1.4101566028647934	-0.4675708866010868
    2	1	0	0	1	1.519406122085042	-0.1521105344879844
    3	1	0	0	1	-5.121683111513246	1.4417764126708223
    4	1	0	0	1	2.4188275607309677	2.5303163340220953
    5	1	0	0	1	2.6214639873372234	1.005904479060761
    6	1	0	0	1	1.0918272785956382	2.8773472639961106
    7	1	0	0	1	1.6444938174059964	-0.3561578100979898
    8	1	0	0	1	-1.3607927771748423	0.049522727283193846
    9	1	0	0	1	-3.9917926508357624	1.8333328019574022
    10	1	0	0	1	-2.494886509291137	2.518184222337186


Phenotypes can also be stored in a separate file. In this case, we require each subject's phenotype to occupy a different row. The file should not include a header line. Each row should be listed in the same order as in the PLINK and (for multivariate analysis) be comma separated. For example, the first 10 rows of such a file looks like:


```julia
;head multivariate.phen
```

    -1.4101566028647934,-0.4675708866010868
    1.519406122085042,-0.1521105344879844
    -5.121683111513246,1.4417764126708223
    2.4188275607309677,2.5303163340220953
    2.6214639873372234,1.005904479060761
    1.0918272785956382,2.8773472639961106
    1.6444938174059964,-0.3561578100979898
    -1.3607927771748423,0.049522727283193846
    -3.9917926508357624,1.8333328019574022
    -2.494886509291137,2.518184222337186


### Run multivariate IHT

The values specified in `path` corresponds to the total number of non-zero `k` to be tested in cross validation. Since we simulated 10 true genetic predictors and 2 non-genetic predictors (an intercept term for each trait), $k_{true} = 12$. Because non-genetic covariates are not specified, an intercept with automatically be included. 


```julia
# genotypes stored in multivariate.bed and phenotypes in multivariate.phen
mses = cross_validate("multivariate", MvNormal, phenotypes="multivariate.phen", path=1:20);

# use columns 6 and 7 of .fam as phenotypes
# mses = cross_validate("multivariate", MvNormal, phenotypes=[6, 7], path=1:20)

# run directly with xla and Y (note: transpose is necessary to make samples into columns)
# mses = cv_iht(Matrix(Y'), Transpose(xla), path=1:20)
```

    ****                   MendelIHT Version 1.4.1                  ****
    ****     Benjamin Chu, Kevin Keys, Chris German, Hua Zhou       ****
    ****   Jin Zhou, Eric Sobel, Janet Sinsheimer, Kenneth Lange    ****
    ****                                                            ****
    ****                 Please cite our paper!                     ****
    ****         https://doi.org/10.1093/gigascience/giaa044        ****
    


    [32mCross validating...100%|████████████████████████████████| Time: 0:00:08[39m


    
    
    Crossvalidation Results:
    	k	MSE
    	1	2066.2949539850247
    	2	2065.448718184275
    	3	1028.1142776401573
    	4	891.2895081829226
    	5	761.9356018442186
    	6	615.9435020075197
    	7	548.8271710694912
    	8	538.4145437051037
    	9	526.7503996446849
    	10	528.2000153966353
    	11	528.4124971317248
    	12	523.3419138092179
    	13	538.1305969330168
    	14	527.5197299749807
    	15	527.7948630499403
    	16	531.8215074425368
    	17	533.44640197685
    	18	535.7714663457024
    	19	540.7820708871902
    	20	548.1997323593387
    
    Best k = 12
    


The best MSE is achieved at $k=12$. Let's run IHT with this estimate of $k$. Similarly, there are multiple ways to do so:


```julia
# genotypes stored in multivariate.bed and phenotypes in multivariate.phen
result = iht("multivariate", 12, MvNormal, phenotypes="multivariate.phen")

# genotypes stored in multivariate.bed use columns 6 and 7 of .fam as phenotypes
# result = iht("multivariate", 12, MvNormal, phenotypes=[6, 7])

# run cross validation directly with xla and Y (note: transpose is necessary to make samples into columns)
# result = fit_iht(Matrix(Y'), Transpose(xla), k=12)
```

    ****                   MendelIHT Version 1.4.1                  ****
    ****     Benjamin Chu, Kevin Keys, Chris German, Hua Zhou       ****
    ****   Jin Zhou, Eric Sobel, Janet Sinsheimer, Kenneth Lange    ****
    ****                                                            ****
    ****                 Please cite our paper!                     ****
    ****         https://doi.org/10.1093/gigascience/giaa044        ****
    
    Running sparse Multivariate Gaussian regression
    Number of threads = 8
    Link functin = IdentityLink()
    Sparsity parameter (k) = 12
    Prior weight scaling = off
    Doubly sparse projection = off
    Debias = off
    Max IHT iterations = 200
    Converging when tol < 0.0001 and iteration ≥ 5:
    
    Iteration 1: loglikelihood = -2376.4820296189982, backtracks = 0, tol = 0.0
    Iteration 2: loglikelihood = -2376.479923292264, backtracks = 0, tol = 1.1257263586193733e-6
    Iteration 3: loglikelihood = -1957.3455984022858, backtracks = 0, tol = 0.34272673055481445
    Iteration 4: loglikelihood = -1492.3671161424047, backtracks = 0, tol = 0.25917873408698855
    Iteration 5: loglikelihood = -1182.1735594370286, backtracks = 0, tol = 0.23675182454701588
    Iteration 6: loglikelihood = -1173.136551480164, backtracks = 0, tol = 0.07796393235528722
    Iteration 7: loglikelihood = -1172.0113774449553, backtracks = 1, tol = 0.011552571250170897
    Iteration 8: loglikelihood = -1171.889593655075, backtracks = 1, tol = 0.00348271732377267
    Iteration 9: loglikelihood = -1171.8731432400346, backtracks = 1, tol = 0.0009816062764585914
    Iteration 10: loglikelihood = -1171.8696778652588, backtracks = 1, tol = 0.0004151796336081729
    Iteration 11: loglikelihood = -1171.868629414193, backtracks = 1, tol = 0.00023372232580958058
    Iteration 12: loglikelihood = -1171.868259797147, backtracks = 1, tol = 0.0001366081371933243
    Iteration 13: loglikelihood = -1171.8681230933576, backtracks = 1, tol = 8.144197684245368e-5





    
    Compute time (sec):     0.17782807350158691
    Final loglikelihood:    -1171.8681230933576
    Iterations:             13
    Trait 1's SNP PVE:      0.8882386104054829
    Trait 2's SNP PVE:      0.1797217149389984
    
    Estimated trait covariance:
    [1m2×2 DataFrame[0m
    [1m Row [0m│[1m trait1    [0m[1m trait2    [0m
    [1m     [0m│[90m Float64   [0m[90m Float64   [0m
    ─────┼──────────────────────
       1 │  0.907309  -0.131274
       2 │ -0.131274   1.57327
    
    Trait 1: IHT estimated 6 nonzero SNP predictors
    [1m6×2 DataFrame[0m
    [1m Row [0m│[1m Position [0m[1m Estimated_β [0m
    [1m     [0m│[90m Int64    [0m[90m Float64     [0m
    ─────┼───────────────────────
       1 │      134    -0.412059
       2 │      442    -1.22405
       3 │      450    -1.5129
       4 │     1891    -1.45489
       5 │     2557     0.780163
       6 │     3243    -0.833766
    
    Trait 1: IHT estimated 1 non-genetic predictors
    [1m1×2 DataFrame[0m
    [1m Row [0m│[1m Position [0m[1m Estimated_β [0m
    [1m     [0m│[90m Int64    [0m[90m Float64     [0m
    ─────┼───────────────────────
       1 │        1     -0.14915
    
    Trait 2: IHT estimated 4 nonzero SNP predictors
    [1m4×2 DataFrame[0m
    [1m Row [0m│[1m Position [0m[1m Estimated_β [0m
    [1m     [0m│[90m Int64    [0m[90m Float64     [0m
    ─────┼───────────────────────
       1 │     1014    -0.381763
       2 │     1570     0.183475
       3 │     5214     0.346505
       4 │     9385    -0.18681
    
    Trait 2: IHT estimated 1 non-genetic predictors
    [1m1×2 DataFrame[0m
    [1m Row [0m│[1m Position [0m[1m Estimated_β [0m
    [1m     [0m│[90m Int64    [0m[90m Float64     [0m
    ─────┼───────────────────────
       1 │        1     0.812977




The convergence criteria can be tuned by keywords `tol` and `min_iter`. 

### Check answers


```julia
# estimated vs true first beta
β1 = result.beta[1, :]
true_b1_idx = findall(!iszero, true_b[:, 1])
[β1[true_b1_idx] true_b[true_b1_idx, 1]]
```




    7×2 Matrix{Float64}:
     -0.412059  -0.388067
     -1.22405   -1.24972
     -1.5129    -1.53835
      0.0       -0.0034339
     -1.45489   -1.47163
      0.780163   0.758756
     -0.833766  -0.847906




```julia
# estimated vs true second beta
β2 = result.beta[2, :]
true_b2_idx = findall(!iszero, true_b[:, 2])
[β2[true_b2_idx] true_b[true_b2_idx, 2]]
```




    3×2 Matrix{Float64}:
     -0.381763  -0.402269
      0.346505   0.296183
      0.0        0.125965




```julia
# estimated vs true non genetic covariates (intercept)
[result.c intercepts']
```




    2×2 Matrix{Float64}:
     -0.14915   -0.172668
      0.812977   0.729135




```julia
# estimated vs true covariance matrix
[vec(result.Σ) vec(true_Σ)]
```




    4×2 Matrix{Float64}:
      0.907309   0.955563
     -0.131274  -0.0884466
     -0.131274  -0.0884466
      1.57327    1.62573



**Conclusion:** 
+ IHT found 9 true positives: 6/7 causal SNPs for trait 1, 2/3 causal SNPs for trait 2, and 1/2 intercept
+ Because we ran IHT with $k=12$, there are 3 false positives. 
+ Estimated trait covariance matrix closely match the true covariance
+ The proportion of phenotypic trait variances explained by genotypes are 0.88 and 0.15.

## Other examples and functionalities

Additional features are available as optional parameters in the [fit_iht](https://github.com/OpenMendel/MendelIHT.jl/blob/master/src/fit.jl#L37) function, but they should be treated as **experimental** features. Interested users are encouraged to explore them and please file issues on GitHub if you encounter a problem.
