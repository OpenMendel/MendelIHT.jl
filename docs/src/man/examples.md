
# Examples

Here we give numerous example analysis of GWAS data with `MendelIHT.jl`. 


```julia
# machine information for reproducibility
versioninfo()
```

    Julia Version 1.5.0
    Commit 96786e22cc (2020-08-01 23:44 UTC)
    Platform Info:
      OS: macOS (x86_64-apple-darwin18.7.0)
      CPU: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz
      WORD_SIZE: 64
      LIBM: libopenlibm
      LLVM: libLLVM-9.0.1 (ORCJIT, skylake)



```julia
# add workers needed for parallel computing. Add only as many CPU cores available
using Distributed
addprocs(4)

#load necessary packages for running all examples below
@everywhere begin
    using Revise
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
end
```

## Using MendelIHT.jl

Users are exposed to 2 levels of interface:
+ Wrapper functions [iht()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.iht) and [cross_validate()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.cross_validate). These functions are simple scripts that import data, runs IHT, and writes result to output automatically. Since they are very simplistic, they might fail for whatever reason (please file an issue on GitHub). If so, please use:
+ Core functions [fit_iht()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.fit_iht) and [cv_iht()](https://openmendel.github.io/MendelIHT.jl/latest/man/api/#MendelIHT.cv_iht). Input arguments for these functions must be first imported into Julia by the user manually.

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





    13-element Array{String,1}:
     ".DS_Store"
     "README.md"
     "covariates.txt"
     "cviht.summary.txt"
     "iht.beta.txt"
     "iht.summary.txt"
     "normal.bed"
     "normal.bim"
     "normal.fam"
     "normal_true_beta.txt"
     "phenotypes.txt"
     "simulate.jl"
     "univariate"



Here `covariates.txt` contains non-genetic covariates, `normal.bed/bim/fam` are the PLINK files storing genetic covariates, `phenotypes.txt` are phenotypes for each sample, `normal_true_beta.txt` is the true statistical model used to generate the phenotypes, and `simulate.jl` is the script used to generate all the files. 

### Step 1: Run cross validation to determine best model size

Here phenotypes are stored in the 6th column of `.fam` file. Other covariates are stored separately (which includes a column of 1 as intercept). Here we cross validate $k = 1,2,...20$. 


```julia
mses = cross_validate("normal", Normal, covariates="covariates.txt", phenotypes=6, path=1:20);

# Alternative syntax
# mses = cross_validate("normal", Normal, covariates="covariates.txt", phenotypes=6, path=[1, 5, 10, 15, 20]) # test k = 1, 5, 10, 15, 20
# mses = cross_validate("normal", Normal, covariates="covariates.txt", phenotypes="phenotypes.txt", path=1:20) # when phenotypes are stored separately
```

    [32mCross validating...100%|████████████████████████████████| Time: 0:00:35[39m


    
    
    Crossvalidation Results:
    	k	MSE
    	1	1407.2735970192766
    	2	858.9323667647981
    	3	695.3281033011649
    	4	574.9357159487766
    	5	426.30145951172085
    	6	336.31511946184327
    	7	289.01531777955694
    	8	230.659699154335
    	9	195.438939949433
    	10	199.83469223996426
    	11	201.34513294669145
    	12	203.75379485200406
    	13	208.37926053125014
    	14	213.51428971882075
    	15	221.47325404994524
    	16	219.64716813029995
    	17	221.40881497802621
    	18	227.25440479675385
    	19	235.0540681425773
    	20	236.588333388475
    
    Best k = 9
    


Do not be alarmed if you get slightly different numbers, because cross validation breaks data into training/testing randomly. Set a seed by `Random.seed!(1234)` if you want reproducibility.

### Step 2: Run IHT on best k

According to cross validation, `k = 9` achieves the minimum MSE. Thus we run IHT on the full dataset.


```julia
result = iht("normal", 9, Normal, covariates="covariates.txt", phenotypes=6)
```

    ****                   MendelIHT Version 1.4.0                  ****
    ****     Benjamin Chu, Kevin Keys, Chris German, Hua Zhou       ****
    ****   Jin Zhou, Eric Sobel, Janet Sinsheimer, Kenneth Lange    ****
    ****                                                            ****
    ****                 Please cite our paper!                     ****
    ****         https://doi.org/10.1093/gigascience/giaa044        ****
    
    Running sparse linear regression
    Link functin = IdentityLink()
    Sparsity parameter (k) = 9
    Prior weight scaling = off
    Doubly sparse projection = off
    Debias = off
    Max IHT iterations = 200
    Converging when tol < 0.0001:
    
    Iteration 1: loglikelihood = -1407.0715526824579, backtracks = 0, tol = 0.7903070014734823
    Iteration 2: loglikelihood = -1397.916093195947, backtracks = 0, tol = 0.025737058258134597
    Iteration 3: loglikelihood = -1397.881115073396, backtracks = 0, tol = 0.0016722147159225114
    Iteration 4: loglikelihood = -1397.8807458850451, backtracks = 0, tol = 0.0001440850615835579
    Iteration 5: loglikelihood = -1397.8807416471927, backtracks = 0, tol = 1.674491317444967e-5
    result = 
    IHT estimated 7 nonzero SNP predictors and 2 non-genetic predictors.
    
    Compute time (sec):     0.3655970096588135
    Final loglikelihood:    -1397.8807416471927
    SNP PVE:                0.834374956294435
    Iterations:             5
    
    Selected genetic predictors:
    7×2 DataFrame
     Row │ Position  Estimated_β
         │ Int64     Float64
    ─────┼───────────────────────
       1 │     3137     0.424377
       2 │     4246     0.52343
       3 │     4717     0.922857
       4 │     6290    -0.677832
       5 │     7755    -0.542981
       6 │     8375    -0.792815
       7 │     9415    -2.17998
    
    Selected nongenetic predictors:
    2×2 DataFrame
     Row │ Position  Estimated_β
         │ Int64     Float64
    ─────┼───────────────────────
       1 │        1     1.65223
       2 │        2     0.749866





    
    IHT estimated 7 nonzero SNP predictors and 2 non-genetic predictors.
    
    Compute time (sec):     0.3655970096588135
    Final loglikelihood:    -1397.8807416471927
    SNP PVE:                0.834374956294435
    Iterations:             5
    
    Selected genetic predictors:
    [1m7×2 DataFrame[0m
    [1m Row [0m│[1m Position [0m[1m Estimated_β [0m
    [1m     [0m│[90m Int64    [0m[90m Float64     [0m
    ─────┼───────────────────────
       1 │     3137     0.424377
       2 │     4246     0.52343
       3 │     4717     0.922857
       4 │     6290    -0.677832
       5 │     7755    -0.542981
       6 │     8375    -0.792815
       7 │     9415    -2.17998
    
    Selected nongenetic predictors:
    [1m2×2 DataFrame[0m
    [1m Row [0m│[1m Position [0m[1m Estimated_β [0m
    [1m     [0m│[90m Int64    [0m[90m Float64     [0m
    ─────┼───────────────────────
       1 │        1     1.65223
       2 │        2     0.749866



### Step 3: Examine results

IHT picked 7 SNPs and 2 non-genetic predictors: intercept and sex. The `Position` argument corresponds to the order in which the SNP appeared in the PLINK file, and the `Estimated_β` argument is the estimated effect size for the selected SNPs. To extract more information (for instance to extract `rs` IDs), we can do


```julia
snpdata = SnpData("normal")                   # import PLINK information
snps_idx = findall(!iszero, result.beta)      # indices of SNPs selected by IHT
selected_snps = snpdata.snp_info[snps_idx, :] # see which SNPs are selected
@show selected_snps;
```

    selected_snps = 7×6 DataFrame
     Row │ chromosome  snpid    genetic_distance  position  allele1  allele2
         │ String      String   Float64           Int64     String   String
    ─────┼───────────────────────────────────────────────────────────────────
       1 │ 1           snp3137               0.0         1  1        2
       2 │ 1           snp4246               0.0         1  1        2
       3 │ 1           snp4717               0.0         1  1        2
       4 │ 1           snp6290               0.0         1  1        2
       5 │ 1           snp7755               0.0         1  1        2
       6 │ 1           snp8375               0.0         1  1        2
       7 │ 1           snp9415               0.0         1  1        2


The table above displays the SNP information for the selected SNPs. 

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
Random.seed!(1111)

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

    ****                   MendelIHT Version 1.4.0                  ****
    ****     Benjamin Chu, Kevin Keys, Chris German, Hua Zhou       ****
    ****   Jin Zhou, Eric Sobel, Janet Sinsheimer, Kenneth Lange    ****
    ****                                                            ****
    ****                 Please cite our paper!                     ****
    ****         https://doi.org/10.1093/gigascience/giaa044        ****
    
    Running sparse logistic regression
    Link functin = LogitLink()
    Sparsity parameter (k) = 10
    Prior weight scaling = off
    Doubly sparse projection = off
    Debias = off
    Max IHT iterations = 200
    Converging when tol < 0.0001:
    
    Iteration 1: loglikelihood = -403.91876912829684, backtracks = 0, tol = 0.5882274462947444
    Iteration 2: loglikelihood = -354.24157363893784, backtracks = 0, tol = 0.282513866845802
    Iteration 3: loglikelihood = -347.5483388154264, backtracks = 0, tol = 0.19289827584644761
    Iteration 4: loglikelihood = -335.9715247464944, backtracks = 0, tol = 0.1426996228393492
    Iteration 5: loglikelihood = -334.49756712078744, backtracks = 1, tol = 0.02283147714926763
    Iteration 6: loglikelihood = -333.543219571108, backtracks = 2, tol = 0.019792429262652955
    Iteration 7: loglikelihood = -332.8067268854347, backtracks = 2, tol = 0.019845664939460095
    Iteration 8: loglikelihood = -332.5588563458224, backtracks = 3, tol = 0.00765066824120313
    Iteration 9: loglikelihood = -332.3619297572997, backtracks = 3, tol = 0.006913691748350025
    Iteration 10: loglikelihood = -332.2064289061609, backtracks = 3, tol = 0.0061597575486623014
    Iteration 11: loglikelihood = -332.0840431892422, backtracks = 3, tol = 0.0054730856932040705
    Iteration 12: loglikelihood = -331.98800416752806, backtracks = 3, tol = 0.0048548461339783565
    Iteration 13: loglikelihood = -331.9128408595039, backtracks = 3, tol = 0.004300010671833966
    Iteration 14: loglikelihood = -331.85415737528945, backtracks = 3, tol = 0.0038033387186573
    Iteration 15: loglikelihood = -331.8084401845103, backtracks = 3, tol = 0.003359795402130408
    Iteration 16: loglikelihood = -331.77289417483837, backtracks = 3, tol = 0.0029645876127234955
    Iteration 17: loglikelihood = -331.74530513071767, backtracks = 3, tol = 0.0026131815201996976
    Iteration 18: loglikelihood = -331.72392566719304, backtracks = 3, tol = 0.002301318021364227
    Iteration 19: loglikelihood = -331.707381516887, backtracks = 3, tol = 0.002025024729429744
    Iteration 20: loglikelihood = -331.6945951739729, backtracks = 3, tol = 0.001780622915677454
    Iteration 21: loglikelihood = -331.6847241325898, backtracks = 3, tol = 0.0015647287330161268
    Iteration 22: loglikelihood = -331.6771112518333, backtracks = 3, tol = 0.0013742489121423226
    Iteration 23: loglikelihood = -331.671245094309, backtracks = 3, tol = 0.0012063716814260336
    Iteration 24: loglikelihood = -331.6667283950377, backtracks = 3, tol = 0.0010585539268262742
    Iteration 25: loglikelihood = -331.66325310683357, backtracks = 3, tol = 0.0009285056582307361
    Iteration 26: loglikelihood = -331.66058072924545, backtracks = 3, tol = 0.0008141727680124563
    Iteration 27: loglikelihood = -331.6585268570943, backtracks = 3, tol = 0.0007137189219975595
    Iteration 28: loglikelihood = -331.6569490812431, backtracks = 3, tol = 0.0006255072563945025
    Iteration 29: loglikelihood = -331.65573754039684, backtracks = 3, tol = 0.0005480823925167159
    Iteration 30: loglikelihood = -331.6548075608974, backtracks = 3, tol = 0.00048015313770763916
    Iteration 31: loglikelihood = -331.65409393532707, backtracks = 3, tol = 0.0004205761209236388
    Iteration 32: loglikelihood = -331.6535464833441, backtracks = 3, tol = 0.00036834051544793833
    Iteration 33: loglikelihood = -331.6531266130592, backtracks = 3, tol = 0.00032255392716466075
    Iteration 34: loglikelihood = -331.6528046612902, backtracks = 3, tol = 0.00028242947163362354
    Iteration 35: loglikelihood = -331.65255783889444, backtracks = 3, tol = 0.0002472740235281775
    Iteration 36: loglikelihood = -331.6523686452689, backtracks = 3, tol = 0.00021647759466681234
    Iteration 37: loglikelihood = -331.652223646102, backtracks = 3, tol = 0.00018950377910949886
    Iteration 38: loglikelihood = -331.6521125319518, backtracks = 3, tol = 0.00016588119325870545
    Iteration 39: loglikelihood = -331.6520273936599, backtracks = 3, tol = 0.00014519583372057257
    Iteration 40: loglikelihood = -331.65196216503995, backtracks = 3, tol = 0.0001270842743388486
    Iteration 41: loglikelihood = -331.65191219446064, backtracks = 3, tol = 0.00011122762514495094
    Iteration 42: loglikelihood = -331.6518739156732, backtracks = 3, tol = 9.734617909701182e-5
    result = 
    IHT estimated 8 nonzero SNP predictors and 2 non-genetic predictors.
    
    Compute time (sec):     2.5963549613952637
    Final loglikelihood:    -331.6518739156732
    SNP PVE:                0.4798854810844273
    Iterations:             42
    
    Selected genetic predictors:
    8×2 DataFrame
     Row │ Position  Estimated_β
         │ Int64     Float64
    ─────┼───────────────────────
       1 │     3137     0.503252
       2 │     4246     0.590809
       3 │     4248    -0.37987
       4 │     4717     1.04006
       5 │     6290    -0.741734
       6 │     7755    -0.437585
       7 │     8375    -0.942293
       8 │     9415    -2.11206
    
    Selected nongenetic predictors:
    2×2 DataFrame
     Row │ Position  Estimated_β
         │ Int64     Float64
    ─────┼───────────────────────
       1 │        1      1.03892
       2 │        2      1.5844





    
    IHT estimated 8 nonzero SNP predictors and 2 non-genetic predictors.
    
    Compute time (sec):     2.5963549613952637
    Final loglikelihood:    -331.6518739156732
    SNP PVE:                0.4798854810844273
    Iterations:             42
    
    Selected genetic predictors:
    [1m8×2 DataFrame[0m
    [1m Row [0m│[1m Position [0m[1m Estimated_β [0m
    [1m     [0m│[90m Int64    [0m[90m Float64     [0m
    ─────┼───────────────────────
       1 │     3137     0.503252
       2 │     4246     0.590809
       3 │     4248    -0.37987
       4 │     4717     1.04006
       5 │     6290    -0.741734
       6 │     7755    -0.437585
       7 │     8375    -0.942293
       8 │     9415    -2.11206
    
    Selected nongenetic predictors:
    [1m2×2 DataFrame[0m
    [1m Row [0m│[1m Position [0m[1m Estimated_β [0m
    [1m     [0m│[90m Int64    [0m[90m Float64     [0m
    ─────┼───────────────────────
       1 │        1      1.03892
       2 │        2      1.5844



Since data is simulated, we can compare IHT's estimated effect size with the truth. 


```julia
[true_b[correct_position] result.beta[correct_position]]
```




    8×2 Array{Float64,2}:
      0.469278    0.503252
      0.554408    0.590809
      0.923213    1.04006
      0.0369732   0.0
     -0.625634   -0.741734
     -0.526553   -0.437585
     -0.815561   -0.942293
     -2.18271    -2.11206



The 1st column are the true beta values, and the 2nd column is the estimated values. IHT found 7/8 genetic predictors, and estimates are reasonably close to truth. IHT missed one SNP with very small effect size ($\beta = 0.0369$). The estimated non-genetic effect size is also very close to the truth (1.0 and 1.5). 


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
d = Poisson          # Response follows Poisson distribution (count data)
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

    [32mCross validating...100%|████████████████████████████████| Time: 0:00:12[39m


    
    
    Crossvalidation Results:
    	k	MSE
    	1	1489.8188363695676
    	2	707.6175237350031
    	3	546.8867981545659
    	4	467.2192708681082
    	5	440.03761893872735
    	6	459.9446241516855
    	7	482.3184687223138
    	8	504.0229684333779
    	9	495.12633308066677
    	10	525.4353275609003
    	11	534.0267905856207
    	12	524.761614819788
    	13	558.2726852255062
    	14	561.6025100531801
    	15	561.3898895087017
    	16	555.5897051455378
    	17	618.3872529214121
    	18	655.395210924614
    	19	652.4915677346956
    	20	561.4237250226572
    
    Best k = 5
    


Now run IHT on the full dataset using the best k (achieved at k = 5)


```julia
result = fit_iht(y, x, k=argmin(mses), d=Poisson(), l=LogLink())
```

    ****                   MendelIHT Version 1.4.0                  ****
    ****     Benjamin Chu, Kevin Keys, Chris German, Hua Zhou       ****
    ****   Jin Zhou, Eric Sobel, Janet Sinsheimer, Kenneth Lange    ****
    ****                                                            ****
    ****                 Please cite our paper!                     ****
    ****         https://doi.org/10.1093/gigascience/giaa044        ****
    
    Running sparse Poisson regression
    Link functin = LogLink()
    Sparsity parameter (k) = 5
    Prior weight scaling = off
    Doubly sparse projection = off
    Debias = off
    Max IHT iterations = 200
    Converging when tol < 0.0001:
    
    Iteration 1: loglikelihood = -2847.082501924986, backtracks = 0, tol = 0.2928574304111579
    Iteration 2: loglikelihood = -2465.401434829009, backtracks = 0, tol = 0.05230999409875657
    Iteration 3: loglikelihood = -2376.6519599956155, backtracks = 0, tol = 0.07104164424891486
    Iteration 4: loglikelihood = -2351.5833133504116, backtracks = 0, tol = 0.026208176849564724
    Iteration 5: loglikelihood = -2343.11078282251, backtracks = 0, tol = 0.02023001613423483
    Iteration 6: loglikelihood = -2339.0692529047387, backtracks = 0, tol = 0.011080308351803736
    Iteration 7: loglikelihood = -2337.149479249759, backtracks = 0, tol = 0.00930914236578197
    Iteration 8: loglikelihood = -2336.1781402958745, backtracks = 0, tol = 0.00556416943618412
    Iteration 9: loglikelihood = -2335.6908426969235, backtracks = 0, tol = 0.004609772037770406
    Iteration 10: loglikelihood = -2335.440388139586, backtracks = 0, tol = 0.002861799512474864
    Iteration 11: loglikelihood = -2335.3124548737906, backtracks = 0, tol = 0.002340881298705853
    Iteration 12: loglikelihood = -2335.2463824561282, backtracks = 0, tol = 0.001479832987797599
    Iteration 13: loglikelihood = -2335.2123851939327, backtracks = 0, tol = 0.0012011066579049358
    Iteration 14: loglikelihood = -2335.1947979604856, backtracks = 0, tol = 0.0007661746047595665
    Iteration 15: loglikelihood = -2335.1857186752313, backtracks = 0, tol = 0.0006191995770663082
    Iteration 16: loglikelihood = -2335.1810189052294, backtracks = 0, tol = 0.00039678836835178535
    Iteration 17: loglikelihood = -2335.178588840017, backtracks = 0, tol = 0.00031993814923964465
    Iteration 18: loglikelihood = -2335.177330620363, backtracks = 0, tol = 0.00020549845888243352
    Iteration 19: loglikelihood = -2335.1766795324024, backtracks = 0, tol = 0.00016549800033345948
    Iteration 20: loglikelihood = -2335.176342377269, backtracks = 0, tol = 0.00010642830220001078
    Iteration 21: loglikelihood = -2335.176167840737, backtracks = 0, tol = 8.565816720868677e-5





    
    IHT estimated 4 nonzero SNP predictors and 1 non-genetic predictors.
    
    Compute time (sec):     0.11967015266418457
    Final loglikelihood:    -2335.176167840737
    SNP PVE:                0.09113449276174615
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




    10×2 Array{Float64,2}:
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

**Conclusion:** In this example, we ran IHT on count response with a general `Array{T, 2}` design matrix. Since we used simulated data, we could compare IHT's estimates with the truth. 

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
       1 │      235       -0.2        -0.218172       0.0
       2 │     2673       -0.2        -0.171002      -0.178483
       3 │     2679       -0.2        -0.236793      -0.213098
       4 │     6383       -0.2        -0.228555      -0.224309
       5 │     6389       -0.2        -0.190352      -0.192022
       6 │     6394        0.2         0.215984       0.198447
       7 │     7862        0.2         0.229254       0.224207
       8 │     7864       -0.2        -0.184551      -0.19331
       9 │     7868       -0.2        -0.174773      -0.177359
      10 │     7870       -0.2        -0.192932      -0.208592
      11 │     9481       -0.2         0.0            0.0
      12 │     9491        0.2         0.0            0.0
      13 │     9493        0.2         0.183659       0.175211
      14 │     9494        0.2         0.117548       0.112946
      15 │     9499       -0.2         0.0            0.0
    
    


**Conclusion:** Ungroup IHT actually found 1 more SNPs than grouped IHT. 

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
       1 │     1264   0.252886     0.270233   0.264713
       2 │     1506  -0.0939841    0.0       -0.125803
       3 │     4866  -0.227394    -0.233703  -0.237007
       4 │     5778  -0.510488    -0.507114  -0.494199
       5 │     5833  -0.311969    -0.324309  -0.322663
       6 │     5956  -0.0548168    0.0        0.0
       7 │     6378  -0.0155173    0.0        0.0
       8 │     7007  -0.123301     0.0        0.0
       9 │     7063   0.0183886    0.0        0.0
      10 │     7995  -0.102122     0.0       -0.142201
    
    


**Conclusion**: weighted IHT found 2 extra predictor than non-weighted IHT.

## Other examples and functionalities

Additional features are available as optional parameters in the [fit_iht](https://github.com/OpenMendel/MendelIHT.jl/blob/master/src/fit.jl#L37) function, but they should be treated as **experimental** features. Interested users are encouraged to explore them and please file issues on GitHub if you encounter a problem.
