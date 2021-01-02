
# Examples

Here we give numerous example analysis of GWAS data with MendelIHT. 


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
#first add workers needed for parallel computing. Add only as many CPU cores available
using Distributed
addprocs(4)

#load necessary packages for running all examples below
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
using Plots
```

    ┌ Info: Precompiling MendelIHT [921c7187-1484-5754-b919-5d3ed9ac03c4]
    └ @ Base loading.jl:1278


## Example 1: GWAS with PLINK files

For PLINK files, users are exposed to a few simple wrapper functions. For demonstration, we use simulated data under the `data` directory, as shown below. This data simulates quantitative (Gaussian) traits using $n=1000$ samples and $p=10,000$ SNPs. There are $8$ causal variants and 2 causal non-genetic covariates (intercept and sex). 

Start Julia and execute the following:


```julia
# change directory to where example data is located
cd(normpath(MendelIHT.datadir()))

# show working directory
@show pwd() 

# show files in current directory
readdir()
```

    pwd() = "/Users/biona001/.julia/dev/MendelIHT/data"





    7-element Array{String,1}:
     "covariates.txt"
     "normal.bed"
     "normal.bim"
     "normal.fam"
     "normal_true_beta.txt"
     "phenotypes.txt"
     "simulate.jl"



Here `covariates.txt` contains non-genetic covariates, `normal.bed/bim/fam` are the PLINK files storing genetic covariates, `phenotypes.txt` are phenotypes for each sample, `normal_true_beta.txt` is the true statistical model used to generate the phenotypes, and `simulate.jl` is the script used to generate all the files. 

### Step 1: Run cross validation to determine best model size

If phenotypes are stored in the `.fam` file and there are no other covariates (except for the intercept which is automatically included), one can run cross validation as:


```julia
# test k = 1, 2, ..., 20
mses = cross_validate("normal", 1:20)
argmin(mses)

# Alternative syntax
# mses = cross_validate("normal", [1, 5, 10, 15, 20]) # test k = 1, 5, 10, 15, 20
# mses = cross_validate("normal", "covariates.txt", 1:20) # include additional covariates in separate file
# mses = cross_validate("phenotypes.txt", "normal", "covariates.txt", 1:20) # when phenotypes are stored separately
```

    
    
    Crossvalidation Results:
    	k	MSE
    	1	1424.4209463397158
    	2	877.4127442461745
    	3	698.4610947750848
    	4	573.1504682310128
    	5	476.31578846449054
    	6	409.82530303194505
    	7	359.5017949797407
    	8	325.0831239222133
    	9	331.76688175689503
    	10	335.24897480823256
    	11	342.2539099548487
    	12	349.5580549505318
    	13	352.87834253489024
    	14	351.1138715603811
    	15	351.0544198232595
    	16	350.27000489574243
    	17	352.9226806566691
    	18	357.8264018809541
    	19	365.6812419015122
    	20	372.10901493254187





    8



### Step 2: Run IHT on best k

According to cross validation, `k = 8` achieves the minimum MSE. Thus we run IHT on the full dataset.


```julia
result = iht("normal", 8)
```

    ****                   MendelIHT Version 1.2.0                  ****
    ****     Benjamin Chu, Kevin Keys, Chris German, Hua Zhou       ****
    ****   Jin Zhou, Eric Sobel, Janet Sinsheimer, Kenneth Lange    ****
    ****                                                            ****
    ****                 Please cite our paper!                     ****
    ****         https://doi.org/10.1093/gigascience/giaa044        ****
    
    Running sparse linear regression
    Link functin = IdentityLink()
    Sparsity parameter (k) = 8
    Prior weight scaling = off
    Doubly sparse projection = off
    Debias = off
    Converging when tol < 0.0001
    
    Iteration 1: tol = 0.7845860052299409
    Iteration 2: tol = 0.02358096868235321
    Iteration 3: tol = 0.001550076526387469
    Iteration 4: tol = 0.00010521336604120053
    Iteration 5: tol = 8.430366413828275e-6





    
    IHT estimated 7 nonzero SNP predictors and 1 non-genetic predictors.
    
    Compute time (sec):     0.0751640796661377
    Final loglikelihood:    -1627.2792448761559
    Iterations:             5
    
    Selected genetic predictors:
    [1m7×2 DataFrame[0m
    [1m Row [0m│[1m Position [0m[1m Estimated_β [0m
    [1m     [0m│[90m Int64    [0m[90m Float64     [0m
    ─────┼───────────────────────
       1 │     3137     0.411838
       2 │     4246     0.572452
       3 │     4717     0.909215
       4 │     6290    -0.693302
       5 │     7755    -0.54482
       6 │     8375    -0.788884
       7 │     9415    -2.15858
    
    Selected nongenetic predictors:
    [1m1×2 DataFrame[0m
    [1m Row [0m│[1m Position [0m[1m Estimated_β [0m
    [1m     [0m│[90m Int64    [0m[90m Float64     [0m
    ─────┼───────────────────────
       1 │        1      1.65223



### Step 3: Examine results

Here IHT picked 7 SNPs and the intercept as the 8 most significant predictor. The SNP position is the order in which the SNP appeared in the PLINK file. To extract more information (for instance to extract `rs` IDs), we can do


```julia
snpdata = SnpData("normal")                   # import PLINK information
selected_snps = findall(!iszero, result.beta) # indices of SNPs selected by IHT
snpdata.snp_info[selected_snps, :]            # see which SNPs are selected
```




<table class="data-frame"><thead><tr><th></th><th>chromosome</th><th>snpid</th><th>genetic_distance</th><th>position</th><th>allele1</th><th>allele2</th></tr><tr><th></th><th>String</th><th>String</th><th>Float64</th><th>Int64</th><th>String</th><th>String</th></tr></thead><tbody><p>7 rows × 6 columns</p><tr><th>1</th><td>1</td><td>snp3137</td><td>0.0</td><td>1</td><td>1</td><td>2</td></tr><tr><th>2</th><td>1</td><td>snp4246</td><td>0.0</td><td>1</td><td>1</td><td>2</td></tr><tr><th>3</th><td>1</td><td>snp4717</td><td>0.0</td><td>1</td><td>1</td><td>2</td></tr><tr><th>4</th><td>1</td><td>snp6290</td><td>0.0</td><td>1</td><td>1</td><td>2</td></tr><tr><th>5</th><td>1</td><td>snp7755</td><td>0.0</td><td>1</td><td>1</td><td>2</td></tr><tr><th>6</th><td>1</td><td>snp8375</td><td>0.0</td><td>1</td><td>1</td><td>2</td></tr><tr><th>7</th><td>1</td><td>snp9415</td><td>0.0</td><td>1</td><td>1</td><td>2</td></tr></tbody></table>



The table above displays the SNP information for the selected SNPs. Since data is simulated, the fields `genetic_distance`, `position`, `allele1`, `allele2` are arbitrary and `snpid` are fake. 

## Example 2: How to simulate data

Here we demonstrate how to use `MendelIHT.jl` and [SnpArrays.jl](https://github.com/OpenMendel/SnpArrays.jl) to simulate data, allowing you to design your own genetic studies. Note all linear algebra routines involving PLINK files are handled by `SnpArrays.jl`. 

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

    Please **standardize** your non-genetic covariates. If you use our `iht()` or `cross_validation()` functions, standardization is automatic. For genotype matrix, `SnpLinAlg` efficiently achieves this standardization. For non-genetic covariates, please use the built-in function `standardize!`. 

## Example 3: Logistic/Poisson/Negative-binomial GWAS

In Example 2, we simulated binary phenotypes, genotypes, non-genetic covariates, and we know true $k = 10$. Let's try running a logistic regression on this data. This is specified using keyword arguments. 


```julia
result = iht("sim", "sim.covariates.txt", 10, d=Bernoulli(), l=LogitLink())

# other responses
# result = iht("sim", 10, d=Bernoulli(), l=ProbitLink())     # Logistic regression using ProbitLink
# result = iht("sim", 10, d=Poisson(), l=LogLink())          # Poisson regression using canonical link
# result = iht("sim", 10, d=NegativeBinomial(), l=LogLink()) # Negative Binomial regression using canonical link
```

    ****                   MendelIHT Version 1.2.0                  ****
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
    Converging when tol < 0.0001
    
    Iteration 1: tol = 0.5882274462947447
    Iteration 2: tol = 0.2825138668458021
    Iteration 3: tol = 0.19289827584644756
    Iteration 4: tol = 0.14269962283934917
    Iteration 5: tol = 0.022831477149267632
    Iteration 6: tol = 0.019792429262653115
    Iteration 7: tol = 0.019845664939460095
    Iteration 8: tol = 0.00765066824120313
    Iteration 9: tol = 0.006913691748350025
    Iteration 10: tol = 0.006159757548662376
    Iteration 11: tol = 0.0054730856932040705
    Iteration 12: tol = 0.004854846133978282
    Iteration 13: tol = 0.004300010671833966
    Iteration 14: tol = 0.0038033387186573
    Iteration 15: tol = 0.003359795402130408
    Iteration 16: tol = 0.0029645876127234955
    Iteration 17: tol = 0.0026131815201996976
    Iteration 18: tol = 0.002301318021364227
    Iteration 19: tol = 0.002025024729429744
    Iteration 20: tol = 0.001780622915677454
    Iteration 21: tol = 0.0015647287330161268
    Iteration 22: tol = 0.0013742489121423226
    Iteration 23: tol = 0.0012063716814260336
    Iteration 24: tol = 0.0010585539268262742
    Iteration 25: tol = 0.0009285056582307361
    Iteration 26: tol = 0.0008141727680124563
    Iteration 27: tol = 0.0007137189219975595
    Iteration 28: tol = 0.0006255072563945025
    Iteration 29: tol = 0.0005480823925167159
    Iteration 30: tol = 0.00048015313770763916
    Iteration 31: tol = 0.0004205761209236388
    Iteration 32: tol = 0.00036834051544793833
    Iteration 33: tol = 0.00032255392716466075
    Iteration 34: tol = 0.00028242947163362354
    Iteration 35: tol = 0.0002472740235281775
    Iteration 36: tol = 0.00021647759466681234
    Iteration 37: tol = 0.00018950377910949886
    Iteration 38: tol = 0.00016588119325870545
    Iteration 39: tol = 0.00014519583372057257
    Iteration 40: tol = 0.0001270842743388486
    Iteration 41: tol = 0.00011122762514495094
    Iteration 42: tol = 9.734617909701182e-5





    
    IHT estimated 8 nonzero SNP predictors and 2 non-genetic predictors.
    
    Compute time (sec):     0.22192096710205078
    Final loglikelihood:    -331.6518739156732
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
      0.469278    0.406926
      0.554408    0.493509
      0.923213    0.76469
      0.0369732   0.0
     -0.625634   -0.550352
     -0.526553   -0.311351
     -0.815561   -0.677824
     -2.18271    -1.56627



IHT found 7/8 genetic predictors, and estimates are reasonably close to truth. IHT missed one SNP with very small effect size ($\beta = 0.0369$). The estimated non-genetic effect size is also very close to the truth (1.0 and 1.5). 


```julia
# remove simulated data once they are no longer needed
rm("sim.bed", force=true)
rm("sim.bim", force=true)
rm("sim.fam", force=true)
rm("sim.covariates.txt", force=true)
rm("sim.phenotypes.txt", force=true)
```

## Example 4: Running IHT on general matrices

To run IHT on genotypes in VCF files, or other general data, one must call `fit` and `cv_iht` directly. These functions are designed to work on `AbstractArray{T, 2}` type where `T` is a `Float64` or `Float32`. Thus, one must first import the data, and then call `fit` and `cv_iht` on it. Note the vector of 1s (intercept) shouldn't be included in the design matrix itself, as it will be automatically included.

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

    
    
    Crossvalidation Results:
    	k	MSE
    	1	1486.5413848514968
    	2	705.22250019531
    	3	541.7969789881145
    	4	465.1963943709811
    	5	442.6796495698108
    	6	458.13373854130487
    	7	460.9838544350322
    	8	484.25939604486814
    	9	474.21817442883844
    	10	505.0327194683676
    	11	499.2379415031341
    	12	504.0934101269238
    	13	487.9485902855123
    	14	548.1525696940757
    	15	507.86874709147395
    	16	530.6537481762397
    	17	511.2047414611475
    	18	583.6122865525828
    	19	548.3901670703196
    	20	569.6848731697289



```julia
# run IHT on best k (achieved at k = 5)
result = fit(y, x, k=argmin(mses), d=Poisson(), l=LogLink())
```

    ****                   MendelIHT Version 1.2.0                  ****
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
    Converging when tol < 0.0001
    
    Iteration 1: tol = 0.2928574304111577
    Iteration 2: tol = 0.05230999409875649
    Iteration 3: tol = 0.07104164424891493
    Iteration 4: tol = 0.026208176849564724
    Iteration 5: tol = 0.02023001613423483
    Iteration 6: tol = 0.011080308351803689
    Iteration 7: tol = 0.00930914236578197
    Iteration 8: tol = 0.00556416943618412
    Iteration 9: tol = 0.004609772037770406
    Iteration 10: tol = 0.002861799512474864
    Iteration 11: tol = 0.002340881298705853
    Iteration 12: tol = 0.001479832987797599
    Iteration 13: tol = 0.0012011066579049358
    Iteration 14: tol = 0.0007661746047595665
    Iteration 15: tol = 0.0006191995770663082
    Iteration 16: tol = 0.00039678836835178535
    Iteration 17: tol = 0.00031993814923964465
    Iteration 18: tol = 0.00020549845888243352
    Iteration 19: tol = 0.00016549800033345948
    Iteration 20: tol = 0.00010642830220001078
    Iteration 21: tol = 8.565816720868677e-5





    
    IHT estimated 4 nonzero SNP predictors and 1 non-genetic predictors.
    
    Compute time (sec):     0.08859395980834961
    Final loglikelihood:    -2335.176167840737
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



In this example, we ran IHT on count response with a general `Array{T, 2}` design matrix. Since many of the true $\beta$ are small, we were only able to find 5 true signals (4 predictors + intercept). 

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
x = simulate_correlated_snparray(n, p, "tmp.bed")
z = ones(n, 1) # the intercept
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
μ = GLM.linkinv.(l, x_float * true_b)
clamp!(μ, -20, 20)
prob = 1 ./ (1 .+ μ ./ r)
y = [rand(d(r, i)) for i in prob] #number of failures before r success occurs
y = Float64.(y);
```


```julia
#run IHT without groups
k = 15
ungrouped = L0_reg(x_float, z, y, 1, k, d(), l, verbose=false)
```




    
    IHT estimated 15 nonzero SNP predictors and 0 non-genetic predictors.
    
    Compute time (sec):     0.11840415000915527
    Final loglikelihood:    -1441.522293255591
    Iterations:             27
    
    Selected genetic predictors:
    15×2 DataFrame
    │ Row │ Position │ Estimated_β │
    │     │ [90mInt64[39m    │ [90mFloat64[39m     │
    ├─────┼──────────┼─────────────┤
    │ 1   │ 3464     │ -0.234958   │
    │ 2   │ 4383     │ -0.135693   │
    │ 3   │ 4927     │ 0.158171    │
    │ 4   │ 4938     │ -0.222613   │
    │ 5   │ 5001     │ -0.193739   │
    │ 6   │ 5011     │ -0.162718   │
    │ 7   │ 5018     │ -0.190532   │
    │ 8   │ 5090     │ 0.226509    │
    │ 9   │ 5092     │ -0.17756    │
    │ 10  │ 5100     │ -0.140337   │
    │ 11  │ 7004     │ 0.151748    │
    │ 12  │ 7011     │ 0.206449    │
    │ 13  │ 7015     │ -0.284706   │
    │ 14  │ 7016     │ 0.218126    │
    │ 15  │ 9902     │ 0.119059    │
    
    Selected nongenetic predictors:
    0×2 DataFrame





```julia
#run doubly sparse (group) IHT by specifying maximum number of SNPs for each group (in order)
J = 5
max_group_snps = ones(Int, num_blocks)
max_group_snps[true_groups] .= collect(1:5)
variable_group = L0_reg(x_float, z, y, J, max_group_snps, d(), l, verbose=false, group=g)
```




    
    IHT estimated 15 nonzero SNP predictors and 0 non-genetic predictors.
    
    Compute time (sec):     0.30719614028930664
    Final loglikelihood:    -1446.3808810786898
    Iterations:             16
    
    Selected genetic predictors:
    15×2 DataFrame
    │ Row │ Position │ Estimated_β │
    │     │ [90mInt64[39m    │ [90mFloat64[39m     │
    ├─────┼──────────┼─────────────┤
    │ 1   │ 3464     │ -0.245853   │
    │ 2   │ 4927     │ 0.160904    │
    │ 3   │ 4938     │ -0.213439   │
    │ 4   │ 5001     │ -0.19624    │
    │ 5   │ 5011     │ -0.149913   │
    │ 6   │ 5018     │ -0.181966   │
    │ 7   │ 5086     │ -0.0560478  │
    │ 8   │ 5090     │ 0.21164     │
    │ 9   │ 5092     │ -0.141968   │
    │ 10  │ 5100     │ -0.157655   │
    │ 11  │ 7004     │ 0.190224    │
    │ 12  │ 7011     │ 0.21294     │
    │ 13  │ 7015     │ -0.256058   │
    │ 14  │ 7016     │ 0.19746     │
    │ 15  │ 7020     │ 0.111755    │
    
    Selected nongenetic predictors:
    0×2 DataFrame




### Group IHT found 1 more SNPs than ungrouped IHT


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
    │ Row │ position │ correct_β │ ungrouped_IHT_β │ grouped_IHT_β │
    │     │ Int64    │ Float64   │ Float64         │ Float64       │
    ├─────┼──────────┼───────────┼─────────────────┼───────────────┤
    │ 1   │ 3464     │ -0.2      │ -0.234958       │ -0.245853     │
    │ 2   │ 4927     │ 0.2       │ 0.158171        │ 0.160904      │
    │ 3   │ 4938     │ -0.2      │ -0.222613       │ -0.213439     │
    │ 4   │ 5001     │ -0.2      │ -0.193739       │ -0.19624      │
    │ 5   │ 5011     │ -0.2      │ -0.162718       │ -0.149913     │
    │ 6   │ 5018     │ -0.2      │ -0.190532       │ -0.181966     │
    │ 7   │ 5084     │ -0.2      │ 0.0             │ 0.0           │
    │ 8   │ 5090     │ 0.2       │ 0.226509        │ 0.21164       │
    │ 9   │ 5098     │ -0.2      │ 0.0             │ 0.0           │
    │ 10  │ 5100     │ -0.2      │ -0.140337       │ -0.157655     │
    │ 11  │ 7004     │ 0.2       │ 0.151748        │ 0.190224      │
    │ 12  │ 7011     │ 0.2       │ 0.206449        │ 0.21294       │
    │ 13  │ 7015     │ -0.2      │ -0.284706       │ -0.256058     │
    │ 14  │ 7016     │ 0.2       │ 0.218126        │ 0.19746       │
    │ 15  │ 7020     │ 0.2       │ 0.0             │ 0.111755      │
    
    


## Example 6: Linear Regression with prior weights

In this example, we show how to include (predetermined) prior weights for each SNP. You can check out [our paper](https://www.biorxiv.org/content/biorxiv/early/2019/11/19/697755.full.pdf) for references of why/how to choose these weights. In this case, we mimic our paper and randomly set $10\%$ of all SNPs to have a weight of $2.0$. Other predictors have weight of $1.0$. All causal SNPs have weights of $2.0$. Under this scenario, SNPs with weight $2.0$ is twice as likely to enter the model identified by IHT. 

Our model is simulated as:

$$y_i \sim \mathbf{x}_i^T\mathbf{\beta} + \epsilon_i$$
$$x_{ij} \sim \rm Binomial(2, \rho_j)$$
$$\rho_j \sim \rm Uniform(0, 0.5)$$
$$\epsilon_i \sim \rm N(0, 1)$$
$$\beta_i \sim \rm N(0, 1)$$


```julia
#random seed
Random.seed!(4)

d = Normal
l = canonicallink(d())
n = 1000
p = 10000
k = 10

# construct snpmatrix, covariate files, and true model b
x = simulate_random_snparray(n, p, "tmp.bed")
X = convert(Matrix{Float64}, x, center=true, scale=true)
z = ones(n, 1) # the intercept
    
#define true_b 
true_b = zeros(p)
true_b[1:10] .= collect(0.1:0.1:1.0)
shuffle!(true_b)
correct_position = findall(!iszero, true_b)

#simulate phenotypes (e.g. vector y)
prob = GLM.linkinv.(l, X * true_b)
clamp!(prob, -20, 20)
y = [rand(d(i)) for i in prob]
y = Float64.(y);
```


```julia
# construct weight vector
w = ones(p + 1)
w[correct_position] .= 2.0
one_tenth = round(Int, p/10)
idx = rand(1:p, one_tenth)
w[idx] .= 2.0; #randomly set ~1/10 of all predictors to 2
```


```julia
#run IHT
unweighted = L0_reg(X, z, y, 1, k, d(), l, verbose=false)
weighted   = L0_reg(X, z, y, 1, k, d(), l, verbose=false, weight=w)

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
    │ Row │ position │ correct │ unweighted │ weighted │
    │     │ Int64    │ Float64 │ Float64    │ Float64  │
    ├─────┼──────────┼─────────┼────────────┼──────────┤
    │ 1   │ 1254     │ 0.4     │ 0.452245   │ 0.450405 │
    │ 2   │ 1495     │ 0.3     │ 0.306081   │ 0.305738 │
    │ 3   │ 4856     │ 0.8     │ 0.853536   │ 0.862223 │
    │ 4   │ 5767     │ 0.1     │ 0.0        │ 0.117286 │
    │ 5   │ 5822     │ 0.7     │ 0.656213   │ 0.651908 │
    │ 6   │ 5945     │ 0.9     │ 0.891915   │ 0.894997 │
    │ 7   │ 6367     │ 0.5     │ 0.469718   │ 0.472524 │
    │ 8   │ 6996     │ 1.0     │ 0.963236   │ 0.973512 │
    │ 9   │ 7052     │ 0.6     │ 0.602162   │ 0.600055 │
    │ 10  │ 7980     │ 0.2     │ 0.231389   │ 0.234094 │
    
    


In this case, weighted IHT found an extra predictor than non-weighted IHT.

## Other examples and functionalities

We explored a few more examples in our manuscript, with [reproducible code](https://github.com/biona001/MendelIHT.jl/tree/master/figures). We invite users to experiment with them as well. 
