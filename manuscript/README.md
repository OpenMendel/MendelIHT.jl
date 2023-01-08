# Multivariate IHT scripts

This folder contains scripts to reproduce out paper [Multivariate Genomewide Association Analysis with IHT](https://www.biorxiv.org/content/10.1101/2021.08.04.455145v2). If anything is unclear or you cannot reproduce our results, please file an issue on GitHub and I will take a look at it asap.

## NFBC_sim folder

This folder contains scripts used to reproduce our comparison with mv-PLINK and GEMMA. Specifically,
+ `NFBC_chr1_sim.jl`: This is the main script that runs IHT, mvPLINK, and GEMMA. To run with 16 threads, one can do `julia --threads 16 NFBC_chr1_sim.jl N` where `N` is between 1 and 6 representing the 6 sets of experiments done in our paper. Note the software GEMMA and mvPLINK must be executable in the current directory. They are called on lines 498 and 556. Using the script will produce 6 folders `set1`, ..., `set6`, each set contains 100 different simulation results. For set 5 and 6, GEMMA and mvPLINK runs very slowly, so one would need to comment out the code that runs them.
+ `run_repeats.jl`: This is the script I used to submit batch jobs to Hoffman2 computing cluster.
+ `summary.ipynb`: This is a jupyter notebook that contains the code to summarizes the simulation results
+ `summary.txt`: This is the output of `summary.ipynb`. Numbers in it are featured in our paper. 

## UKBB_hyptertension folder

This folder contains runtime code and results from our UK Biobank's 3-trait analysis. The data cleaning code and the code used to query the GWAS catalog is not here, because they are not done by me.
+ `ukbb.jl`: This is the UK Biobank run-time script. This is also featured in the appendix of our paper.
+ `joblog.8774947`: This is the raw output of running `ukbb.jl`. 
+ `cviht.summary.roughpath1.txt`: The is mean squared error for cross validating k = 100, 200, ..., 1000.
+ `cviht.summary.roughpath1.txt`: The is mean squared error for cross validating k = 100, 200, ..., 1000.
+ `cviht.summary.roughpath2.txt`: The is mean squared error for cross validating k = 110, 120, ..., 290. 
+ `cviht.summary.txt`: The is mean squared error for cross validating k = 181, 182, ..., 199.
+ `iht.cov.nodebias.txt` This is the estimated covariance matrix among tarits. One can convert covariance matrix to correlation matrix, which is what we ended up reporting. 
+ `iht.summary.nodebias.txt`: This is the output of a separate IHT run with 1500 iterations, used to refine parameter estimates.

## UKBB_metabolomic folder

This folder contains code and results from out UK Biobank's 18-trait analysis. To get an interactive plot visualizing UKB's discoveries, (1) download this folder, (2) decompress `iht.final.beta.txt.zip`, and (3) run the `plot.ipynb` notebook (one must first install Julia, then the IJulia package). 

+ `data_process.jl`: Scripts used to process/clean the data for analysis
+ `iht.jl`: Script used to run IHT (cross validation + full IHT regression)
+ `plot.ipynb`: Scripts for generating the UKB (interactive) plot for visualizing SNP discoveries
+ `cviht.summary*`: Output files from running different cross validation paths
+ `iht.final.summary.txt`: The output from running IHT on final sparsity `k=4678`
+ `iht.final.cov.txt`: Estimated covariance matrix among 18 phenotypes. The order of phenotypes are listed in `plot.ipynb`
+ `iht.final.beta.txt.zip`: Final estimated beta values for each of the 18 trait. The estimated effect size corresponds to the second allele. 

## extra_sim

This folder contains runtime code and summary results for section 2 of the supplement, which investigates the performance of IHT under traits of different polygenecity.
+ `extra_sim.jl`: This is the script that simulates phenotypes and runs IHT. 
+ `submit.jl`: This is the script that submits multiple independent runs to cluster
+ `summarize.ipynb`: This is a jupyter notebook that summarizes the results produced by running `extra_sim.jl`
