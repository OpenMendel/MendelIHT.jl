# Paper 

A preprint of our paper is available on [bioRxiv](https://www.biorxiv.org/content/10.1101/697755v2).

# This folder contains the following:

+ Code to reproduce all simulated data used in figures and tables of our manuscript.
+ Code to reproduce the analysis of simulated and real data which generated the numbers in our figures and tables. 
+ Code to generate all figures

If any part of the code is lacking, ambiguous, and/or does not work, please open a new issue on [github](https://github.com/biona001/MendelIHT.jl/issues) and I will take a look at it ASAP. 

**Note: we display all intermediate results whenever possible.** However, none of the datasets were saved, because:

1. One feature of our package is its ability to generate simulated data, which we illustrate in our figures and tables. 
2. By choosing the same seed, the same simulated data can be generated anywhere. 
3. Some simulated datasets are very large, so simulating remotely is much more efficient than transfering them through the internet. 

## benchmark folder

This folder contains codes to reproduce Figure 2. 

+ `*.jl` are code that performs simulation, runs IHT on simulated data, and saves results. 
+ `*.sh` are scripts that submits the `*.jl` code to UCLA's hoffman2 cluster.
+ `RESPONSE_results` are folders that saves the resulting runtime, memory usage, and iteration count. 
+ `benchmark_figure.ipynb` is a jupyter notebook that reads in different results, computes some summary statistics, and produces Figure 2. 

Note some simulated data is very large. It is not recommended to run these scripts on personal computers, although it should be possible. 

## cross_validation folder

This folder contains code to reproduce Figure 3. 

+ `*.jl` are code that performs simulation, runs IHT on simulated data, and saves results.
+ `*.sh` are scripts that submits the `*.jl` code to UCLA's hoffman2 cluster.
+ `*_cv_drs/memory/run_times` are result after running the `*.jl` files.
+ `drs_figures.ipynb` is the jupyter notebook that reads in different results, computes some summary statistics, and produces Figure 3.  

## precision_recall folder

This folder contains codes to reproduce Table 2. A single jupyter notebook `*.ipynb` for each response is used to simulate data, analyze data, and compute summary statistics. **Each individual results are displayed as well for maximum transparency.** For the Poisson results, we ran the simulations across 5 cores (and hence are saved in 5 different folders) because the pscl package is very slow. 

## repeats folder

This folder contains code to reproduce Table 3. 

+ Four notebooks ending in `*.ipynb` for each response is used to simulate data, analyze data with IHT, and compute summary statistics. **Each individual results are displayed as well for maximum transparency.**

## double_sparse folder

This folder contains code to reproduce Table 4. 

A single jupyter notebook `doubly_sparse_simulated_data.ipynb` performs simulations, runs IHT on simulated data, and computes summary statistics. **Each individual results are displayed as well for maximum transparency.** The summary statistics turned into Table 4. 

The notebook `doubly_sparse-stampeed.ipynb` ran doubly-sparse IHT using real data. This result was not shown because we have no accurate way to obtain group information. 

## weights folder

This folder contains code to reproduce Table 5. 

A single jupyter notebook `weights.ipynb` performs simulations, runs IHT on simulated data and computes summary statistics. **Each individual results are displayed as well for maximum transparency.** The summary statistics was turned into Table 5. 

## ukbiobank folder

This folder contains code to reproduce Table 6. Raw data are not inside these folders, but can be downloaded from [ukbiobank](https://www.ukbiobank.ac.uk/) under Project ID 48152 and 15678. The data filtering process follows exactly the same protocol as 
```
German et al. "Ordered Multinomial Regression for Genetic Association Analysis of Ordinal Phenotypes at Biobank Scale (under review)"
```

## stampeed	folder

This folder contains code to reproduce Table 7. Raw data are not inside these folders, but can be [downloaded from DbGap](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/dataset.cgi?study_id=phs000276.v2.p1&pht=2005) under data accession pht002005.v1.p1.  

In `HDL` subfolder, `analyze_stampeed_HDL.ipynb` is the jupyter notebook that performed all data cleaning, processing, runs IHT, and compute summary statistics. Even steps not done in Julia are explicitly included. For instance, computation of top principal components was done with PLINK 2.0 alpha, but the exact command for doing that are displayed. `analyze_stampeed_HDL_binary.ipynb` contains result for which HDL was analyzed as a binary trait, truncating at 60ml/DL. Finally, we looked at correlation between SNPs associated with HDL, which is performed by the `Analyze correlation.ipynb` notebook. 

In `LDL subfolder`, `LDL_thresholded_145.ipynb` is the jupyter notebook that performed all data cleaning, processing, runs IHT, and compute summary statistics. As above, steps not done in Julia are explicitly included. 

We also tried to fit a Poisson model on C-protein levels, but the result was not displayed in our paper. 