# File Description

The files in this directory are:

+ PLINK files are stored as `normal.bed`, `normal.bim`, `normal.fam`
+ Phenotypes are stored in 6th column of the `.fam` files and also separatedly in `phenotypes.txt`
+ Non-genetic covariates include intercept and sex, which is stored separately in `covariates.txt`
+ The true (genetic predictors) β used in simulation is stored in `normal_true_beta.txt`
+ `simulate.jl` is the script used to generate the all files above. 

## Simulation details

This example data contains Gaussian phenotypes. The precise model is:

```math
\begin{aligned}
    y_i &\sim N({\bf X \beta + Z \gamma}, \epsilon_i), \quad \epsilon_i \sim N(0, 1)\\
    x_{ij} &\sim \rm Binomial(2, \rho_j)\\
    \rho_j &\sim \rm Uniform(0, 0.5)\\
    \beta_i &\sim \rm N(0, 1)\\
    \beta_{\rm intercept} &= 1\\
    \beta_{\rm sex} &= 1.5
\end{aligned}
```

where there are 8 non-zero $\beta_i$s. Thus a total of $k=10$ SNPs contribute to the phenotype. 
