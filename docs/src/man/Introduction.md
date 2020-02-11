
# What is IHT?

Iterative hard thresholding (IHT) is a sparse approximation method that performs variable selection and parameter estimation for high dimensional datasets. IHT returns a sparse model with prespecified $k \in \mathbb{Z}_+$ (or fewer) non-zero entries. In MendelIHT.jl the objective function is:

\begin{align}
\text{maximize } & \quad L(\beta)\\
\text{subject to } & \quad ||\beta||_0 \le k
\end{align}

The objective is solved via:

$$\beta_{n+1} = P_{S_k}(\beta_n - s_n \nabla L(\beta_n))$$

where:

+ $\nabla L(\beta)$ is the score (gradient) vector of loglikelihood

+ $J(\beta)$ is the expected information (hessian) matrix

+ $s = \frac{||\nabla L(\beta)||_2^2}{\nabla L(\beta)^tJ(\beta)\nabla L(\beta)}$ is the step size

+ $P_{S_k}(v)$ projects vector $v$ to sparsity set $S_k$ by setting all but the top $k$ entries to 0. 

See [our paper](https://www.biorxiv.org/content/10.1101/697755v2) for more details and computational tricks to do each of these efficiently.

## Supported GLM models and Link functions

MendelIHT borrows distribution and link functions implementationed in [GLM.jl](http://juliastats.github.io/GLM.jl/stable/) and [Distributions.jl](https://juliastats.github.io/Distributions.jl/stable/).

| Distribution | Canonical Link | Status |
|:---:|:---:|:---:|
| Normal | IdentityLink | $\checkmark$ |
| Bernoulli | LogitLink |$\checkmark$ |
| Poisson | LogLink |  $\checkmark$ |
| NegativeBinomial | LogLink |  $\checkmark$ |
| Gamma | InverseLink | experimental |
| InverseGaussian | InverseSquareLink | experimental |

Examples of these distributions in their default value is visualized in [this post](https://github.com/JuliaStats/GLM.jl/issues/289).

### Available link functions

    CauchitLink
    CloglogLink
    IdentityLink
    InverseLink
    InverseSquareLink
    LogitLink
    LogLink
    ProbitLink
    SqrtLink
