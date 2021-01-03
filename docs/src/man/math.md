
# Details of Parameter Estimation

This note is meant to supplement our [paper](https://doi.org/10.1093/gigascience/giaa044). 

For a review on generalized linear models, the following resources are recommended:
+ (3rd edition) Chapter 15.3 of [Applied regression analysis and generalized linear models](https://www.amazon.com/Applied-Regression-Analysis-Generalized-Linear/dp/1452205663/ref=sr_1_2?dchild=1&keywords=Applied+Regression+Analysis+and+Generalized+Linear+Models&qid=1609298891&s=books&sr=1-2) by John Fox
+ (3rd edition) Chapter 3-5 of [An introduction to generalized linear models](https://www.amazon.com/Introduction-Generalized-Chapman-Statistical-Science/dp/1138741515/ref=sr_1_2?crid=18BN4MONNYYJH&dchild=1&keywords=an+introduction+to+generalized+linear+models&qid=1609298924&s=books&sprefix=an+introduction+to+ge%2Cstripbooks%2C222&sr=1-2) by Dobson and Barnett

For review on projected gradient descent, I recommend
+ Chapter 5 of [MM optimization algorithms](https://www.amazon.com/MM-Optimization-Algorithms-Kenneth-Lange/dp/1611974399) by Kenneth Lange (2nd edition is almost out, as of 1/2/2021)

## Generalized linear models

In `MendelIHT.jl`, phenotypes $(\bf y)$ are modeled as a [generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model):
```math
\begin{aligned}
    \mu_i = E(y_i) = g({\bf x}_i^t {\boldsymbol \beta})
\end{aligned}
```
where $\bf x$ is sample $i$'s $p$-dimensional vector of *covariates* (genotypes + other fixed effects), $\boldsymbol \beta$ is a $p$-dimensional regression coefficients, $g$ is a non-linear *inverse-link* function, $y_i$ is sample $i$'s phenotype value, and $\mu_i$ is the *average predicted value* of $y_i$ given $\bf x$. 

The full design matrix ${\bf X}_{n \times p}$ and phenotypes ${\bf y}_{n \times 1}$ are observed. The regression coefficients $\boldsymbol \beta$ are not observed and are estimated by *maximum likelihood* methods, traditionally via iteratively reweighted least squares (IRLS). For high dimensional problems where $n < p$, we substitute iterative hard thresholding in place of IRLS. 

GLMs offer a natural way to model common non-continuous phenotypes. For instance, logistic regression for binary phenotypes and Poisson regression for integer valued phenotypes are special cases under the GLM framework. Of course, when $g(\alpha) = \alpha,$ we get standard linear regression used for Gaussian phenotypes. 

## Loglikelihood, gradient, and expected information

In GLM, the distribution of $\bf y$ is from the exponential family with density
```math
\begin{aligned}
    f(y \mid \theta, \phi) = \exp \left[ \frac{y \theta - b(\theta)}{a(\phi)} + c(y, \phi) \right].
\end{aligned}
```
Here $\theta$ is called the *canonical (location) parameter* and under the canonical link, $\theta = g(\bf x^t \bf \beta)$. $\phi$ is the *dispersion (scale) parameter*. The functions $a, b, c$ are known functions that vary depending on the distribution of $y$. 

Given $n$ independent observations, the loglikelihood is:
```math
\begin{aligned}
    L({\bf \theta}, \phi; {\bf y}) &= \sum_{i=1}^n \frac{y_i\theta_i - b(\theta_i)}{a_i(\phi)} + c(y_i, \phi).
\end{aligned}
```
To evaluate the loglikelihood, we use the [logpdf](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.logpdf-Tuple{Distribution{Univariate,S}%20where%20S%3C:ValueSupport,Real}) function in [Distributions.jl](https://github.com/JuliaStats/Distributions.jl).

The perform maximum likelihood estimation, we compute partial derivatives for $\beta$s. The $j$th score component is (eq 4.18 in Dobson):
```math
\begin{aligned}
    \frac{\partial L}{\partial \beta_j} = \sum_{i=1}^n \left[\frac{y_i - \mu_i}{var(y_i)}x_{ij}\left(\frac{\partial \mu_i}{\partial \eta_i}\right)\right].
\end{aligned}
```
Thus the full *gradient* is
```math
\begin{aligned}
    \nabla L&= {\bf X}^t{\bf W}({\bf y} - \boldsymbol\mu), \quad W_{ii} = \frac{1}{var(y_i)}\left(\frac{\partial \mu_i}{\partial \eta_i}\right),
\end{aligned}
```
and similarly, the *expected information* is (eq 4.23 in Dobson):
```math
\begin{aligned}
    J = {\bf X^t\tilde{W}X}, \quad \tilde{W}_{ii} = \frac{1}{var(y_i)}\left(\frac{\partial \mu_i}{\partial \eta_i}\right)^2.
\end{aligned}
```
To evaluate $\nabla L$ and $J$, note ${\bf y}$ and ${\bf X}$ are known, so we just need to calculate $\boldsymbol\mu, \frac{\partial\mu_i}{\partial\eta_i},$ and $var(y_i)$. The first simply uses the inverse link: $\mu_i = g({\bf x}_i^t {\boldsymbol \beta})$. For the second, note $\frac{\partial \mu_i}{\partial\eta_i} = \frac{\partial g({\bf x}_i^t {\boldsymbol \beta})}{\partial{\bf x}_i^t {\boldsymbol \beta}}$ is just the derivative of the link function evaluated at the linear predictor $\eta_i = {\bf x}_i^t {\boldsymbol \beta}$. This is already implemented for various link functions as [mueta](https://github.com/JuliaStats/GLM.jl/blob/master/src/glmtools.jl#L149) in [GLM.jl](https://github.com/JuliaStats/GLM.jl), which we call internally. To compute $var(y_i)$, we note that the exponential family distributions have variance
```math
\begin{aligned}
    var(y) &= a(\phi)b''(\theta) = a(\phi)\frac{\partial^2b(\theta)}{\partial\theta^2} = a(\phi) var(\mu).
\end{aligned}
```
That is, $var(y_i)$ is a product of 2 terms where the first depends solely on $\phi$, and the second solely on $\mu_i = g({\bf x}_i^t {\boldsymbol \beta})$. In our code, we use [glmvar](https://github.com/JuliaStats/GLM.jl/blob/master/src/glmtools.jl#L315) implemented in [GLM.jl](https://github.com/JuliaStats/GLM.jl) to calculate $var(\mu)$. Because $\phi$ is unknown, we assume $a(\phi) = 1$ for all models in computing $W_{ii}$ and $\tilde{W}_{ii}$, except for the negative binomial model. For negative binomial model, we discuss how to estimate $\phi$ and $\boldsymbol\beta$ using alternate block descent below.  

## Iterative hard thresholding

In `MendelIHT.jl`, the loglikelihood is maximized using iterative hard thresholding. This is achieved by repeating the following iteration:
```math
\begin{aligned}
    \boldsymbol\beta_{n+1} = \overbrace{P_{S_k}}^{(3)}\big(\boldsymbol\beta_n + \underbrace{s_n}_{(2)} \overbrace{\nabla f(\boldsymbol\beta_n)}^{(1)}\big)
\end{aligned}
```
where $f$ is the loglikelihood to maximize. Step (1) computes the gradient as previously discussed. Step (2) computes the step size $s_k$. Step (3) evaluates the projection operator $P_{S_k}$, which sets all but $k$ largest entries in magnitude to $0$. To perform $P_{S_k}$, we first partially sort the *dense* vector $\beta_n + s_n \nabla f(\beta_n)$, and set the smallest $k+1 ... n$ entries in magnitude to $0$. Note the step size $s_n$ is derived in our paper to be
```math
\begin{aligned}
    s_n = \frac{||\nabla f(\boldsymbol\beta_n)||_2^2}{\nabla f(\boldsymbol\beta_n)^t J(\boldsymbol\beta_n) \nabla f(\boldsymbol\beta_n)}
\end{aligned}
```
where $J = {\bf X^t\tilde{W}X}$ is the expected information matrix (derived in the previous section) *which should never be explicitly formed*. To evaluate the denominator, observe that 
```math
\begin{aligned}
    \nabla f(\boldsymbol\beta_n)^t J(\boldsymbol\beta_n) \nabla f(\boldsymbol\beta_n) = \left(\nabla f(\boldsymbol\beta_n)^t{\bf X}^t \sqrt(\tilde{W})\right)\left(\sqrt(\tilde{W}){\bf X}\nabla f(\boldsymbol\beta_n)\right).
\end{aligned}
```
Thus one computes ${\bf v} = \sqrt(\tilde{W}){\bf X}\nabla f(\boldsymbol\beta_n)$ and calculate its inner product with itself. 

## Nuisance parameter estimation

Currently `MendelIHT.jl` only estimates nuisance parameter for the Negative Binomial model. Estimation of $\phi$ and $\boldsymbol \beta$ can be achieved with alternating block updates. That is, we run 1 IHT iteration to estimate $\boldsymbol \beta_n$, followed by 1 iteration of Newton or MM update to estimate $\phi_n$. Below we derive the Newton and MM updates. 

Note 1: This feature is provided by our [2019 Bruins in Genomics](https://qcb.ucla.edu/big-summer/big2019-2/) summer student [Vivian Garcia](https://github.com/viviangarcia) and [Francis Adusei](https://github.com/fadusei). 

Note 2: for Gaussian response, one can use the sample variance formula to estimate $\phi$ from the estimated mean $\hat{\mu}$. 

### Parametrization for Negative Binomial model

The negative binomial distribution has density
```math
\begin{aligned}
	P(Y = y) = \binom{y+r-1}{y}p^r(1-p)^y
\end{aligned}
```
where $y$ is the number of failures before the $r$th success and $p$ is the probability of success in each individual trial. Adhering to these definitions, the mean and variance according to [WOLFRAM](https://reference.wolfram.com/language/ref/NegativeBinomialDistribution.html) is 
```math
\begin{aligned}
	\mu_i = \frac{r(1-p_i)}{r}, \quad
	Var(y_i) = \frac{r(1-p_i)}{p_i^2}.
\end{aligned}
```
Note these formula are different than the default on [wikipedia](https://en.wikipedia.org/wiki/Negative_binomial_distribution) because in wiki $y$ is the number of *success* and $r$ is the number of *failure*. 
Therefore, solving for $p_i$, we have 
```math
\begin{aligned}
	p_i = \frac{r}{\mu_i + r} = \frac{r}{e^{\mathbf{x}_i^T\beta} + r} \in (0, 1).
\end{aligned}
```
And indeed this this is how we [parametrize the negative binomial model](https://github.com/OpenMendel/MendelIHT.jl/blob/master/src/utilities.jl#L41). **Importantly, we can interpret $p_i$ as a probability**, since $\mathbf{x}_i^T\beta$ can take on any number between $-\infty$ and $+\infty$ (since $\beta$ and $\mathbf{x}_i$ can have positive and negative entries), so $exp(\mathbf{x}_i^T\beta)\in(0, \infty)$.

We can also try to express $Var(y_i)$ in terms of $\mu_i$ and $r$ by doing some algebra:
```math
\begin{aligned}
	Var(y_i)
	&= \frac{r(1-p_i)}{p_i^2} = \frac{r\left( 1 - \frac{r}{\mu_i + r} \right)}{\frac{r^2}{(\mu_i + r)^2}} = \frac{1}{r}\left(1 - \frac{r}{\mu_i + r}\right)(\mu_i + r)^2 \\
	&= \frac{1}{r} \left[ (\mu_i + r)^2 - r(\mu_r + r) \right] = \frac{1}{r}(\mu_i + r)\mu_i\\
	&= \mu_i \left( \frac{\mu_i}{r} + 1 \right)
\end{aligned}
```
You can verify [in GLM.jl](https://github.com/JuliaStats/GLM.jl/blob/ef246bb8fdbfa3f3058435035d0b0cf42abdd06e/src/glmtools.jl#L320) that this is indeed how they compute the variance of a negative binomial distribution. 

### Estimating nuisance parameter using MM algorithms

The MM algorithm is very stable, but converges much slower than Newton's alogorithm below. Thus use MM only if Newton's method fails.

The loglikelihood for $n$ independent samples under a Negative Binomial model is 
```math
\begin{aligned}
	L(p_1, ..., p_m, r)
	&= \sum_{i=1}^m \ln \binom{y_i+r-1}{y_i} + r\ln(p_i) + y_i\ln(1-p_i)\\
	&= \sum_{i=1}^m \left[ \sum_{j=0}^{y_i - 1} \ln(r+j) + r\ln(p_i) - \ln(y_i!) + y_i\ln(1-p_i) \right]\\
	&\geq \sum_{i=1}^m\left[ \sum_{j=0}^{y_i-1}\frac{r_n}{r_n+j}\ln(r) + c_n + r\ln(p_i) - \ln(y_i!) + y_i \ln(1-p_i) \right]\\
    &\equiv M(p_1, ..., p_m, r)
\end{aligned}
```
The last inequality can be seen by applying Jensen's inequality:
```math
\begin{aligned}
	f\left[ \sum_{i}u_i(\boldsymbol\theta)\right] \leq \sum_{i} \frac{u_i(\boldsymbol\theta_n)}{\sum_j u_j(\boldsymbol\theta_n)}f \left[ \frac{\sum_j u_j(\boldsymbol\theta_n)}{u_i(\boldsymbol\theta_n)} u_i(\boldsymbol\theta)\right]
\end{aligned}
```
to the function $f(u) = - \ln(u).$ Maximizing $M$ over $r$ (i.e. differentiating with respect to $r$ and setting equal to zero, then solving for $r$), we have
```math
\begin{aligned}
    \frac{d}{dr} M
	&= \sum_{i=1}^{m} \left[ \sum_{j=0}^{y_i-1} \frac{r_n}{r_n + j} \frac{1}{r} + \ln(p_i) \right] \\
	&= \sum_{i=1}^{m}\sum_{j=0}^{y_i-1} \frac{r_n}{r_n + j} \frac{1}{r} + \sum_{i=1}^{m}\ln(p_i)\\ 
	&\equiv 0\\
	\iff r_{n+1} &= \frac{-\sum_{i=1}^{m}\sum_{j=0}^{y_i-1} \frac{r_n}{r_n + j}}{\sum_{i=1}^{m}\ln(p_i) } 
\end{aligned}
```
Since $L \ge M$ (M *minorizes* L), maximizing $M$ will maximize $L$. 

### Estimating Nuisance parameter using Newton's method


Since we are dealing with 1 parameter optimization, Newton's method is likely a better candidate due to its quadratic rate of convergence. To estimate the nuisance parameter ($r$), we use maximum likelihood estimates. By $p_i = r / (\mu_i + r)$ in above, we have
```math
\begin{aligned}
	& L(p_1, ..., p_m, r)\\
	=& \sum_{i=1}^m \ln \binom{y_i+r-1}{y_i} + r\ln(p_i) + y_i\ln(1-p_i)\\
	=& \sum_{i=1}^m \left[ \ln\left((y_i+r-1)!\right) - \ln\left(y_i!\right) - \ln\left((r-1)!\right) + r\ln(r) - r\ln(\mu_i+r) + y_i\ln(\mu_i) + y_i\ln(\mu_i + r)\right]\\
	=& \sum_{i=1}^m\left[\ln\left((y_i+r-1)!\right)-\ln(y_i!) - \ln\left((r-1)\right) + r\ln(r) - (r+y_i)\ln(\mu_i + r) + y_i\ln(\mu_i)\right]
\end{aligned}
```
Recalling the definition of [digamma and trigamma functions](https://en.wikipedia.org/wiki/Digamma_function), the first and second derivative of our last expression with respect to $r$ is:
```math
\begin{aligned}
	\frac{d}{dr} L(p_1, ..., p_m, r) = & \sum_{i=1}^m \left[ \operatorname{digamma}(y_i+r) - \operatorname{digamma}(r) + 1 + \ln(r) - \frac{r+y_i}{\mu_i+r} - \ln(\mu_i + r) \right]\\
	\frac{d^2}{dr^2} L(p_1, ..., p_m, r) =&\sum_{i=1}^m \left[ \operatorname{trigamma}(y_i+r) - \operatorname{trigamma}(r) + \frac{1}{r} - \frac{2}{\mu_i + r} + \frac{r+y_i}{(\mu_i + r)^2} \right]
\end{aligned}
```
So the iteration to use is:
```math
\begin{aligned}
	r_{n+1} = r_n - \frac{\frac{d}{dr}L(p_1,...,p_m,r)}{\frac{d^2}{dr^2}L(p_1,...,p_m,r)}.
\end{aligned}
```
For stability, we set the denominator equal to $1$ if it is less than 0. That is, we use gradient descent if the current iteration has non-positive definite Hessian matrices. 
