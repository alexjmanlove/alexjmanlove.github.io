---
title: Notes on Linear Regression
date: 2020-03-08
categories: [Statistics]
tags: [linear regression, statistics, supervised learning]
math: true
---

>Linear regression is a quintessential technique in  statistical modelling. In this document we first explore the intuition behind linear regression from the perspective of optimisation theory, before contrasting this with a statistical interpretation under a probabilstic framework.

---

## **What is regression?**

Regression in machine learning refers to a subset of supervised learning techniques which aim to predict the value of a target output given a corresponding input. 

In general in the supervised setting we have some $N$-set of input variables $\mathbf X = [\mathbf x_1, ..., \mathbf x_N]$ and an associated set of output variables $$\mathbf y = [y_1,...,y_N]$$. Each observation in our dataset $$\mathcal D$$ contains the ordered tuples of input and the output, $$\mathcal{D} = \left\{(x_1, y_1), ..., (x_N, y_N)\right\}$$. 

We assume the existence of some underlying latent function $f$ which relates the inputs to the outputs such that $$y_n = f(\mathbf x_n)$$. In the noisy setting, we assume that our observed outputs are corrupted by noise $\varepsilon$, so that $$y_n = f(\mathbf x_n) + \varepsilon$$. To that extent we might describe our data as being comprised of some part true signal and some part noise. 

Our task is to identify and capture the true signal contained within the data in order to learn the structure of the data generating process and to make predictions for unseen values at novel test inputs.  

>There are many synonyms used to describe the concepts of *inputs* and *outputs* across the various fields of maths, statistics and the sciences. Here are a few examples:
> - **Inputs**: &nbsp; features, &nbsp; predictors, &nbsp; covariates, &nbsp; independent variables.
> - **Outputs**: &nbsp; targets, &nbsp; outcomes, &nbsp; responses, &nbsp; dependent variables. 
{:.prompt-info}

## **Model Formulation**

To facilitate the modelling process, we often permit some simplifying assumptions about the data or the data generating process. In the case of linear regression, we assume that the underlying relationship between the inputs and outputs is linear. 

### **Linear Least Squares Regression**.

The straightforward and simple way to think about linear regression is simply finding the straight line that fits nicely through all the data points. We have an $N$-set of $D$-dimensional features or inputs $\mathbf X = [\mathbf x_1,...,\mathbf x_N]$ which we believe is linearly related to a target variable $\mathbf y = [y_1,...,y_N]$. This is to say that we believe there exists some coefficients $\beta \in \mathbb R^D$ and intercept $\beta_0$ such that, $\ \mathbf y = \beta \ \mathbf X + \beta_0. \ $ In other words we have, $y_n = \beta_0 + \beta_1 x_{n1} + ... + \beta_D x_{nD}. \ $ The $N \times D$ dimensional data matrix $\mathbf X$ is known as the design matrix whose rows correspond to our independent samples $\mathbf x_n \in \mathbb{R}^D$. 

For notational convenience we often prepend a one to each feature vector and denote all linear parameters $\beta,\beta_0$ as a single parameter vector $\mathbf \beta = (\beta_0, \beta_1, ..., \beta_D).$ Then we simply denote $\mathbf{\hat{y}} = \beta \ \mathbf{X}$ as the matrix product of augmented design matrix $\mathbf{X} \in \mathbb R^{N \times (D+1)}$ and the parameter vector $\beta.$

Our aim is to use our sample of data $$\mathcal{D} = \left\{(x_1, y_1), ..., (x_N, y_N)\right\}$$ to estimate appropriate values for $\beta$. That is, we want to find values $\hat \beta$ which provide optimal target predictions $\mathbf{\hat{y}} = \hat \beta \ \mathbf x$ so that $\mathbf{\hat{y}}$ is as close as possible to the true observed target variables $\mathbf y.$ 

Mathematically we formulate this as a **minimisation of squared error terms**, which is why the method is often called *linear least squares regression*. We formulate a loss function and seek to minimise it. We want to find $\hat \beta$ and $\hat c$ which solve 

$$
\arg \min_{\beta, c}\left\{ \ ||\mathbf y - \mathbf{\hat{y}} ||^2 \ \right\} \ = \  \arg \min_{\beta, c}\left\{ \ \sum_{n=1}^N (y_n - \hat y_n)^2 \ \right\}, \phantom{123}\text{where }\mathbf{\hat{y}} = \hat \beta \ \mathbf X.
$$

This optimisation problem has an analytical solution, and can be solved exactly so long as the matrix $\mathbf{X}^T\mathbf{X}$ is invertible (where $\mathbf{X}^T$ is the transpose of $\mathbf{X}$). Let $L(\beta; \mathcal D)$ denote our objective which is a function with respect $\beta$ conditioned on the observed data $\mathcal D$. We have

$$
L(\beta; \mathcal D) = || \mathbf y - \mathbf X \beta || = (\mathbf y - \mathbf X \beta)^T(\mathbf y - \mathbf X \beta).
$$

We differentiate this objective to determine the optimising values of the vectors coefficients $\beta$. We get, 

$$
\frac{\partial}{\partial \beta}\left[(\mathbf y - \mathbf X \beta)^T(\mathbf y - \mathbf X \beta)\right ] = 2 \mathbf X^T (\mathbf y - \mathbf X \beta).
$$

Setting this quantity equal to zero and rearranging, we get, 

$$
\begin{align*}
2 \mathbf X^T (\mathbf y - \mathbf X \beta) 
& = 0 
\\ 
\implies \ \mathbf X^T\mathbf X \beta &= \mathbf X^T \mathbf y
\\ 
\implies \qquad \ \ \beta &= (\mathbf X^T\mathbf X)^{-1} \mathbf X^T \mathbf y.
\end{align*}
$$

We obtain a closed-form solution for the optimal value estimate of the parameter vector $\hat \beta$. This is given by 

$$
\hat \beta = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

This equation is known as the **Normal Equation**. The term $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is essentially a projection matrix, projecting the target vector $\mathbf{y}$ onto the space spanned by the columns of $\mathbf{X}$. 

This deterministic analytical approach to linear regression, encapsulated by the Normal Equation, offers a clear and direct method for parameter estimation. It hinges on the assumption of a linear relationship between the independent variables and the dependent variable, and it focuses on minimizing the sum of squared differences between observed and predicted values. We might intuitively think about this approach as fitting a straight line through the data which minimises the distances or errors from datapoints to line. Note that linear regression can be used with both continuous or categorical features as well as with both continuous or categorical targets.

> Be careful with how you prepare and encode your categorical features. If one feature describes distinct nominal categories, e.g., [`cat`, `dog`, `frog`], it may make more sense to one-hot encode rather than integer encode. I.e., `cat` $\to [1,0,0]$, `dog` $\to [0,1,0]$ and `frog` $\to [0,0,1]$, rather than integer encoding `cat`$\to [0]$, `dog`$\to [1]$ and `frog`$\to [2]$. Otherwise we nonsensically suggest to the model that these categories are in some way ordered, or that the additive contribution of `frog` is twice that of `dog`.
{:.prompt-danger}

>For very large datasets, it can be computationally prohibitive to directly compute the analytical solution to the normal equations. In such a setting, it may be appropriate to use an approximate iterative optimisation technique such as stochastic or minibatch gradient descent. 
{:.prompt-tip}

### **A Probabilistic Formulation**.

An alternative approach is to reason about the problem from a probabilistic perspective. In general with probabilistic regression we assume that there exists some latent or unobserved function $f$ which exactly relates inputs $\mathbf X$ and outputs $\mathbf y$. This is the true deterministic signal that we want to capture with our model. Unfortunately the observed data is noisy. In other words the observed data = true signal + noise. We consider this noise to be a probabilistic component that is an obstacle to us learning the true form of $f$. 

In the case of a linear regression, we constrain the space of possible functions to be linear as a modelling assumption. In fact, when we look at real world data, we see it rarely (if ever) follows an exactly linear relationship. The data points deviate by some margin of error from being a straight line. Again as a modelling assumption, we assume that this margin of error or noise is normally distributed with mean zero. 

In other words, under this framework we suppose that each $\ y_n = \beta \mathbf x_n + \varepsilon_n, \quad \varepsilon_n \sim \mathcal N(0,\sigma^2). \ $ That is, we hypothesise that the observed target values $y_n$ lie along the hyperplane $\beta \mathbf{X}$, but deviate slightly from the hyperplane by some margin of error $\varepsilon_n$. 

This can be expressed in notation as $y_n \mid \beta, \sigma^2, \mathbf x_n \sim \mathcal N(\beta \ \mathbf x_n, \sigma^2).\ $ That is, the conditional distribution of the target values $y_n$ is normally distributed around the mean $\beta \ \mathbf{x}_n$ with variance $\sigma^2$.

>Pay attention to the distinction regarding what we prescribe to be normally distributed. We do not require that the marginal distribution of the targets or features be Gaussian. That is, we do not require $\mathbf y \sim \mathcal N(\cdot, \cdot),$ nor do we require $\mathbf X \sim \mathcal N(\cdot, \cdot)$. Rather we specify that the conditional distribution $\mathbf y \mid \mathbf X \sim \mathcal N(\cdot, \cdot)$ of targets given features is Gaussian. 
{:.prompt-warning}

A key asssumption here is that of **homoscedasticity**. In other words we assume a fixed or constant variance $\sigma^2$ which does not depend on the value of the input. The amount of noise variance is constant regardless of which region of input space we are considering. 

Although this variance term $\sigma^2$ does not vary conditional on the inputs $\mathbf X$, it does depends on the observed outputs $\mathbf y$. Therefore we consider it a parameter to be learned during optimisation. 

We use maximum likelihood estimation to determine point estimates for the most likely values of the coefficient parameters $\beta$ and noise variances $\sigma^2$ conditioned on the observed data $\mathcal D = \{(\mathbf x_1,y_1), ...,(\mathbf x_n,y_n)\}.$

What we will see is that maximum likelihood estimation and mean squared error minimisation are equivalent in the case of Gaussian likelihood terms. To show this, we first recall that the probability density function for a Gaussian distribution on some variable $y$ with mean $\mu$ and variance $\sigma^2$ is:

$$
\text{Normal PDF}: \quad p(y \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right).
$$

Suppose we have a dataset $\mathcal D$ of $N$ observations $\{(x_n, y_n)\}_{n=1}^N$, where we assume that the relationship between $x_n$ and $y_n$ is linear with additive Gaussian noise, i.e. $y_n = \beta x_n + \epsilon_n$, where $\epsilon_n \sim \mathcal{N}(0, \sigma^2)$. We assume each observation in the dataset is independent. Therefore the likelihood function is the product of the independent densities $y_n \mid \beta, \sigma^2, \mathbf x_n \sim \mathcal N(\beta \mathbf x_n, \sigma^2)$,

$$
\begin{align*}
p(y_1, \ldots, y_N \mid \mathbf x_1, \ldots, \mathbf x_N, \beta, \sigma^2) &= \prod_{n=1}^N p(y_n \mid \mathbf x_n, \beta_0, \beta_1, \sigma^2) \\[.5em]
&= \prod_{n=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_n - \beta \mathbf x_n)^2}{2\sigma^2}\right)
\end{align*}
$$

Taking the logarithm of the likelihood, we obtain the log-likelihood,

$$
\begin{align*}
& = \log \left\{\prod_{n=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_n - \beta \mathbf x_n)^2}{2\sigma^2}\right)\right\}

\\[1em]

& = \sum_{n=1}^N \log \left\{
\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(
-\frac{(y_n - \beta \mathbf x_n)^2}{2\sigma^2}
\right)\right\}

\\[1em]

&= \sum_{n=1}^N \left\{
\log \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)+
\log\left(\exp\left(-\frac{(y_n - \beta \mathbf x_n)^2}{2\sigma^2}\right)\right)
\right\} 

\\[1em]

&= \sum_{n=1}^N \left\{
-\log \left(\sqrt{2\pi\sigma^2}\right)-
\left(\frac{(y_n - \beta \mathbf x_n)^2}{2\sigma^2}\right)
\right\} 

\\[1em]

&= -\sum_{n=1}^N \left\{
\log \left(\sqrt{2\pi\sigma^2}\right)+
\left(\frac{(y_n - \beta \mathbf x_n)^2}{2\sigma^2}\right)
\right\} 

\end{align*}
$$

Taking the negative, we obtain the negative log-likelihood, $\mathcal L,$ 

$$
\mathcal{L}(\ \beta ; \ \mathcal D \ ) = \sum_{n=1}^N \left\{
\log \left(\sqrt{2\pi\sigma^2}\right)+
\left(\frac{(y_n - \beta \mathbf x_n)^2}{2\sigma^2}\right)
\right\} 
$$

The MLE estimates of the linear coefficients $\beta = [\beta_0, \beta_1, ..., \beta_D]$ and noise variance $\hat \sigma^2$ are found by minimizing the negative log-likelihood function:

$$
\hat{\beta}, \hat{\sigma}^2 = \arg\min_{\beta, \sigma^2} \mathcal{L}(\beta, \sigma^2; \ \mathcal D)
\\[-.5em]
$$

We see that the first log term in the summation does not depend on $\beta_0, \beta_1$ and so these terms vanish in the derivatives $\frac{\partial\mathcal{L}}{\partial \beta_0}$ and $\frac{\partial\mathcal{L}}{\partial \beta_1}.$ Likewise the constant coefficient $\sigma^{-2}$ does not influence the optimising parameters. We can see then that MLE is equivalent to minimizing the MSE between the predicted values $\hat{y}_n = \beta \mathbf x_n$ and the observed values $y_n$,

$$
\begin{align*}
\arg \min_{\beta, \sigma^2} \ \mathcal{L}(\beta; \mathcal D) 
&= 
\arg \min_{\beta, \sigma^2} \ \sum_{n=1}^N \left\{
\log \left(\sqrt{2\pi\sigma^2}\right)+
\left(\frac{(y_n - \beta \mathbf x_n)^2}{2\sigma^2}\right)
\right\} 
\\[1.5em] 
&=
\arg \min_{\beta, \sigma^2} \ \sum_{n=1}^N \left\{ \frac{(y_n - \beta \mathbf x_n)^2}{2\sigma^2}
\right\} 
\\[1.5em]
&=
\arg \min_{\beta, \sigma^2} \ \left\{ \frac{1}{2\sigma^2} \sum_{n=1}^N  (y_n - \beta \mathbf x_n)^2 \right\}.
\end{align*}
$$

We see that the loss function reduces down to a quantity proportional to the mean squared error formulation from the previous section. From here we need simply determine the particular values of the noise variance $\sigma^2$ and linear coefficients $\beta$ which minimise this negative log likelihood $\mathcal L$, subject to the constraint that $\sigma^2$ > 0. 

The process of learning or determining optimal model parameters is often called *training*. In this section we use maximum likelihood estimation to determine the optimal value for the linear coefficients $\beta$ and noise variance $\sigma^2$.  Intuitively we might think of this approach as considering the most likely or plausible values for the parameters, under the assumption of Gaussian likelihoods. 


## **Inference**

The term inference does a lot of heavy lifting in statistics. In this case we use the term inference to describe the learning of model parameters, as well as the prediction of target values at unseen test inputs. 

After training our model we can condition on a particular input value to obtain a point estimate for the output value. We simply compute the product of the test input $\hat{\mathbf{x}}$ and the learned coefficient vector $\hat{\beta},$ so that $\hat{y} = \hat{\beta} \hat{\mathbf{x}}$.

>Recall the five basic assumptions of linear regression.
>1. linearity, the inputs and outputs are linearly related.
>2. independence, observation are sampled independently.
>3. homoscedasticity, assuming constant noise variance,
>4. residuals, the model errors have mean zero and are uncorrelated. 
>5. normality, our additive noise is Gaussian distributed. 
{:.prompt-info}


### **Uncertainty Quantification**

The other delightful benefit of probabilistic modelling is in the capacity for quantifying the uncertainty of our estimates. In this case, we can construct confidence intervals around our point estimate to quantify uncertainty. What do the confidence intervals describe? 

Well we are grappling with random variables rather than deterministic formulas. Confidence intervals capture the variation in the sampling process. For a particular sampled person with height $X=x$, we estimate that the person will have weight $y = mx + c$, where $m$ and $c$ are our OLS parameter estimates. However $y$ is Gaussian with mean $mx+c$ and variance $\sigma^2$. You could have a short heavy person or a tall light person. Depending on who we get in the sample, there is some variation. 

Suppose we construct a $95%$ CI (equivalent to +/- 2 sigma from the mean). We would expect on average that if we sampled 100 people at weight $X=x$, then $95%$ of them would be inside the interval and $5%$ would be outside, assuming our model is true.  


## **Linear Regression in Python**

In Python, linear regression can be easily implemented using libraries such as `scikit-learn`. The `LinearRegression` class in `scikit-learn` provides a simple interface for fitting a linear model to data. Here's an example of how to use it:

```python
from sklearn.linear_model import LinearRegression

# Assuming X and y are already defined as the features and target
regressor = LinearRegression()
regressor.fit(X, y)

# The coefficients can be accessed
beta_hat = regressor.coef_
c_hat = regressor.intercept_
```


## **Evaluating Goodness of Fit**

In supervised learning, determining how well a model fits the data is a crucial step. Various statistical measures and tests are employed to assess this fit, each providing insights into different aspects of the model's performance.

#### R-squared (Coefficient of Determination)

The R-squared, or the coefficient of determination, is one of the most commonly used metrics in regression analysis. It represents the proportion of the variance in the dependent variable that is predictable from the independent variables. R-squared values range from 0 to 1, where a higher value indicates a better fit.

The R-squared is calculated as:

$$\begin{align*} R^2 := 1 − \frac{\text{Sum of Squares of Residuals (SSR)}}{\text{Total Sum of Squares (SST)}}, \end{align*}$$

where SSR is the sum of the squared differences between the observed values and the predicted values, and SST is the total variance in the observed data.

#### Adjusted R-squared

While R-squared is a useful indicator, it has a limitation: it tends to increase with the addition of more predictors, regardless of their relevance. The Adjusted R-squared addresses this by adjusting for the number of predictors in the model:

$$\text{Adjusted } R^2 := 1 - \frac{(1-R^2)(N-1)}{(N-P-1)},$$

where $P \in \mathbb N$ is the number of predictors or features. This adjustment makes it a more reliable metric, especially for models with a large number of predictors.


#### Root Mean Sqaured Error

RMSE is a commonly used measure that quantifies the difference between the predicted values and the observed values. It's particularly useful because it is in the same units as the response variable. Lower values of RMSE indicate a better fit.

$$\text{RMSE}(\mathbf y, \mathbf{\hat{y}}) := \sqrt{\frac{\sum_{n=1}^N (y_n - \hat{y_n})}{N}}.$$


#### Residual Analysis. 

The discrepancies or errors between what our models predicts and what was truly observed are known as residuals. Our aim was to capture the signal within the data, under the framework that signal = data + noise. Therefore we would hope that the model residuals appear like random Gaussian noise, with no discernible trend or pattern. If there is a some clear pattern or structure visible in the residuals, this would suggest that our model has failed to capture some informative aspect or signal within the data. 

Likewise, we would expect to observe that the variance of these residuals is similar across the entire input space. This would be in line with our assumption of homoscedasticity. If there is some structure within the residuals, such that the variance appears to depend on the mean, then the standard linear regression may not necessarily the most appropriate model for the data. We may instead consider a generalised linear model.  


## **Basis Function Expansion**

In the realm of linear regression, the concept of basis function expansion plays a pivotal role in enhancing the model's flexibility and capability to capture complex relationships in data. While linear regression is powerful in its simplicity, its straightforward approach of modeling relationships directly as linear combinations of input variables can be limiting when dealing with non-linear patterns. 

The core idea of this section can be summarised in a single sentence, "*linear models are linear in the parameters, not in the features*". But what does this mean? 

We can engineer our features using basis function expansion to better captrue nonlinearities in the data. Basis functions are a set of functions that are used to transform the original input variables into a new set of features. Essentially, they are a means of preprocessing the input data to enable the linear model to capture more complex relationships. Common choices for basis functions include polynomial functions and trigonometric functions.

By applying these basis functions to the input variables, we transform the input space into a new feature space where linear relationships are more likely to exist or are more pronounced. For example, applying a quadratic basis function to an input feautre $x$ would transform it into $x^2$, enabling the linear model to capture quadratic relationships in the original data.

Let $\phi(\cdot)$ denote some basis function. For an input $\mathbf x_n$, the transformed feature vector $\phi(\mathbf x_n) = [\phi(x_{n1}), \ldots, \phi(x_{nD})].$ The linear regression model then becomes,
$$\hat y_n = \beta^T \phi(\mathbf x_n).$$

Here the vector of coefficients $\beta$ acts upon the transformed feature space. The design matrix $\mathbf X$ in the original formulation is replaced by a transformed design matrix $\Phi$, where each row is the transformed feature vector of the corresponding observation. 





<!-- 
```python
Code Block
```
 -->


## *Further Reading*

- https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1308&context=pare 
- https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote08.html
- https://www.youtube.com/watch?v=YIfANKRmEu4&t=6s