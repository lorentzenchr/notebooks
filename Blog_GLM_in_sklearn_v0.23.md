# Generalized Linear Models arrived in scikit-learn 0.23

###### tags: `scikit-learn, GLM`

### TLDR

While scikit-learn already had some Generalized Linear Models (GLM) implemented, e.g. [LogisticRegression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), other losses than mean squared error and log-loss were missing.
As the world is almost (surely) never normally distributed, regression tasks might benefit a lot from the new [PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html#sklearn.linear_model.PoissonRegressor), [GammaRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html#sklearn.linear_model.GammaRegressor) and [TweedieRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html#sklearn.linear_model.TweedieRegressor) estimators, added in scikit-learn 0.23. Read more below and in the [User Guide](https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-regression).🚀 

**Table of Content**
[ToC]


## 1 The world is not normally distributed

Like *real life*, real world data is most often far from *normality*.
Still, data is often assumed, sometimes implicitly, to follow a [Normal (or Gaussian) distribution](https://en.wikipedia.org/wiki/Normal_distribution).
Maybe, the two most important assumptions made when choosing a Normal distribution or squared error for regression tasks are<sup>1</sup>:

1. The data is distributed symmetrically around the expectation.
   Hence, expectation and median are the same.
2. The variance of the data does not depend on the expectation.

On top, it is well known to be sensitive to outliers.
Here, we want to point out that&mdash;potentially better&mdash; alternatives are available.

Typical instances of data that is not normally distributed are counts (discrete) or frequencies (counts per some unit).
For these, the simple [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution) might be much better suited. 
A few examples that come to mind are:

- number of clicks per second in a Geiger counter
- number of patients per day in a hospital
- number of persons per day using their bike
- number of goals scored per game and player
- number of smiles per day and person😃 *cl: Would LOVE to hove those data!!!* *Think about make their distribution more normal*

In what follows, we have chosen the [diamonds dataset](https://ggplot2.tidyverse.org/reference/diamonds.html) to show the non-normality and the convenience of GLMs in modelling such targets.

The diamonds dataset consists of prices of over 50 000 round cut diamonds with a few explaining variables, also called features $X$, such as carat, color, cut quality, clarity and so forth.
We start with a plot of the (marginal) cumulative distribution function (CDF) of the target variable $Y=\textrm{price}$ and compare to a fitted Normal and [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution) which have both two parameters each.

TODO: Insert second plot (empricial CDFs) of https://github.com/lorentzenchr/notebooks/blob/master/Blog_GLM_in_sklearn_v0.23.ipynb

These plots show clearly that the Gamma distribution might be a better fit to the marginal distribution of $Y$ than the Normal distribution.

After a more theoretical intermezzo in Chapter 2, we will resume to the diamonds dataset in Chapter 3.


## 2 Generalized Linear Models

### 2.1 GLMs in a Nutshell

GLMs are statistical models for regression tasks that aim to estimate and predict the conditional expectation of a target variable $Y$, i.e. $E[Y|X]$.
They unify many different target types under one framework: [Ordinary Least Squares](https://en.wikipedia.org/wiki/Ordinary_least_squares), [Logistic](https://en.wikipedia.org/wiki/Logistic_regression), [Probit](https://en.wikipedia.org/wiki/Probit_model) and [multinomial model](https://en.wikipedia.org/wiki/Multinomial_logistic_regression), [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression), Gamma and many more.
GLMs were formalized by [John Nelder](https://en.wikipedia.org/wiki/John_Nelder) and
[Robert Wedderburn](https://en.wikipedia.org/wiki/Robert_Wedderburn_(statistician)) in 1972, long after artificial neural networks😉

The basic assumptions for instance or data row $i$ are
$$
E[Y_i|x_i] = \mu_i= h(x_i \cdot \beta)\,,
\\
Var[Y_i|x_i]] \sim \frac{v(\mu_i)}{w_i}\,.
$$
One needs to specify:
1. the target variable $Y_i$,
2. the inverse link function $h$, which maps real numbers to the range of $Y$ (or better the range of $E[Y]$),
3. optionally, sample weights $w_i$,
4. the variance function $v(\mu)$, which is equivalent to specifying a loss function or a specific distribution from the family of the [exponential dispersion model](https://en.wikipedia.org/wiki/Exponential_dispersion_model) (EDM),
5. and the feature matrix $X$ with row vectors $x_i$.

Note that the choice of the loss or distribution function or, equivalently, the variance function is crucial. It should, at least, reflect the domain of $Y$.
Some typical combinations of domain, loss and link function are:

| target domain     | distribution | link     | example
| ----------------- | ------------ | -------- | ---------------------- |
| real numbers      | Normal       | identity | measurement error      |
| positive numbers  | Gamma        | log      | insurance claim size   |
| non-negative      | Poisson      | log      | see examples above     |
| interval $[0, 1]$ | Binomial     | logit    | probability of success |

TODO: Insert first plot of https://github.com/lorentzenchr/notebooks/blob/master/Blog_GLM_in_sklearn_v0.23.ipynb

Once you have chosen the first four points, what remains is to find a good feature matrix $X$.
Unlike other machine learning algorithms such as boosted trees, there are very few hyperparemeters to tune.
A typical hyperparemeter is the regularization strength when penalties are applied.
Therefore, the biggest leverage to improve your GLM is manual feature engineering of $X$.
This includes among others feature selection, encoding schemes for categorical features, interaction terms, non-linear terms like $x^2$.

### 2.2 Strengths
- Very well understood and established, proven over and over in practice, e.g. stability, see next point.
- Very stable: slight changes of training data do not alter the fitted model much (counter example: decision trees).
- Versatile as to model different targets with different link and loss functions.
  - Example: Log link gives a multiplicative structure: effects are interpreted on a relative scale.
    Together with a Poisson distribution, this still works even when some target values are exactly zero.
- Mathematical tractable => good theoretical understanding and fast fitting even for large datasets.
- Ease of interpretation.
- As flexible as the building of the feature matrix $X$.
- Some losses, like Poisson loss, can handle a certain amount of excess of zeros.

### 2.3 Weaknesses
- Feature matrix $X$ has to be build manually, in particular interaction terms and non-linear effects.
- Unbiasedness depends on (correct) specification of $X$ and of combination of link and loss function.
- Predictive performance often worse than for boosted tree models or neural networks.

### 2.4 Current Minimal Implementation in Scikit-Learn
The new GLM regressors are available as
```python
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.linear_model import TweedieRegressor
```
The `TweedieRegressor` has a parameter `power`, which corresponds to the exponent of the variance function $v(\mu) \sim \mu^p$. For ease of the most common use, `PoissonRegressor` and `GammaRegressor` are the same as `TweedieRegressor(power=1)` and `TweedieRegressor(power=2)`, respectively, with built-in log link.
All of them also support an L2-penalty on the coefficients by setting the penalization strength `alpha`.
The underlying optimization problem is solved via the [l-bfgs solver of scipy](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb).

Please note that the release 0.23 also introduced the Poisson loss for the histogram gradient boosting regressor as `HistGradientBoostingRegressor(loss='poisson')`.

## 3 Gamma GLM for Diamonds

After all that theory, we come back to our real world dataset: diamonds.
Although, in the first section, we were analysing the marginal distribution of $Y$ and not the conditional (on the features $X$) distribution, we take the plot as a hint to fit a Gamma GLM with log-link, i.e. $h(x) = \exp(x)$. Furthermore, we split the data textbook-like into 80% training set and 20% test set<sup>2</sup> and use the ColumnTransformer to handle columns differently.
Our feature engineering consists of selecting only the four columns `"carat"`, `"clarity"`, `"color"` and `"cut"`, log-transforming `"carat"` as well as one-hot-encoding the other three.

TODO: Show some plots and figures.

Note: Fitting OLS on log(prices) works also quite well. This is to be expected, as Log-Normal and Gamma are very similar distributions, both with $Var[Y] \sim E[Y]^2 = \mu^2$.

## 4 Outlook

There are several open issues and pull request for improving GLMs and fitting of non-normal data. Let's hope that we'll see some of them in the near future:

- Poisson splitting criterion for decision trees [PR #17386](https://github.com/scikit-learn/scikit-learn/pull/17386) in v0.24
- Spline Transformer [PR #18368](https://github.com/scikit-learn/scikit-learn/pull/18368) 
- L1 penalty and coordinate descent solver [Issue #16637](https://github.com/scikit-learn/scikit-learn/issues/16637)
- [IRLS](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares) solver if bencharking shows improvement over l-bfgs [Issue #16634](https://github.com/scikit-learn/scikit-learn/issues/16634)
- Better handling of categorical data
  - Better support for interaction terms [Issue #15263](https://github.com/scikit-learn/scikit-learn/issues/15263)
  - Native categorical support [Issue #18893](https://github.com/scikit-learn/scikit-learn/issues/18893)
- Feature names [SLEP015](https://github.com/scikit-learn/enhancement_proposals/pull/48)🎉


By Christian Lorentzen


### Footnotes

<sup>1</sup> Algorithms and estimation methods are often well able to deal with some deviation from the Normal distribution. In addition, the [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem) justifies a Normal distribution when considering averages or means, and the [Gauss–Markov theorem](https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem) is a cornerstone for usage of least squares with linear estimators (linear in the target $Y$).

<sup>2</sup> Rows in the diamonds dataset seem to be highly correlated as there are many rows with the same values for carat, cut, color, clarity and price, while the values for x, y and z seem to be permuted.
Therefore, we define the new group variable that is unique for carat, cut, color, clarity and price.
Then, we split stratified by group, i.e. using a `GroupShuffleSplit`.<br>
Having correlated train and test sets invalidates the i.i.d assumption and may render test scores too optimistical.
