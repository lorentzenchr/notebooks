**DISCUSSION**

- [ ] Give motivating examples for non Normal real world regression tasks
  e.g. Poisson for count processes
  - [ ] **DECISSION: Use diamonds dataset** with target = diamond price
    This is roughly Gamma distributed. Poisson is more difficult to show empirically, espcecially for large means, because is is discrete and one would need an overdisperson parameter which is not possible with the Poisson distribution alone. Without it, distribution becomes very narrow for large mean.
    *rth: I was thinking a dataset from everyday life that's easy to understand.*
    1. *I have started looking into a [bike sharing dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) before realizing that's the dataset also used as illustration for the [interpretabel ML book](https://christophm.github.io/interpretable-ml-book/bike-data.html) and it was also uploaded on Kaggle so there is pleinty of material online on it, which is not so great. Though it's pretty good in terms of [count distribution](https://github.com/pgebert/bike-sharing-dataset#remove-outliers-from-data). Looking for for alternatives.*
    2. *There is also the dataset of [bike renting in Paris](https://github.com/lovasoa/historique-velib-opendata) it requires a bit more pre-processing and I'm less convinced about applicability of the PoissonRegressor to the target distribution below. I can add weather information there, the dataset would be essentially the number of rented bikes as a function of hour/date and weather. Though this dataset is for 2020 so it would also include the lockdown period wich complicates things.*
    ![](https://i.imgur.com/rfrRPTu.png) 
- [ ] Mention scikit-learn roadmap?

Regression models for continuous targets are most often fit with a squared error which can be derived from the likelihood function of a Normal distribution. Squared error makes two important assumptions:
1. The target is distributed symmetrically around the expectation.
2. The variance of the target does not depend on the expectation.
It is known to be sensitive to outliers.

**NOTES:**
- *cl: So far, too mathematical/statistical, I know. But can help to cut out the right message in the end.*
- *rth: One thing that bothers me is that although the 'normal distribution' assumption is rarely verified perfectly, in a lot of cases using a model with MSE still works reasonably well as a start. Maybe we could word it as: using GLMs there would lead to more accurate modeling, though since we compare with different metrics, the comparison is not too straightforward either.*
- *cl:*
  - *For low frequencies (small lambda parameter) and little data, Poisson is clearly better suited than Normal.*
  - *How to enforce prediction>=0 with MSE? If you use a log link, you get biased results (same as Gamma with log link and MSE with logit link)*
  - *Convergence to "true" parameters is faster if you're closer to the "true" distribution, i.e. the more realistic variance assumption (hence the "little data" argument above)*

[comment]: <> (End of discussion)
---


# Generalized Linear Models arrived in scikit-learn 0.23

###### tags: `scikit-learn, GLM`

### TLDR

While scikit-learn already had some Generalized Linear Models (GLM) implemented, e.g. [LogisticRegression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), other losses than mean squared error and log-loss were missing.
As the world is almost (surely) never Normal distributed, regression tasks might benefit a lot from the new [PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html#sklearn.linear_model.PoissonRegressor), [GammaRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html#sklearn.linear_model.GammaRegressor) and [TweedieRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html#sklearn.linear_model.TweedieRegressor) estimators, added in scikit-learn 0.23. Read more below and in the [User Guide](https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-regression).🚀 

**Table of Content**
[ToC]


## 1 The world is not Normal distributed

Like *real life*, real world data are most often far from *normality*. Their distribution tend to a [normal or Gaussian distribution](https://en.wikipedia.org/wiki/Normal_distribution) 

Other instances of data, that is not Normal distributed, are counts (discrete) or frequencies (counts per some unit). A few examples that come to mind are:
- number of clicks per second in a Geiger counter
- number of patients per day in a hospital
- number of persons per day using their bike
- number of goals scored per game and player
- number of smiles per day and person😃 *cl: Would LOVE to hove those data!!!*

In what follows, we have chosen the [diamonds dataset](https://ggplot2.tidyverse.org/reference/diamonds.html) to show the non-normality and the convenience of GLMs in modelling such targets.

The diamonds dataset consists of prices of over 50 000 round cut diamonds with a few explaining variables, also called features $X$, such as carat, color, cut quality, clarity and so forth.
We start with a plot of the cumulative distribution function (CDF) of the target variable $Y=\textrm{price}$ and compare to a fitted [Normal](https://en.wikipedia.org/wiki/Normal_distribution) and [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution) (two parameters each).

TODO: Insert first plot of https://github.com/lorentzenchr/notebooks/blob/master/Blog_GLM_in_sklearn_v0.23.ipynb

These plots show clearly that the Gamma distribution might be a better fit to the marginal distribution of $Y$ than the Normal distribution.

The simplest distribution for those is the [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution).


## 2 Introduction to GLMs

GLMs are statistical models for regression tasks that aim to estimate and predict the conditional expectation of $Y$, $E[Y|X]$.
They unify many different target types under one framework: [Ordinary Least Squares](https://en.wikipedia.org/wiki/Ordinary_least_squares), [Logistic](https://en.wikipedia.org/wiki/Logistic_regression), [Probit](https://en.wikipedia.org/wiki/Probit_model) and [multinomial model](https://en.wikipedia.org/wiki/Multinomial_logistic_regression), [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression), Gamma and many more. GLMs were invented by John Nelder and Robert Wedderburn in 1972, long after artificial neural networks😉

The basic assumptions for instance or data row $i$ are
$$
E[Y_i|x_i] = \mu_i= h(x_i^T\beta)\,,
\\
Var[Y_i|x_i]] \sim \frac{v(\mu_i)}{w_i}\,.
$$
*cl: The following is maybe too much*<br>
Where, one needs to specify:
- a target $Y_i$,
- an inverse link function $h$, which maps real numbers to the range of $Y$,
- optionally, sample weights $w_i$,
- and a variance function $v(\mu)$, which is equivalent to specifying a loss function or a specific distribution from the family of the [exponential dispersion model](https://en.wikipedia.org/wiki/Exponential_dispersion_model) (EDM).
- a feature matrix $X$ with row vectors $x_i$,

Note that the choice of the loss or distribution function or, equivalently, a variance function is crucial. It should, at least, reflect the domain of $Y$. Some typical combinations of domain, loss and link function are:
- real numbers, Normal distribution, identity link
- positive numbers: Gamma distribution, log link
- non-negative: Poisson distribution (works for integers as well as continuous targets), log link
- interval $[0, 1]$: Binomial distribution, logit link

Once you have chosen the first four points, what remains is to find a good feature matrix $X$. Similarly to classical linear models, and unlike other machine learning algorithms such as boosted trees, there are very few hyperparemeters to tune. A typical hyperparemeter is regularization strength. Therefore the big leverage to improve your model is manual feature engineering of $X$.

#### Strengths
- Very well understood and established, proven over and over in practice, e.g. stability, see next point.
- Very stable: slight changes of training data do not alter the fitted model much (counter example: decision trees).
- Versatile as to model different targets with different link and loss functions.
- Mathematical tractable => good theoretical understanding and fast fitting even for large datasets.
- Ease of interpretation.
- As flexible as the building of the feature matrix $X$.
- Some losses, like Poisson loss, can handle a certain amount of excess of zeros.

#### Weaknesses
- Feature matrix $X$ has to be build manually, in particular for interactions and non-linear effects.
- Unbiasedness depends on (correct) specification of $X$ and of combination of link and loss function.
- Predictive performance often worse than for boosted tree models or neural networks.

#### Current Minimal Implementation in Scikit-Learn
The new GLM regressors are available as
```python
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.linear_model import TweedieRegressor
```
The `TweedieRegressor` has a parameter `power`, which corresponds to the exponent of the variance function $v(\mu) \sim \mu^p$. For ease of the most common use, `PoissonRegressor` and `GammaRegressor` are the same as `TweedieRegressor(power=1)` and `TweedieRegressor(power=2)`, respectively.
All of them also support an L2-penalty on the coefficients by setting the penalization strength `alpha`.
The underlying optimization problem is solved via the [l-bfgs solver of scipy](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb).

Please note that the release 0.23 also introduced the Poisson loss for the histogram gradient boosting regressor as `HistGradientBoostingRegressor(loss='poisson')`.

## 3 Gamma GLM for Diamonds

Although, in the first section, we were analysing the marginal distribution of $Y$ and not the conditional (on the features $X$) distribution, we take away the suggestion to fit a Gamma GLM with log-link, i.e. $h(x) = \exp(x)$. Furthermore, we split the data textbook-like into 80% training set and 20% test set and use the ColumnTransformer to handle columns differently. Our feature engineering consists of selecting only the four columns `"carat"`, `"clarity"`, `"color"` and `"cut"`, log-transforming `"carat"` as well as one-hot-encoding the other three.

TODO: Show some plots and figures.

Note: Fitting OLS on log(prices) works also quite well. This is to be expected, as Log-Normal and Gamma are very similar distributions, both with $v(\mu) \sim \mu^2$.

## 4 Outlook

There are several open issues and pull request for improving GLMs and fitting of non-normal data. Let's hope that we'll see some of them in the near future:

- Poisson splitting criterion for decision trees [PR #17386](https://github.com/scikit-learn/scikit-learn/pull/17386) in v0.24
- Spline Transformer [PR #18368](https://github.com/scikit-learn/scikit-learn/pull/18368) 
- L1 penalty and coordinate descent solver [Issue #16637](https://github.com/scikit-learn/scikit-learn/issues/16637)
- [IRLS](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares) solver if bencharking shows improvement over l-bfgs [Issue #16634](https://github.com/scikit-learn/scikit-learn/issues/16634)
- Better handling of categorical data
  - Better support for interaction terms [Issue 15263](https://github.com/scikit-learn/scikit-learn/issues/15263)
- Feature names [SLEP015](https://github.com/scikit-learn/enhancement_proposals/pull/48)🎉


By Christian Lorentzen
