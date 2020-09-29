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
As the world is almost (surely) never Normal distributed, regression tasks might profit a lot from the new [PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html#sklearn.linear_model.PoissonRegressor), [GammaRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html#sklearn.linear_model.GammaRegressor) and [TweedieRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html#sklearn.linear_model.TweedieRegressor) estimators, added in scikit-learn 0.23. Read more below and in the [User Guide](https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-regression).🚀 

**Table of Content**
[ToC]


## 1 The world is not Normal distributed

Like *real life*, real world data is most often far from *normality*.
In what follows, we have chosen the [diamonds dataset](https://ggplot2.tidyverse.org/reference/diamonds.html) to show the non-normality and the convenience of GLMs in modelling such targets.

The diamonds dataset consists of prices of over 50.000 round cut diamonds with a few explaining variables, also called features $X$, such as carat, color, cut quality, clarity and so forth.
We start with a plot of the cumulative distribution function (CDF) of the target variable $Y=\textrm{price}$ and compare to a fitted [Normal](https://en.wikipedia.org/wiki/Normal_distribution) and [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution) (two parameters each).

TODO: Insert first plot of https://github.com/lorentzenchr/notebooks/blob/master/Blog_GLM_in_sklearn_v0.23.ipynb

These plots show clearly that the Gamma distribution might be a better fit to the marginal distribution of $Y$ than the Normal distribution.

Other instances of data, that is not Normal distributed, are counts (discrete) or frequencies (counts per some unit). A few examples that come up to mind are:
- number of clicks per second in a Geiger counter
- number of patients per day in a hospital
- number of persons per day using their bike
- number of goals scored per game and player
- number of smiles per day and person😃 *cl: Would LOVE to hove those data!!!*

The simplest distribution for those is the [Poisson distribbution](https://en.wikipedia.org/wiki/Poisson_distribution).


## 2 Introduction to GLMs

GLMs are statistical models for regression tasks that aim to estimate and predict the conditional expectation of $Y$, $E[Y|X]$.
They unify many different target types $Y$ under one framework: [Ordinary Least Squares](https://en.wikipedia.org/wiki/Ordinary_least_squares), [Logistic](https://en.wikipedia.org/wiki/Logistic_regression), [Probit](https://en.wikipedia.org/wiki/Probit_model) and [multinomial model](https://en.wikipedia.org/wiki/Multinomial_logistic_regression), [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression), Gamma and many more. GLMs were invented by John Nelder and Robert Wedderburn in 1972, long after artificial neural networks😉

The basic assumptions for instance or data row $i$ are
$$
E[Y_i|x_i] = \mu_i= h(x_i^T\beta)
\\
Var[Y_i|x_i]] \sim \frac{v(\mu_i)}{w_i}
$$
*cl: The following is maybe too much*
Therefore, one needs to specify:
- a target $Y_i$,
- an inverse link function $h$, which maps the real numbers to the range of $Y$,
- optionally, sample weights $w_i$,
- feature vectors $x_i$ forming the rows of the feature matrix $X$,
- and a variance function $v(\mu)$, which is equivalent to specifying a loss function or a specific distribution from the family of the [exponential dispersion model](https://en.wikipedia.org/wiki/Exponential_dispersion_model) (EDM).

Although we are analysing here the marginal distribution of $Y$ and not the conditional (on the features $X$), 

**Implementation**
- [ ] Mention something about implementation?
- Minimal implementation, losses, links, only lbfgs, only L2.

**Strengths**
Compare
- Very well understood and established, proven over and over in practice (e.g. stability).
- Versatile as to model different targets with different link and loss functions.
- Ease of interpretation.
- As flexible as the building of the feature matrix $X$.
- Mathematical tractable => fast fitting even for big data
- Certain losses can handle a certain amount of excess of zeros.

**Weaknesses**
- Feature matrix $X$ has to be build manually, in particular interactions an non-linear effects.
- Unbiasedness depends on (correct) specification of $X$ and of combination of link and loss function.
- Predictive performance often less than boosted tree models or neural networks.


## 3 A Poisson example

- [ ] According to https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html ?

## 4 Outlook

- [ ] Mention some PR explicitly?



By Christian Lorentzen and Roman Yurchak