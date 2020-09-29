# Generalized Linear Models arrived in scikit-learn 0.23

###### tags: `scikit-learn, GLM`

**NOTES:**
- *cl: So far, too mathematical/statistical, I know. But can help to cut out the right message in the end.*
- *rth: One thing that bothers me is that although the 'normal distribution' assumption is rarely verified perfectly, in a lot of cases using a model with MSE still works reasonably well as a start. Maybe we could word it as: using GLMs there would lead to more accurate modeling, though since we compare with different metrics, the comparison is not too straightforward either.*
- *cl:*
  - *For low frequencies (small lambda parameter) and little data, Poisson is clearly better suited than Normal.*
  - *How to enforce prediction>=0 with MSE? If you use a log link, you get biased results (same as Gamma with log link and MSE with logit link)*
  - *Convergence to "true" parameters is faster if you're closer to the "true" distribution, i.e. the more realistic variance assumption (hence the "little data" argument above)*


### TLDR

While scikit-learn already had some Generalized Linear Models (GLM) implemented, e.g. [LogisticRegression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), other losses than mean squared error and log-loss were missing. As the world is almost (surely) never Normal distributed, regression tasks might profit a lot from the new [PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html#sklearn.linear_model.PoissonRegressor), [GammaRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html#sklearn.linear_model.GammaRegressor) and [TweedieRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html#sklearn.linear_model.TweedieRegressor) estimators, added in scikit-learn 0.23. Read more below and in the [User Guide](https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-regression).:rocket: 

**Table of Content**
[ToC]


## 1 The world is not Normal distributed

---
**DISCUSSION**

- [ ] Give motivating examples for non Normal real world regression tasks
  e.g. Poisson for count processes
  - [ ] plot of some real world targets (diamond prices, car accident claim count, california houses)?
    *rth: I was thinking a dataset from everyday life that's easy to understand.*
    1. *I have started looking into a [bike sharing dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) before realizing that's the dataset also used as illustration for the [interpretabel ML book](https://christophm.github.io/interpretable-ml-book/bike-data.html) and it was also uploaded on Kaggle so there is pleinty of material online on it, which is not so great. Though it's pretty good in terms of [count distribution](https://github.com/pgebert/bike-sharing-dataset#remove-outliers-from-data). Looking for for alternatives.*
    2. *There is also the dataset of [bike renting in Paris](https://github.com/lovasoa/historique-velib-opendata) it requires a bit more pre-processing and I'm less convinced about applicability of the PoissonRegressor to the target distribution below. I can add weather information there, the dataset would be essentially the number of rented bikes as a function of hour/date and weather. Though this dataset is for 2020 so it would also include the lockdown period wich complicates things.*
    ![](https://i.imgur.com/rfrRPTu.png)
    *cl: Why not this one? Can I help with preprocessing?*
  - *cl: Maybe, it is easier to plot empirical distributions vs Normal and Gamma than vs Poisson. Reason: Poisson is discrete and you can't plot overdispersed Poisson (complicated) which likely would be necessary.* 

- [ ] Range of $Y$
- [ ] Distribution/Skewness/Havy tailedness of $Y$
- [ ] Mention scikit-learn roadmap?

Regression models for continuous targets are most often fit with a squared error which can be derived from the likelihood function of a Normal distribution. Squared error makes two important assumptions:
1. The target is distributed symmetrically around the expectation.
2. The variance of the target does not depend on the expectation.
It is known to be sensitive to outliers.

[comment]: <> (End of discussion)
---

Like *real life*, real world data is most often far from *normality*. Below, we've chosen some public datasets and plotted the empirical cumulative distribution function (CDF) as well as the CDFs of fitted Normal/Gaussian and Gamma distribution.

TODO: Plots, see https://github.com/lorentzenchr/notebooks/blob/master/real_world_non_normal.ipynb

Other examples of data, that is not Normal distributed are counts or frequencies (counts per some unit). Examples of count data are:
- number of clicks per second in a Geiger counter
- number of patients per day in hospital
- number of persons per day using their bike
- number of births per year in a population
- number of goals scored per game and player
- number of smiles per day and person *cl: Would LOVE to hove those data!!!*



## 2 Introduction to GLMs

GLMs are statistical models for regression tasks for the expectation $E[Y]$ that unify many different target types $Y$ under one framework (OLS, Logit/Probit, Multinomial, Poisson, Gamma and many more). It was invented by John Nelder and Robert Wedderburn in 1972, long after neural networks :smirk:

The basic assumptions are
$$
E[Y_i] = \mu_i= h(x_i^T\beta)
\\
Var[Y_i] \sim \frac{v(\mu_i)}{w_i}
$$
*The following is maybe too much*
Therefore, one needs to specify:
- a target $Y_i$,
- an inverse link function $h$, which maps the reals numbers to the range of $Y$,
- sample weights $w_i$,
- feature vectors $x_i$ forming the feature matrix $X$,
- and a variance function $v$, which is equivalent to specifying a loss function or a specific distribution.

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