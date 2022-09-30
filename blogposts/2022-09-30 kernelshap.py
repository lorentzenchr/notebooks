import numpy as np
import pandas as pd
from plotnine.data import diamonds
from statsmodels.formula.api import ols
from shap import KernelExplainer

# Turn categoricals into integers because, inconveniently, kernel SHAP
# requires numpy array as input
ord = ["clarity", "color", "cut"]
x = ["carat"] + ord
diamonds[ord] = diamonds[ord].apply(lambda x: x.cat.codes)
X = diamonds[x].to_numpy()

# Fit model with interactions and dummy variables
fit = ols(
  "np.log(price) ~ np.log(carat) * (C(clarity) + C(cut) + C(color))", 
  data=diamonds
).fit()

# Background data (120 rows)
bg_X = X[0:len(X):450]

# Define subset of 1018 diamonds to explain
X_small = X[0:len(X):53]

# Calculate KernelSHAP values
ks = KernelExplainer(
  model=lambda X: fit.predict(pd.DataFrame(X, columns=x)), 
  data = bg_X
)
sv = ks.shap_values(X_small)  # 74 seconds
sv[0:2]

# array([[-2.05007406, -0.28048747,  0.12812216,  0.01587382],
#        [-2.0858379 ,  0.04050415,  0.12830103,  0.03731644]])

## NON-SENSICAL EXAMPLE WITH 9 FEATURES

x = ["carat"] + ord + ["table", "depth", "x", "y", "z"]
X = diamonds[x].to_numpy()

# Fit model with interactions and dummy variables
fit = ols(
  "np.log(price) ~ np.log(carat) * (C(clarity) + C(cut) + C(color)) + table + depth + x + y + z", 
  data=diamonds
).fit()

# Background data (120 rows)
bg_X = X[0:len(X):450]

# Define subset of 1018 diamonds to explain
X_small = X[0:len(X):53]

# Calculate KernelSHAP values: 12 minutes
ks = KernelExplainer(
  model=lambda X: fit.predict(pd.DataFrame(X, columns=x)), 
  data = bg_X
)
sv = ks.shap_values(X_small)
sv[0:2]
# array([[-1.84279897e+00, -2.70338744e-01,  1.26610769e-01,
#          1.42423108e-02,  1.77876470e-03, -7.08444295e-04,
#         -1.72078182e-01,  1.33027467e-03, -6.44569296e-03],
#        [-1.87670887e+00,  3.93291219e-02,  1.26654599e-01,
#          3.85695742e-02, -4.87177593e-04, -4.20263565e-04,
#         -1.73988040e-01,  1.39779179e-03, -6.56062359e-03]])

# Now, using a hybrid between exact and sampling: 5 minutes
sv = ks.shap_values(X_small, nsamples=126)
sv[0:2]
# array([[-1.84279897e+00, -2.70338744e-01,  1.26610769e-01,
#          1.42423108e-02,  1.77876470e-03, -7.08444295e-04,
#         -1.72078182e-01,  1.33027467e-03, -6.44569296e-03],
#        [-1.87670887e+00,  3.93291219e-02,  1.26654599e-01,
#          3.85695742e-02, -4.87177593e-04, -4.20263565e-04,
#         -1.73988040e-01,  1.39779179e-03, -6.56062359e-03]])
