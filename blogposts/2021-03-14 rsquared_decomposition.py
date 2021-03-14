# Code to demonstrate decomposition of insample R-squared
# into crossproduct of correlations with target and normalized betas.

# Import packages
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load data
iris = datasets.load_iris(as_frame=True).data
print("The data:", iris.head(3), sep = "\n")

# Specify response
yvar = "sepal width (cm)"

# Correlations of everyone with response
cors = iris.corrwith(iris[yvar]).drop(yvar)
print("\nCorrelations:", cors, sep = "\n")

# Prepare scaled response and covariables
X = StandardScaler().fit_transform(iris.drop(yvar, axis=1))
y = StandardScaler().fit_transform(iris[[yvar]])

# Fit linear regression
OLS = LinearRegression().fit(X, y)
betas = OLS.coef_[0]
print("\nScaled coefs:", betas, sep = "\n")

# R-squared via scikit-learn: 0.524
print(f"\nUsual R-squared:\t {OLS.score(X, y): .3f}")

# R-squared via decomposition: 0.524
rsquared = betas @ cors.values
print(f"Applying the formula:\t {rsquared: .3f}")