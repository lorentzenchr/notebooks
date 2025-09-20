library(xgboost)
library(ggplot2)
library(patchwork)
library(shapviz)
library(kernelshap)

options(shapviz.viridis_args = list(option = "D", begin = 0.1, end = 0.9))

set.seed(1)

# https://github.com/stedy/Machine-Learning-with-R-datasets
df <- read.csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv")

# Gamma GLM with interactions
fit_glm <- glm(charges ~ . * smoker, data = df, family = Gamma(link = "log"))

# Use SHAP to explain
xvars <- c("age", "sex", "bmi", "children", "smoker", "region")
X_explain <- head(df[xvars], 500)

# The new sampling permutation algo (forced with exact = FALSE)
shap_glm <- permshap(fit_glm, X_explain, exact = FALSE, seed = 1) |>
  shapviz()

sv_importance(shap_glm, kind = "bee")

sv_dependence(
  shap_glm,
  v = xvars,
  share_y = TRUE,
  color_var = "smoker"
) + # we rotate axis labels of *last* plot, otherwise use &
  guides(x = guide_axis(angle = 45))

# XGBoost model (sloppily without tuning)
X_num <- data.matrix(df[xvars])
fit_xgb <- xgb.train(
  params = list(objective = "reg:gamma", learning_rate = 0.2),
  data = xgb.DMatrix(X_num, label = df$charges),
  nrounds = 100
)

shap_xgb <- shapviz(fit_xgb, X_pred = X_num, X = df, interactions = TRUE)

# SHAP interaction/main-effect strength
sv_interaction(shap_xgb, kind = "bar", fill = "darkred")

# Study interaction/main-effects of "smoking"
sv_dependence(
  shap_xgb,
  v = xvars,
  color_var = "smoker",
  ylim = c(-1, 1),
  interactions = TRUE
) + # we rotate axis labels of *last* plot, otherwise use &
  guides(x = guide_axis(angle = 45))
