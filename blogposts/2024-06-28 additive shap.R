library(lightgbm)
library(kernelshap)
library(shapviz)

#===================================================================
# Make small data
#===================================================================

make_data <- function(n = 100) {
  x1 <- seq(0.01, 1, length = n)
  data.frame(
    x1 = x1,
    x2 = log(x1),
    x3 = x1 > 0.7
  ) |>
    transform(y = 1 + 0.2 * x1 + 0.5 * x2 + x3 + sin(2 * pi * x1))
}
df <- make_data()
head(df)
cor(df) |>
  round(2)

#===================================================================
# Additive linear model and additive boosted trees
#===================================================================

# Linear regression
fit_lm <- lm(y ~ poly(x1, 3) + poly(x2, 3) + x3, data = df)
summary(fit_lm)

# Boosted trees
xvars <- setdiff(colnames(df), "y")
X <- data.matrix(df[xvars])

params <- list(
  learning_rate = 0.05,
  objective = "mse",
  max_depth = 1,
  colsample_bynode = 0.7
)

fit_lgb <- lgb.train(
  params = params,
  data = lgb.Dataset(X, label = df$y),
  nrounds = 300
)

#===================================================================
# Three different types of SHAP for the two models
#===================================================================

system.time({  # 1s
  shap_lm <- list(
    add = shapviz(additive_shap(fit_lm, df)),
    kern = kernelshap(fit_lm, X = df[xvars], bg_X = df),
    perm = permshap(fit_lm, X = df[xvars], bg_X = df)
  )

  shap_lgb <- list(
    tree = shapviz(fit_lgb, X),
    kern = kernelshap(fit_lgb, X = X, bg_X = X),
    perm = permshap(fit_lgb, X = X, bg_X = X)
  )
})

# Consistent SHAP values for linear regression
all.equal(shap_lm$add$S, shap_lm$perm$S)
all.equal(shap_lm$kern$S, shap_lm$perm$S)

# Consistent SHAP values for boosted trees
all.equal(shap_lgb$lgb_tree$S, shap_lgb$lgb_perm$S)
all.equal(shap_lgb$lgb_kern$S, shap_lgb$lgb_perm$S)

# Linear coefficient of x3 equals slope of SHAP values
tail(coef(fit_lm), 1)                # 1.112096
diff(range(shap_lm$kern$S[, "x3"]))  # 1.112096

sv_dependence(shap_lm$add, xvars, color_var = NULL)

