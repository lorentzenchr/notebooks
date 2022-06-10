library(shapviz)
library(ggplot2)
library(xgboost)

set.seed(3653)

# Create model
X <- diamonds[c("carat", "cut", "color", "clarity")]
dtrain <- xgb.DMatrix(data.matrix(X), label = diamonds$price)

fit <- xgb.train(
  params = list(learning_rate = 0.1, objective = "reg:squarederror"),
  data = dtrain,
  nrounds = 65L
)
X_small <- X[sample(nrow(X), 2000L), ]

# Create "shapviz" object
shp <- shapviz(fit, X_pred = data.matrix(X_small), X = X_small)

# Two ways to visualize single predictions
sv_waterfall(shp, row_id = 1)
sv_force(shp, row_id = 1)

# Different types of importance plots
sv_importance(shp)
sv_importance(shp, kind = "bar")
sv_importance(shp, kind = "both", alpha = 0.2, width = 0.2)

# Dependence plot
sv_dependence(shp, v = "color", "auto")
