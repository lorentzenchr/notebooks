library(effectplots)
library(OpenML)
library(lightgbm)

dim(df <- getOMLDataSet(data.id = 45106L)$data)  # 1000000 7
head(df)

#   year town driver_age car_weight car_power car_age claim_nb
# 0 2018    1         51       1760       173       3        0
# 1 2019    1         41       1760       248       2        0
# 2 2018    1         25       1240       111       2        0
# 3 2019    0         40       1010        83       9        0
# 4 2018    0         43       2180       169       5        0
# 5 2018    1         45       1170       149       1        1

yvar <- "claim_nb"
xvars <- setdiff(colnames(df), yvar)

ix <- 1:800000
train <- df[ix, ]
test <- df[-ix, ]
X_train <- data.matrix(train[xvars])
X_test <- data.matrix(test[xvars])

# Training, using slightly optimized parameters found via cross-validation
params <- list(
  learning_rate = 0.05,
  objective = "poisson",
  num_leaves = 7,
  min_data_in_leaf = 50,
  min_sum_hessian_in_leaf = 0.001,
  colsample_bynode = 0.8,
  bagging_fraction = 0.8,
  lambda_l1 = 3,
  lambda_l2 = 5,
  num_threads = 7
)

set.seed(1)

fit <- lgb.train(
  params = params,
  data = lgb.Dataset(X_train, label = train$claim_nb),
  nrounds = 300
)

# 0.3 s
feature_effects(fit, v = xvars, data = X_test, y = test$claim_nb) |>
  plot(share_y = "all")

# ggsave("effect_plot.png", width = 7, height = 7)
