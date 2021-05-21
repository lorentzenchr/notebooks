# Random forests via XGBoost

# This script shows how to parametrize XGBoost's random forest mode in order to
# produce similar performance than a true random forest.
# The official https://xgboost.readthedocs.io/en/latest/tutorials/rf.html
# of this XGBoost feature is great. We found it important to change the
# default of parameters like `reg_lambda` or `max_depth` in order
# to get close to a standard random forest.

# To illustrate, we use a data set of information on 20'000 houses
# from Kings County, see below.

# Bad examples e.g. on Kaggle (not quoting them)

library(OpenML)
library(dplyr)
library(ranger)
library(xgboost)

set.seed(83454)

rmse <- function(y, pred) {
  sqrt(mean((y-pred)^2))
}

# Load King Country house prices dataset on OpenML
# ID 42092, https://www.openml.org/d/42092
df <- getOMLDataSet(data.id = 42092)$data
head(df)

# Prepare
df <- df %>%
  mutate(
    log_price = log(price),
    year = as.numeric(substr(date, 1, 4)),
    building_age = year - yr_built,
    zipcode = as.integer(as.character(zipcode))
)

# Define response and features
y <- "log_price"
x <- c("grade", "year", "building_age", "sqft_living",
       "sqft_lot", "bedrooms", "bathrooms", "floors", "zipcode",
       "lat", "long", "condition", "waterfront")
m <- length(x)

# random split
ix <- sample(nrow(df), 0.8 * nrow(df))

# Fit untuned random forest
system.time( # 3 s
  fit_rf <- ranger(reformulate(x, y), data = df[ix, ])
)
y_test <- df[-ix, y]

# Test RMSE: 0.173
rmse(y_test, predict(fit_rf, df[-ix, ])$pred)
# object.size(fit_rf) # 180 MB

# Fit untuned, but good(!) XGBoost random forest
dtrain <- xgb.DMatrix(data.matrix(df[ix, x]),
                      label = df[ix, y])

params <- list(
  objective = "reg:squarederror",
  learning_rate = 1,
  num_parallel_tree = 500,
  subsample = 0.63,
  colsample_bynode = floor(sqrt(m)) / m,
  reg_lambda = 0,
  max_depth = 20,
  min_child_weight = 2
)

system.time( # 20 s
  fit_xgb <- xgb.train(
    params,
    data = dtrain,
    nrounds = 1,
    verbose = 0
  )
)

pred <- predict(fit_xgb, data.matrix(df[-ix, x]))

# Test RMSE: 0.174
rmse(y_test, pred)
# xgb.save(fit_xgb, "xgb.model") # 140 MB

#==============================================================================
# diamonds data
#==============================================================================

library(ggplot2)
library(splitTools)

set.seed(345)

# We add group id and its size
dia <- diamonds %>%
  group_by(carat, cut, clarity, color, price) %>%
  mutate(id = cur_group_id(),
         id_size = n()) %>%
  ungroup() %>%
  arrange(id) %>%
  mutate(log_price = log(price))

y <- "log_price"
x <- c("carat", "cut", "clarity", "color")
m <- length(x)

# Sample training indices

ix <- partition(dia$id, p = c(train = 0.8, test = 0.2))

# Ranger
system.time( # 3 s
  fit_rf <- ranger(reformulate(x, y), data = dia[ix$train, ])
)

# 0.1043
rmse(dia[[y]][ix$test], predict(fit_rf, dia[ix$test, ])$pred)

# Fit untuned, but good(!) XGBoost random forest
dtrain <- xgb.DMatrix(data.matrix(dia[ix$train, x]),
                      label = dia[[y]][ix$train])

params <- list(
  objective = "reg:squarederror",
  learning_rate = 1,
  num_parallel_tree = 500,
  subsample = 0.63,
  colsample_bynode = floor(sqrt(m)) / m,
  reg_lambda = 0,
  max_depth = 20,
  min_child_weight=2
)

system.time( # 18 s
  fit_xgb <- xgb.train(
    params,
    data = dtrain,
    nrounds = 1,
    verbose = 0
  )
)

pred <- predict(fit_xgb, data.matrix(dia[ix$test, x]))

# Test RMSE: 0.1042
rmse(dia[[y]][ix$test], pred)
