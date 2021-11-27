# Random forests with monotonic constraints

# This script shows how to use XGBoost's random forest mode in order to
# produce random forests with monotonic constraints.

# To illustrate, we use a data set of information on 20'000 houses
# from Kings County, see below.


library(farff)
library(OpenML)
library(dplyr)
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
    log_sqft_lot = log(sqft_lot),
    year = as.numeric(substr(date, 1, 4)),
    building_age = year - yr_built,
    zipcode = as.integer(as.character(zipcode))
  )

# Define response and features
y <- "log_price"
x <- c("grade", "year", "building_age", "sqft_living",
       "log_sqft_lot", "bedrooms", "bathrooms", "floors", "zipcode",
       "lat", "long", "condition", "waterfront")
m <- length(x)

# random split
ix <- sample(nrow(df), 0.8 * nrow(df))
y_test <- df[[y]][-ix]

# Fit untuned, but good(!) XGBoost random forest
dtrain <- xgb.DMatrix(data.matrix(df[ix, x]),
                      label = df[ix, y])

params <- list(
  objective = "reg:squarederror",
  learning_rate = 1,
  num_parallel_tree = 500,
  subsample = 0.63,
  colsample_bynode = 1/3,
  reg_lambda = 0,
  max_depth = 20,
  min_child_weight = 2
)

system.time( # 25 s
  unconstrained <- xgb.train(
    params,
    data = dtrain,
    nrounds = 1,
    verbose = 0
  )
)

pred <- predict(unconstrained, data.matrix(df[-ix, x]))

# Test RMSE: 0.173
rmse(y_test, pred)

# ICE curves via our flashlight package
library(flashlight)

pred_xgb <- function(m, X) predict(m, data.matrix(X[, x]))

fl <- flashlight(
  model = unconstrained,
  label = "unconstrained",
  data = df[ix, ],
  predict_function = pred_xgb
)

light_ice(fl, v = "log_sqft_lot", indices = 1:9,
          evaluate_at = seq(7, 11, by = 0.1)) %>%
  plot()

#=================================================
# Constrained random forests
#=================================================


# Monotonic increasing constraint
(params$monotone_constraints <- 1 * (x == "log_sqft_lot"))

system.time( #  179s
  monotonic <- xgb.train(
  params,
  data = dtrain,
  nrounds = 1,
  verbose = 0
  )
)

pred <- predict(monotonic, data.matrix(df[-ix, x]))

# Test RMSE: 0.176
rmse(y_test, pred)

fl_m <- flashlight(
  model = monotonic,
  label = "monotonic",
  data = df[ix, ],
  predict_function = pred_xgb
)

light_ice(fl_m, v = "log_sqft_lot", indices = 1:9,
          evaluate_at = seq(7, 11, by = 0.1)) %>%
  plot()
