library(shapviz)
library(tidyverse)
library(h2o)

h2o.init()

set.seed(1)

# Get rid of that darn ordinals
ord <- c("clarity", "cut", "color")
diamonds[, ord] <- lapply(diamonds[, ord], factor, ordered = FALSE)

# Minimally tuned GBM with 260 trees, determined by early-stopping with CV
dia_h2o <- as.h2o(diamonds)
fit <- h2o.gbm(
  c("carat", "clarity", "color", "cut"),
  y = "price",
  training_frame = dia_h2o,
  nfolds = 5,
  learn_rate = 0.05,
  max_depth = 4,
  ntrees = 10000,
  stopping_rounds = 10,
  score_each_iteration = TRUE
)
fit

# SHAP analysis on about 2000 diamonds
X_small <- diamonds %>%
  filter(carat <= 2.5) %>%
  sample_n(2000) %>%
  as.h2o()

shp <- shapviz(fit, X_pred = X_small)

sv_importance(shp, show_numbers = TRUE)
sv_importance(shp, show_numbers = TRUE, kind = "bee")
sv_dependence(shp, "color", "auto", alpha = 0.5)
sv_force(shp, row_id = 1)
sv_waterfall(shp, row_id = 1)
