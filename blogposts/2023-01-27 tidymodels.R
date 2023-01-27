# Inspired by https://juliasilge.com/blog/board-games/

library(tidyverse)
library(tidymodels)
library(shapviz)

# Integer encode factors
dia_recipe <- diamonds %>%
  recipe(price ~ carat + cut + clarity + color) %>% 
  step_integer(all_nominal())

# Will explain THIS dataset later
set.seed(2)
dia_small <- diamonds[sample(nrow(diamonds), 1000), ]
dia_small_prep <- bake(
  prep(dia_recipe), 
  has_role("predictor"),
  new_data = dia_small, 
  composition = "matrix"
)
head(dia_small_prep)

# Just for illustration - in practice needs tuning!
xgboost_model <- boost_tree(
  mode = "regression",
  trees = 200,
  tree_depth = 5,
  learn_rate = 0.05,
  engine = "xgboost"
)

dia_wf <- workflow() %>%
  add_recipe(dia_recipe) %>%
  add_model(xgboost_model)

fit <- dia_wf %>%
  fit(diamonds)

# SHAP analysis
shap <- shapviz(extract_fit_engine(fit), X_pred = dia_small_prep, X = dia_small)

sv_importance(shap, kind = "both", show_numbers = TRUE)
sv_dependence(shap, "carat", color_var = "auto")
sv_dependence(shap, "clarity", color_var = "auto")
sv_force(shap, row_id = 1)
sv_waterfall(shap, row_id = 1)
