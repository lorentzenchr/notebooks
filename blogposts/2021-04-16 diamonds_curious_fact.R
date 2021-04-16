#==============================================================================
# A curious fact about diamonds data
#==============================================================================

#==============================================================================
# THE FACT
#==============================================================================

library(tidyverse)

# We add group id and its size
dia <- diamonds %>% 
  group_by(carat, cut, clarity, color, price) %>% 
  mutate(id = cur_group_id(),
         id_size = n()) %>% 
  ungroup() %>% 
  arrange(id)

# Proportion duplicates
1 - max(dia$id) / nrow(dia)  # 0.26

# Some examples
dia %>% 
  filter(id_size > 1) %>%
  head(10)

# Most frequent
dia %>% 
  arrange(-id_size) %>% 
  head(.$id_size[1])

# A random large diamond appearing multiple times
dia %>% 
  filter(id_size > 3) %>% 
  arrange(-carat) %>% 
  head(.$id_size[1])


#==============================================================================
# MODELS
#==============================================================================

library(ranger)
library(splitTools) # one of our packages on CRAN

set.seed(8325)

# We model log(price)
dia <- dia %>% 
  mutate(y = log(price))

# Helper function: calculate rmse
rmse <- function(obs, pred) {
  sqrt(mean((obs - pred)^2))
}

# Helper function: fit model on one fold and evaluate
fit_on_fold <- function(fold, data) {
  fit <- ranger(y ~ carat + cut + color + clarity, data = data[fold, ])
  rmse(data$y[-fold], predict(fit, data[-fold, ])$pred)
}
  
# 5-fold CV for different split types
cross_validate <- function(type, data) {
  folds <- create_folds(data$id, k = 5, type = type)
  mean(sapply(folds, fit_on_fold, data = dia))
}

# Apply and plot
(results <- sapply(c("basic", "grouped"), cross_validate, data = dia))
barplot(results, col = "orange", ylab = "RMSE by 5-fold CV")
