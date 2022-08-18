# Kernel SHAP with Keras
# Note that the results are different from blog post due to different seeds

library(tidyverse)
library(keras)

set.seed(950)

# Response and covariates
y <- as.numeric(diamonds$price)
x <- c("carat", "color", "cut", "clarity")
X <- scale(data.matrix(diamonds[x]))

# Input layer: we have 4 covariates
input <- layer_input(shape = 4)

# Two hidden layers with contracting number of nodes
output <- input %>%
  layer_dense(units = 30, activation = "tanh") %>% 
  layer_dense(units = 15, activation = "tanh") %>% 
  layer_dense(units = 1, activation = k_exp)

# Create and compile model
nn <- keras_model(inputs = input, outputs = output)
summary(nn)

# Gamma regression loss
loss_gamma <- function(y_true, y_pred) {
  -k_log(y_true / y_pred) + y_true / y_pred
}

nn %>% 
  compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = loss_gamma
  )

# Callbacks
cb <- list(
  callback_early_stopping(patience = 20),
  callback_reduce_lr_on_plateau(patience = 5)
)

# Fit model
history <- nn %>% 
  fit(
    x = X,
    y = y,
    epochs = 100,
    batch_size = 400, 
    validation_split = 0.2,
    callbacks = cb
  )

history$metrics[c("loss", "val_loss")] %>% 
  data.frame() %>% 
  mutate(epoch = row_number()) %>% 
  filter(epoch >= 3) %>% 
  pivot_longer(cols = c("loss", "val_loss")) %>% 
ggplot(aes(x = epoch, y = value, group = name, color = name)) +
  geom_line(size = 1.4)


# Interpretation on 500 randomly selected diamonds
library(kernelshap)
library(shapviz)

ind <- sample(nrow(X), 500)

dia_small <- X[ind, ]

# 77 seconds
system.time(
  ks <- kernelshap(
    dia_small, 
    pred_fun = function(X) as.numeric(predict(nn, X, batch_size = nrow(X))), 
    bg_X = dia_small
  )
)
ks

# Output
# 'kernelshap' object representing 
# - SHAP matrix of dimension 500 x 4 
# - feature data.frame/matrix of dimension 500 x 4 
# - baseline value of 3744.153
# 
# SHAP values of first 2 observations:
#         carat     color       cut   clarity
# [1,] -110.738 -240.2758  5.254733 -720.3610
# [2,] 2379.065  263.3112 56.413680  452.3044
# 
# Corresponding standard errors:
#         carat      color       cut  clarity
# [1,] 2.064393 0.05113337 0.1374942 2.150754
# [2,] 2.614281 0.84934844 0.9373701 0.827563

sv <- shapviz(ks, X = diamonds[ind, x])
sv_waterfall(sv, 1)
sv_importance(sv, "both")
sv_dependence(sv, "carat", "auto")
