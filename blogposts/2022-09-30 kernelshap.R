library(ggplot2)
library(kernelshap)

# Turn ordinal factors into unordered
ord <- c("clarity", "color", "cut")
diamonds[, ord] <- lapply(diamonds[ord], factor, ordered = FALSE)

# Fit model
fit <- lm(log(price) ~ log(carat) * (clarity + color + cut), data = diamonds)

# Subset of 120 diamonds used as background data
bg_X <- diamonds[seq(1, nrow(diamonds), 450), ]

# Subset of 1018 diamonds to explain
X_small <- diamonds[seq(1, nrow(diamonds), 53), c("carat", ord)]

# Exact KernelSHAP (5 seconds)
system.time(
  ks <- kernelshap(fit, X_small, bg_X = bg_X)  
)
ks

# SHAP values of first 2 observations:
#          carat     clarity     color        cut
# [1,] -2.050074 -0.28048747 0.1281222 0.01587382
# [2,] -2.085838  0.04050415 0.1283010 0.03731644

# Using parallel backend
library("doFuture")

registerDoFuture()
plan(multisession, workers = 2)  # Windows
# plan(multicore, workers = 2)   # Linux, macOS, Solaris

# 3 seconds on second call
system.time(
  ks3 <- kernelshap(fit, X_small, bg_X = bg_X, parallel = TRUE)  
)

# Visualization
library(shapviz)

sv <- shapviz(ks)
sv_importance(sv, "bee")


## NON-SENSICAL EXAMPLE WITH 9 FEATURES

fit <- lm(
  log(price) ~ log(carat) * (clarity + color + cut) + x + y + z + table + depth, 
  data = diamonds
)

# Subset of 1018 diamonds to explain
X_small <- diamonds[seq(1, nrow(diamonds), 53), setdiff(names(diamonds), "price")]

# Exact Kernel SHAP: 61 seconds
system.time(
  ks <- kernelshap(fit, X_small, bg_X = bg_X, exact = TRUE)  
)
ks
#          carat        cut     color     clarity         depth         table          x           y            z
# [1,] -1.842799 0.01424231 0.1266108 -0.27033874 -0.0007084443  0.0017787647 -0.1720782 0.001330275 -0.006445693
# [2,] -1.876709 0.03856957 0.1266546  0.03932912 -0.0004202636 -0.0004871776 -0.1739880 0.001397792 -0.006560624

# Default, using an almost exact hybrid algorithm: 17 seconds
system.time(
  ks <- kernelshap(fit, X_small, bg_X = bg_X, parallel = TRUE)  
)
#          carat        cut     color     clarity         depth         table          x           y            z
# [1,] -1.842799 0.01424231 0.1266108 -0.27033874 -0.0007084443  0.0017787647 -0.1720782 0.001330275 -0.006445693
# [2,] -1.876709 0.03856957 0.1266546  0.03932912 -0.0004202636 -0.0004871776 -0.1739880 0.001397792 -0.006560624