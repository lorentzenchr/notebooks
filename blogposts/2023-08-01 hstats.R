# EXAMPLE 1

library(ranger)
library(ggplot2)
library(hstats)

v <- setdiff(colnames(iris), "Species")
fit <- ranger(Species ~ ., data = iris, probability = TRUE, seed = 1)
s <- hstats(fit, v = v, X = iris)  # 8 seconds run-time
s
# Proportion of prediction variability unexplained by main effects of v:
#      setosa  versicolor   virginica 
# 0.002705945 0.065629375 0.046742035

plot(s, normalize = FALSE, squared = FALSE) +
  ggtitle("Unnormalized statistics") +
  scale_fill_viridis_d(begin = 0.1, end = 0.9)

ice(fit, v = "Petal.Length", X = iris, BY = "Petal.Width", n_max = 150) |> 
  plot(center = TRUE) +
  ggtitle("Centered ICE plots")


# EXAMPLE 2

library(DALEX)
library(ranger)
library(hstats)

set.seed(1)

fit <- ranger(Sepal.Length ~ ., data = iris)
ex <- explain(fit, data = iris[-1], y = iris[, 1])

s <- hstats(ex)  # 2 seconds
s  # Non-additivity index 0.054
plot(s)
plot(ice(ex, v = "Sepal.Width", BY = "Petal.Width"), center = TRUE)