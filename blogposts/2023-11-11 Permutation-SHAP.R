library(kernelshap)
library(ranger)

differences <- numeric(4)

for (depth in 1:4) {
  fit <- ranger(
    Sepal.Length ~ Petal.Width + Petal.Length + Species, 
    mtry = 3,
    data = iris, 
    max.depth = depth,
    seed = 1
  )
  ps <- permshap(fit, iris[3:5], bg_X = iris)
  ks <- kernelshap(fit, iris[3:5], bg_X = iris)
  differences[depth] <- mean(abs(ks$S - ps$S))
}
differences
ps
ks

for (depth in 1:4) {
  fit <- ranger(
    Sepal.Length ~ Petal.Width + Petal.Length + Species, 
    mtry = 3,
    data = iris, 
    max.depth = depth,
    seed = 1
  )
  ps <- permshap(fit, iris[2:5], bg_X = iris)
  ks <- kernelshap(fit, iris[2:5], bg_X = iris)
  differences[depth] <- mean(abs(ks$S - ps$S))
}
differences
ps
ks