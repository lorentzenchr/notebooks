y <- "Sepal.Width"
x <- c("Sepal.Length", "Petal.Length", "Petal.Width")

# Scaled version of iris
iris2 <- data.frame(scale(iris[c(y, x)]))

# Fit model 
fit <- lm(reformulate(x, y), data = iris2)
summary(fit) # multiple R-squared: 0.524
(betas <- coef(fit)[x])
# Sepal.Length Petal.Length  Petal.Width 
#    1.1533143   -2.3734841    0.9758767 

# Correlations (scaling does not matter here)
(cors <- cor(iris[, y], iris[x]))
# Sepal.Length Petal.Length Petal.Width
#   -0.1175698   -0.4284401  -0.3661259
 
# The R-squared?
sum(betas * cors) # 0.524