# Fix seed, set constants
set.seed(2006)
sample_sizes <- c(1, 10, 30, 1000)
nsims <- 10000

# Helper function: Mean of one sample of X
one_mean <- function(n, p = c(0.8, 0.2)) {
  mean(sample(0:1, n, replace = TRUE, prob = p))
}
# one_mean(10)

# Simulate and plot
par(mfrow = c(2, 2), mai = rep(0.4, 4))

for (n in sample_sizes) {
  means <- replicate(nsims, one_mean(n))
  hist(means, breaks = "FD", 
       # xlim = 0:1, # uncomment for LLN
       main = sprintf("n=%i", n))
}
