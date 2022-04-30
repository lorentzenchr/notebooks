library(tidyverse)
library(bench)

#=================================================================
# Square root (a simple function)
#=================================================================

# Calculate square root for each element in loop
sqrt_loop <- function(x) {
  out <- numeric(length(x))
  for (i in seq_along(x)) {
    out[i] <- sqrt(x[i])
  }
  out
}

# Example
sqrt_loop(1:4) # 1.000000 1.414214 1.732051 2.000000

# Compare its performance with two alternatives
sqrt_benchmark <- function(n) {
  x <- rexp(n)
  mark(
    vectorized = sqrt(x),
    loop = sqrt_loop(x),
    vapply = vapply(x, sqrt, FUN.VALUE = 0.0),
    relative = TRUE
  )
}

# Combine results of multiple benchmarks and plot results
multiple_benchmarks <- function(one_bench, N) {
  res <- vector("list", length(N))
  for (i in seq_along(N)) {
    res[[i]] <- one_bench(N[i]) %>% 
      mutate(n = N[i], expression = names(expression))
  }
  
  ggplot(bind_rows(res), aes(n, median, color = expression)) +
    geom_point(size = 3) +
    geom_line(size = 1) +
    scale_x_log10() +
    ggtitle(deparse1(substitute(one_bench))) +
    theme(legend.position = c(0.8, 0.15))
}

# Apply simulation
multiple_benchmarks(sqrt_benchmark, N = 10^seq(3, 6, 0.25))


#=================================================================
# Paste text with some digits of float ("complicated" calculation)
#=================================================================

pretty_paste <- function(x) {
  paste("Number", prettyNum(x, digits = 5))
}

# Example
pretty_paste(pi) # "Number 3.1416"

# Again, call pretty_paste() for each element in a loop
paste_loop <- function(x) {
  out <- character(length(x))
  for (i in seq_along(x)) {
    out[i] <- pretty_paste(x[i])
  }
  out
}

# Compare its performance with two alternatives
paste_benchmark <- function(n) {
  x <- rexp(n)
  mark(
    vectorized = pretty_paste(x),
    loop = paste_loop(x),
    vapply = vapply(x, pretty_paste, FUN.VALUE = ""),
    relative = TRUE
  )
}

multiple_benchmarks(paste_benchmark, N = 10^seq(3, 5, 0.25))
