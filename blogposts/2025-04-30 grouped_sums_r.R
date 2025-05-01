# ====================================================================
# Naive benchmark for grouped calculations
#
# - grouped counts
# - grouped means
# - grouped weighted means
#
# Update as per 2025-05-01 thanks to hints from Christian Lorentzen
#
# ====================================================================

# ====================================================================
# BASE R
# ====================================================================

set.seed(1)

n <- 1e6

y <- rexp(n)
w <- runif(n)
g <- factor(sample(LETTERS[1:3], n, TRUE))
df <- data.frame(y = y, g = g, w = w)

# Grouped counts
tabulate(g)
# 333469 333569 332962

# Grouped means
# Usually faster than lapply(split(y, g), FUN = mean)
rowsum(y, g) / tabulate(g)

# A 1.000869
# B 1.001043
# C 1.000445

# Grouped weighted mean
ws <- rowsum(data.frame(y = y * w, w), g)
ws[, 1L] / ws[, 2L]
# 1.0022749 1.0017816 0.9997058

# ====================================================================
# dplyr
# ====================================================================

library(tidyverse)

df <- tibble(df)

# Grouped counts
dplyr::count(df, g)

# Grouped means
df |>
  group_by(g) |>
  summarize(mean(y))

# Grouped weighted means
df |>
  group_by(g) |>
  summarize(sum(w * y) / sum(w))

# ====================================================================
# data.table
# ====================================================================

library(data.table)

dt <- data.table(df)

# Grouped counts (use keyby for sorted output)
dt[, .N, by = g]
#         g      N
#    <fctr>  <int>
# 1:      C 332962
# 2:      B 333569
# 3:      A 333469

# Grouped means
dt[, mean(y), by = g]

# Grouped weighted means
dt[, sum(w * y) / sum(w), by = g]
dt[, weighted.mean(y, w), by = g]

# ====================================================================
# DuckDB
# ====================================================================

library(duckdb)

con <- dbConnect(duckdb())

# only registers: duckdb_register(con, name = "df", df = df)
dbWriteTable(con, name = "df", value = df)

dbGetQuery(con, "SELECT g, COUNT(*) N FROM df GROUP BY g")
dbGetQuery(con, "SELECT g, AVG(y) AS mean FROM df GROUP BY g")
con |>
  dbGetQuery("
    SELECT g, SUM(y * w) / sum(w) as wmean
    FROM df
    GROUP BY g
  ")
#   g     wmean
# 1 A 1.0022749
# 2 B 1.0017816
# 3 C 0.9997058

# ====================================================================
# collapse
# ====================================================================

library(collapse)

fcount(g)
fnobs(g, g) # Faster and does not need memory, but ignores missing values
fmean(y, g = g)
fmean(y, g = g, w = w)
#         A         B         C
# 1.0022749 1.0017816 0.9997058

# ====================================================================
# Polars
# ====================================================================

# Sys.setenv(NOT_CRAN = "true")
# install.packages("polars", repos = "https://community.r-multiverse.org")
library(polars)

dfp <- as_polars_df(df)

# Grouped counts
dfp$get_column("g")$value_counts()
dfp$select("g")$with_columns(pl$lit(1L))$group_by("g")$sum() # Faster (why...)

# Grouped means
(
  dfp
  $group_by("g")
  $agg(pl$col("y")$mean())
)

# Grouped weighted means
(
  dfp
  $group_by("g")
  $agg((pl$col("y") * pl$col("w"))$sum() / pl$col("w")$sum())
)

# shape: (3, 2)
# ┌─────┬──────────┐
# │ g   ┆ y        │
# │ --- ┆ ---      │
# │ cat ┆ f64      │
# ╞═════╪══════════╡
# │ C   ┆ 0.999706 │
# │ B   ┆ 1.001782 │
# │ A   ┆ 1.002275 │
# └─────┴──────────┘

# ====================================================================
# Benchmarking different sample sizes
# ====================================================================

# We run the code in a fresh session

library(tidyverse) # 2.0.0
library(duckdb) # 1.2.1
library(data.table) # 1.16.0
library(collapse) # 2.0.19
library(polars) # 0.22.3

polars_info() # 8 threads
setDTthreads(8)
con <- dbConnect(duckdb(config = list(threads = "8")))

set.seed(1)

N <- 10^(5:8)
m_queries <- 3
results <- vector("list", length(N) * m_queries)

for (i in seq_along(N)) {
  n <- N[i]

  # Create data
  y <- rexp(n)
  w <- runif(n)
  g <- factor(sample(LETTERS, n, TRUE))

  df <- tibble(y = y, g = g, w = w)
  dt <- data.table(df)
  dfp <- as_polars_df(df)
  dbWriteTable(con, name = "df", value = df, overwrite = TRUE)

  # Grouped counts
  results[[1 + (i - 1) * m_queries]] <- bench::mark(
    base = tabulate(g),
    dplyr = dplyr::count(df, g),
    data.table = dt[, .N, by = g],
    polars = dfp$get_column("g")$value_counts(),
    collapse = fcount(g),
    duckdb = dbGetQuery(con, "SELECT g, COUNT(*) N FROM df GROUP BY g"),
    check = FALSE,
    min_iterations = 5,
  ) |>
    bind_cols(n = n, query = "counts")

  results[[2 + (i - 1) * m_queries]] <- bench::mark(
    base = rowsum(y, g) / tabulate(g),
    dplyr = df |> group_by(g) |> summarize(mean(y)),
    data.table = dt[, mean(y), by = g],
    polars = dfp$group_by("g")$agg(pl$col("y")$mean()),
    collapse = fmean(y, g = g),
    duckdb = dbGetQuery(con, "SELECT g, AVG(y) AS mean FROM df GROUP BY g"),
    check = FALSE,
    min_iterations = 5
  ) |>
    bind_cols(n = n, query = "means")

  results[[3 + (i - 1) * m_queries]] <- bench::mark(
    base = {
      ws <- rowsum(data.frame(y = y * w, w), g)
      ws[, 1L] / ws[, 2L]
    },
    dplyr = df |> group_by(g) |> summarize(sum(w * y) / sum(w)),
    data.table = dt[, sum(w * y) / sum(w), by = g],
    polars = (
      dfp
      $group_by("g")
      $agg((pl$col("y") * pl$col("w"))$sum() / pl$col("w")$sum())
    ),
    collapse = fmean(y, g = g, w = w),
    duckdb = dbGetQuery(
      con,
      "SELECT g, SUM(y * w) / sum(w) as wmean FROM df GROUP BY g"
    ),
    check = FALSE,
    min_iterations = 5
  ) |>
    bind_cols(n = n, query = "weighted means")
}

results_df <- bind_rows(results) |>
  group_by(n, query) |>
  mutate(
    time = as.numeric(median) * 1000, # ms
    n = as.factor(n),
    approach = as.character(expression),
    relative = as.numeric(time / min(time))
  ) |>
  ungroup()

ggplot(
  results_df, aes(y = time, x = n, group = approach, color = approach)
) +
  geom_point() +
  geom_line() +
  scale_y_log10(labels = scales::label_number()) +
  facet_wrap("query") +
  labs(x = "n", y = "Time [ms]", color = element_blank()) +
  theme_gray(base_size = 14)
