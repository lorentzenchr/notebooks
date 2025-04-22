# =====================================================================
# Streaming Conversion of CSV to Parquet
# =====================================================================

# Make large CSV
library(data.table)

set.seed(1)

n <- 1e8

df <- data.frame(
  X = sample(letters[1:3], n, TRUE),
  Y = runif(n),
  Z = sample(1:5, n, TRUE)
)

fwrite(df, "data.csv")


# Approach 1: DuckDB

library(duckdb)

con <- dbConnect(duckdb(config = list(threads = "8", memory_limit = "4GB")))

system.time(
  dbSendQuery(
    con,
    "
    COPY (
      SELECT Y, Z
      FROM 'data.csv'
      WHERE X == 'a'
      ORDER BY Y
    ) TO 'data.parquet' (FORMAT parquet, COMPRESSION zstd)
    "
  )
)

# Check
dbGetQuery(con, "SELECT COUNT(*) N FROM 'data.parquet'") # 33329488
dbGetQuery(con, "SELECT * FROM 'data.parquet' LIMIT 5")
#              Y Z
# 1 5.355105e-09 4
# 2 9.080395e-09 5
# 3 2.258457e-08 2
# 4 3.445894e-08 2
# 5 6.891787e-08 1

# Now with Polars
# Sys.setenv(NOT_CRAN = "true")
# install.packages("polars", repos = "https://community.r-multiverse.org")
library(polars)

polars_info()

system.time(
  (
    pl$scan_csv("data.csv")
    $filter(pl$col("X") == "a")
    $drop("X")
    $sort("Y")
    $sink_parquet("data.parquet", row_group_size = 1e5)
  )
)

# Check
pl$scan_parquet("data.parquet")$head()$collect()
# shape: (5, 2)
# ┌───────────┬─────┐
# │ Y         ┆ Z   │
# │ ---       ┆ --- │
# │ f64       ┆ i64 │
# ╞═══════════╪═════╡
# │ 5.3551e-9 ┆ 4   │
# │ 9.0804e-9 ┆ 5   │
# │ 2.2585e-8 ┆ 2   │
# │ 3.4459e-8 ┆ 2   │
# │ 6.8918e-8 ┆ 1   │
# └───────────┴─────┘
