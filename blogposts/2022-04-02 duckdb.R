library(OpenML)
library(duckdb)
library(tidyverse)
library(arrow)

# # Load data
df <- getOMLDataSet(data.id = 42092)$data

# Initialize duckdb, register df and materialize first query
con = dbConnect(duckdb())

duckdb_register(con, name = "df", df = df)
con %>% 
  dbSendQuery("SELECT * FROM df limit 5") %>% 
  dbFetch()

# Average price per grade
query <- 
  "
  SELECT AVG(price) avg_price, grade 
  FROM df 
  GROUP BY grade
  ORDER BY grade
  "
avg <- con %>% 
  dbSendQuery(query) %>% 
  dbFetch()

avg

# Save df and avg to different file types
write_parquet(df, "housing.parquet")
write.csv(avg, "housing_avg.csv", row.names = FALSE)

# "Complex" query
query2 <- "
  SELECT price, sqft_living, A.grade, avg_price
  FROM 'housing.parquet' A
  LEFT JOIN 'housing_avg.csv' B
  ON A.grade = B.grade
  WHERE B.avg_price > 1000000
  "

expensive_grades <- con %>% 
  dbSendQuery(query2) %>% 
  dbFetch()

head(expensive_grades)

dbDisconnect(con)
