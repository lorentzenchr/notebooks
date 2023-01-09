# Some examples how to use dplyr style with standard R
library(ggplot2)  # Need diamonds

# What does the native pipe do?
quote(diamonds |> head())  

# OUTPUT
# head(diamonds)

# Grouped statistics
diamonds |> 
  aggregate(cbind(price, carat) ~ color, FUN = mean)

# OUTPUT
#   color    price     carat
# 1     D 3169.954 0.6577948
# 2     E 3076.752 0.6578667
# 3     F 3724.886 0.7365385
# 4     G 3999.136 0.7711902
# 5     H 4486.669 0.9117991
# 6     I 5091.875 1.0269273
# 7     J 5323.818 1.1621368

# Join back grouped stats to relevant columns
diamonds |> 
  subset(select = c(price, color, carat)) |> 
  transform(price_per_color = ave(price, color)) |> 
  head()

# OUTPUT
#   price color carat price_per_color
# 1   326     E  0.23        3076.752
# 2   326     E  0.21        3076.752
# 3   327     E  0.23        3076.752
# 4   334     I  0.29        5091.875
# 5   335     J  0.31        5323.818
# 6   336     J  0.24        5323.818

# Plot transformed values
diamonds |> 
  transform(
    log_price = log(price),
    log_carat = log(carat)
  ) |> 
  plot(log_price ~ log_carat, col = "chartreuse4", pch = ".", data = _)

# Distribution of color within clarity
diamonds |> 
  subset(select = c(color, clarity)) |> 
  table() |> 
  prop.table(margin = 2) |> 
  addmargins(margin = 1) |> 
  round(3)

# OUTPUT
# clarity
# color      I1   SI2   SI1   VS2   VS1  VVS2  VVS1    IF
#     D   0.057 0.149 0.159 0.138 0.086 0.109 0.069 0.041
#     E   0.138 0.186 0.186 0.202 0.157 0.196 0.179 0.088
#     F   0.193 0.175 0.163 0.180 0.167 0.192 0.201 0.215
#     G   0.202 0.168 0.151 0.191 0.263 0.285 0.273 0.380
#     H   0.219 0.170 0.174 0.134 0.143 0.120 0.160 0.167
#     I   0.124 0.099 0.109 0.095 0.118 0.072 0.097 0.080
#     J   0.067 0.052 0.057 0.060 0.066 0.026 0.020 0.028
#     Sum 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000


# Barplot from discrete column
diamonds$color |> 
  table() |> 
  prop.table() |> 
  barplot(col = "chartreuse4", main = "Color")--
