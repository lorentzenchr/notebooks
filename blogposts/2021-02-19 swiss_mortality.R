library(tidyverse)
library(lubridate)

# Fetch data
df_mortality = read_csv(
  "https://www.mortality.org/Public/STMF/Outputs/stmf.csv",
  skip=2,
)

# 1. Select countries of interest and only "both" sexes
# Note: Germany "DEUTNP" and "USA" have short time series
# 2. Change to ISO-3166-1 ALPHA-3 codes
# 3.Create population pro rata temporis (exposure) to ease aggregation
df_mortality <- df_original
df_mortality <- df_mortality %>% 
  filter(CountryCode %in% c("CAN", "CHE", "FRATNP", "GBRTENW", "SWE"),
         Sex == "b") %>% 
  mutate(CountryCode = recode(CountryCode,"FRATNP" = "FRA",
                              "GBRTENW" = "England & Wales"),
         population = DTotal / RTotal)

# Data aggregation per year and country
df <- df_mortality %>%
  group_by(Year, CountryCode) %>% 
  summarise(population = sum(population),
            deaths = sum(DTotal)) %>% 
  mutate(CDR = deaths / population,
         Year = ymd(Year, truncated = 2))

ggplot(df, aes(x = Year, y = CDR, color = CountryCode)) +
  geom_line(size = 1) +
  ylab("Crude Death Rate per Year") +
  theme(legend.position = c(0.2, 0.8))
