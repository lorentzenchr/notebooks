library(tidyverse)

# Source and countries
link <- "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
countries <- c("Switzerland", "United_States_of_America",
               "Germany", "Sweden")

# Import
df0 <- read_csv(link)

# Data prep
df <- df0 %>%
  mutate(Date = lubridate::dmy(dateRep),
         Deaths = deaths_weekly / (popData2019 / 1e6))  %>%
  rename(Country = countriesAndTerritories) %>%
  filter(Date >= "2020-03-01",
         Country %in% countries)

# Plot
ggplot(df, aes(x = Date, y = Deaths, color = Country)) +
  geom_line(size = 1) +
  ylab("Weekly deaths per Mio") +
  theme(legend.position = c(0.2, 0.85))
