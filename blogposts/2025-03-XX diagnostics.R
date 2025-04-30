library(OpenML)  # getOMLDataSet, v1.12
library(effectplots)  # feature_effects, v0.2.2
library(dplyr)
library(ggplot2)  # v3.5.1
library(ggfortify)  # autoplot, v.0.4.17
library(splines)  # bs

# munich-rent-index-1999
# https://www.openml.org/d/46772
df = getOMLDataSet(data.id = 46772L)$data

# train-test split
set.seed(112358)
train_ind = sample(seq_len(nrow(df)), size = floor(0.8 * nrow(df)))
df_train = df[train_ind, ]
df_test = df[-train_ind, ]

# linear model
model = lm(
  formula = rent ~ bs(area, degree = 3, df = 4) + yearc + location + bath + kitchen + cheating,
  data = df_train
)
summary(model)
confint(model)

# residuals vs fitted plot
# plot(model, which = c(1, 2))  # this is the standard, but we want a more modern look
autoplot(model, which = c(1, 2))
# density plot of residuals
ggplot(model, aes(x = .fitted, y = .resid)) + geom_point() +
  geom_density_2d() + geom_density_2d_filled(alpha = 0.5)

# average bias
print(paste("Train set mean residual:", mean(resid(model))))
print(paste("Test set mean residual: ", mean(df_test$rent - predict(model, df_test))))

# reliability diagram
iso_train = isoreg(x = model$fitted.values, y = df_train$rent)
iso_test = isoreg(x = predict(model, df_test), y = df_test$rent)
bind_rows(
  tibble(set = "train", x = iso_train$x[iso_train$ord], y = iso_train$yf),
  tibble(set = "test", x = iso_test$x[iso_test$ord], y = iso_test$yf),
) |>
  ggplot(aes(x=x, y=y, color=set)) + geom_line() +
  geom_abline(intercept = 0, slope = 1, linetype="dashed") +
  ggtitle("Reliability Diagram")

# effect plots
xvars = c("area", "yearc", "bath", "kitchen", "cheating")
m_train = feature_effects(model, v = xvars, data = df_train, y = df_train$rent)
m_test = feature_effects(model, v = xvars, data = df_test, y = df_test$rent)

c(m_train, m_test) |> 
  plot(
    share_y = "rows",
    ncol = 2,
    byrow = FALSE,
    stats = c("y_mean", "pred_mean", "pd"),
    subplot_titles = FALSE,
    # plotly = TRUE,
    title = "Left: Train - Right: Test",
  )

# bias plot
c(m_train[c("area", "yearc")], m_test[c("area", "yearc")]) |> 
  plot(
    ylim = c(-150, 150),
    ncol = 2,
    byrow = FALSE,
    stats = "resid_mean",
    subplot_titles = FALSE,
    title = "Left: Train - Right: Test",
    # plotly = TRUE,
    interval = "ci"
  )
