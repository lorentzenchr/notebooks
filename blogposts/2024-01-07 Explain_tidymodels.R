# https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

library(tidyverse)
library(tidymodels)
library(hstats)
library(kernelshap)
library(shapviz)
library(patchwork)

df0 <- read.csv("diabetes_prediction_dataset.csv")  # from above Kaggle link
dim(df0)  # 100000 9
head(df0)
# gender age hypertension heart_disease smoking_history   bmi HbA1c_level blood_glucose_level diabetes
# Female  80            0             1           never 25.19         6.6                 140        0
# Female  54            0             0         No Info 27.32         6.6                  80        0
#   Male  28            0             0           never 27.32         5.7                 158        0
# Female  36            0             0         current 23.45         5.0                 155        0
#   Male  76            1             1         current 20.14         4.8                 155        0
# Female  20            0             0           never 27.32         6.6                  85        0

summary(df0)
anyNA(df0)  # FALSE
table(df0$smoking_history, useNA = "ifany")

# DATA PREPARATION

# Note: tidymodels needs a factor response for classification
df1 <- df0 |>
  transform(
    y = factor(diabetes, levels = 0:1, labels = c("No", "Yes")),
    female = (gender == "Female") * 1,
    smoking_history = factor(
      smoking_history, 
      levels = c("No Info", "never", "former", "not current", "current", "ever")
    ),
    bmi = pmin(bmi, 50)
  )

# UNIVARIATE ANALYSIS

df1  |>  
  select(age, bmi, HbA1c_level, blood_glucose_level) |> 
  pivot_longer(everything()) |> 
  ggplot(aes(value)) +
  geom_histogram(fill = "chartreuse4", bins = 19) +
  facet_wrap(~ name, scale = "free_x")

ggplot(df1, aes(smoking_history)) +
  geom_bar(fill = "chartreuse4")

df1 |> 
  select(heart_disease, hypertension, female, diabetes) |>
  pivot_longer(everything()) |> 
  ggplot(aes(name, value)) +
  stat_summary(fun = mean, geom = "bar", fill = "chartreuse4") +
  xlab(element_blank())

# MODELING

# We don't use blood indicators here to not run into causality problems and so that
# everyone can judge his/her risk

set.seed(1)
ix <- initial_split(df1, strata = diabetes, prop = 0.8)
train <- training(ix)
test <- testing(ix)

xvars <- c("age", "bmi", "smoking_history", "heart_disease", "hypertension", "female")

rf_spec <- rand_forest(trees = 500) |> 
  set_mode("classification") |> 
  set_engine("ranger", num.threads = NULL, seed = 49)

rf_wf <- workflow() |> 
  add_model(rf_spec) |>
  add_formula(reformulate(xvars, "y"))

model <- rf_wf |> 
    fit(train)

# predict() gives No/Yes columns
predict(model, head(test), type = "prob")
# .pred_No .pred_Yes
#    0.981    0.0185

# We need to extract only the "Yes" probabilities
pf <- function(m, X) {
  predict(m, X, type = "prob")$.pred_Yes
}
pf(model, head(test))  # 0.01854290 ...

# CLASSIC XAI

# 4 times repeated permutation importance wrt test logloss
imp <- perm_importance(
  model, X = test, y = "diabetes", v = xvars, pred_fun = pf, loss = "logloss"
)
plot(imp) +
  xlab("Increase in test logloss")

# Partial dependence of age
partial_dep(model, v = "age", train, pred_fun = pf) |> 
  plot()

# All PDP in one patchwork
p <- lapply(xvars, function(x) plot(partial_dep(model, v = x, X = train, pred_fun = pf)))
wrap_plots(p) &
  ylim(0, 0.23) &
  ylab("Probability")

# Friedman's H stats
system.time( # 20 s
  H <- hstats(model, train[xvars], approx = TRUE, pred_fun = pf)
)
H  # 15% of prediction variability comes from interactions
plot(H)

# Stratified PDP of strongest interaction
partial_dep(model, "age", BY = "bmi", X = train, pred_fun = pf) |> 
  plot(show_points = FALSE)

# SHAP ANALYSIS

# Note 1: If p is larger, use kernelshap() instead of permshap()
# Note 2: Takes long because predict function of ranger() is slow
# Note 3: Can try out {treeshap}, but it might eat too much memory
set.seed(1)
X_explain <- train[sample(1:nrow(train), 1000), xvars]
X_background <- train[sample(1:nrow(train), 200), ]

system.time(  # 10 minutes
  shap_values <- permshap(model, X = X_explain, bg_X = X_background, pred_fun = pf)
)
shap_values <- shapviz(shap_values)
shap_values  # 'shapviz' object representing 1000 x 6 SHAP matrix
saveRDS(shap_values, file = "shap_values.rds")
# shap_values <- readRDS("shap_values.rds")

sv_importance(shap_values, show_numbers = TRUE)
sv_importance(shap_values, kind = "bee")
sv_dependence(shap_values, v = xvars) &
  ylim(-0.14, 0.24) &
  ylab("Probability")
