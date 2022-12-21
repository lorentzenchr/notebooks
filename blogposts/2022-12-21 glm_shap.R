library(OpenML)
library(tidyverse)
library(splines)
library(doFuture)
library(kernelshap)
library(shapviz)

raw <- OpenML::getOMLDataSet(43093)$data

# Lump rare level 3 and log transform the land size
prep <- raw %>%
  mutate(
    structure_quality = factor(structure_quality, labels = c(1, 2, 4, 4, 5)),
    log_landsize = log(LND_SQFOOT)
  )

# 0) Build model
xvars <- c("TOT_LVG_AREA", "log_landsize", "structure_quality",
           "CNTR_DIST", "age", "month_sold")

fit <- glm(
  SALE_PRC ~ ns(log(CNTR_DIST), df = 4) * ns(log(TOT_LVG_AREA), df = 4) +
    log_landsize + structure_quality + ns(age, df = 4) + ns(month_sold, df = 4),
  family = Gamma("log"),
  data = prep
)
summary(fit)

# 1) Select rows to explain
set.seed(1)
X <- prep[sample(nrow(prep), 1000), xvars]

# 2) Select small representative background data
bg_X <- prep[sample(nrow(prep), 100), ]

# 3) Calculate SHAP values in fully parallel mode
registerDoFuture()
plan(multisession, workers = 6)  # Windows
# plan(multicore, workers = 6)   # Linux, macOS, Solaris

system.time( # <10 seconds
  shap_values <- kernelshap(
    fit, X, bg_X = bg_X, parallel = T, parallel_args = list(.packages = "splines")
  )
)

# 4) Analyze them
sv <- shapviz(shap_values)

sv_importance(sv, show_numbers = TRUE) +
  ggtitle("SHAP Feature Importance")

sv_dependence(sv, "log_landsize")
sv_dependence(sv, "structure_quality")
sv_dependence(sv, "age")
sv_dependence(sv, "month_sold")
sv_dependence(sv, "TOT_LVG_AREA", color_var = "auto")
sv_dependence(sv, "CNTR_DIST", color_var = "auto")

# Slope of log_landsize: 0.2255946
diff(sv$S[1:2, "log_landsize"]) / diff(sv$X[1:2, "log_landsize"])

# Difference between structure quality 4 and 5: 0.2184365
diff(sv$S[2:3, "structure_quality"])
