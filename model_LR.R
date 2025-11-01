library(tidymodels)  
library(rsample)
library(parsnip)
library(workflows)
library(broom)
library(recipes)
library(dplyr)
library(ggplot2)
library(car)        
library(glmnet)     
library(purrr)
library(tibble)



if (!exists("train") || !exists("test")) {
  if (file.exists("Kalibracja_quarto.qmd")) {
    message("Pobieram dane z Kalibracja_quarto.qmd")
    knitr::purl("Kalibracja_quarto.qmd", output = "dane_po_kalibracji.R", quiet = TRUE)
    source("dane_po_kalibracji.R")
  } else {
    stop("Brak obiektów 'train' i 'test'. Upewnij się, że skrypt EDA został uruchomiony.")
  }
} 


#chcemy przewidywac pm10
target <- "grimm_pm10"


if("date" %in% names(train_tr)) {
  train_tr <- train_tr %>% select(-date)
  test_tr  <- test_tr  %>% select(-date)
}


n <- nrow(train_tr)
initial <- floor(0.6 * n)       
assess  <- floor(0.2 * n)       
skip    <- floor(0.05 * n)      

ts_cv <- rolling_origin(
  data = train_tr,
  initial = initial,
  assess = assess,
  cumulative = TRUE,
  skip = skip
)


length(ts_cv$splits)


lm_spec <- linear_reg() %>%
  set_engine("lm")


lm_wf <- workflow() %>%
  add_model(lm_spec) %>%
  add_formula(as.formula(paste(target, "~ .")))


lm_resamples <- fit_resamples(
  lm_wf,
  resamples = ts_cv,
  metrics = metric_set(rmse, rsq, mae),
  control = control_resamples(save_pred = TRUE)
)

collect_metrics(lm_resamples)


lm_preds <- collect_predictions(lm_resamples)


lm_final <- lm_wf %>% fit(data = train_tr)


tidy(lm_final %>% extract_fit_parsnip())


lm_obj <- lm_final %>% pull_workflow_fit() %>% .$fit
vif_safe <- function(model) { #vif(lm_obj) wcześniej był problem z tą funkcją
  mm <- model.matrix(model)
  mm <- mm[, !is.na(coef(model))] 
  car::vif(lm(model$y ~ mm - 1))
}
train_tr_lm <- train_tr
test_tr_lm  <- test_tr


train_tr_lm$.fitted <- predict(lm_obj)
train_tr_lm$.resid  <- residuals(lm_obj)


train_tr_ml <- train_tr %>% select(-any_of(c(".fitted", ".resid")))
test_tr_ml  <- test_tr  %>% select(-any_of(c(".fitted", ".resid")))



ggplot(train_tr_lm, aes(.fitted, .resid)) +
  geom_point(alpha = 0.4) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residuals vs Fitted (train)")

##QQ plot
ggplot(train_tr_lm, aes(sample = .resid)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "QQ plot reszt")

# Test Shapiro-Wilka
shapiro_test <- shapiro.test(train_tr_lm$.resid[1:min(5000, nrow(train_tr_lm))])
print(shapiro_test)


test_pred_lm <- predict(lm_obj, newdata = test_tr_ml) %>% 
  bind_cols(test_tr_ml %>% select(all_of(target)))

test_pred_lm <- tibble(
  .pred = predict(lm_obj, newdata = test_tr_ml),
  !!target := test_tr_ml[[target]]
)

metrics_test_lm <- metric_set(rmse, rsq, mae)(
  test_pred_lm, truth = !!sym(target), estimate = .pred
)
print("Metryki LM na zbiorze testowym:")
print(metrics_test_lm)


ggplot(test_pred_lm, aes(x = .pred, y = !!sym(target))) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Observed vs Predicted (LM, test)", x = "Predicted", y = "Observed")

lm_coefs <- tidy(lm_obj) %>% arrange(p.value)
print("Główne współczynniki (LM):")
print(head(lm_coefs, 20))

