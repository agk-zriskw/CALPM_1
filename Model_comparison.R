# XGB, SVR, LR Models comparison

# Wczytanie bibliotek
library(tidyverse)
library(tidymodels)
library(ggplot2)
library(patchwork)
library(knitr)
library(xgboost)
library(kernlab)

tidymodels_prefer()
set.seed(42)

# Wczytanie danych testowych

if (!exists("test") || !exists("test_tr")) {
  message("Pobieram dane z Kalibracja_quarto.qmd do wygenerowania `test` i `test_tr`")
  if (file.exists("Kalibracja_quarto.qmd")) {
    knitr::purl("Kalibracja_quarto.qmd", output = "dane_po_kalibracji.R", quiet = TRUE)
    source("dane_po_kalibracji.R")
  } else {
    stop("Brak pliku 'Kalibracja_quarto.qmd' do wygenerowania danych.")
  }
} else {
  message("Dane `test` i `test_tr` są już wczytane.")
}

# Definicja metryk
metrics_set <- metric_set(rmse, rsq, mae)

# Wczytanie modeli i generowanie predykcji

wszystkie_predykcje <- tibble()
wszystkie_metryki <- tibble()

# Model 1 - XGBoost
xgb_model_path <- "wyniki_xgb/final_xgb_workflow.rds"
if (file.exists(xgb_model_path)) {
  message("Wczytuję model XGB...")
  xgb_final_wf <- readRDS(xgb_model_path)
  
  # Generowanie predykcji na `test`
  xgb_preds <- predict(xgb_final_wf, new_data = test) |>
    bind_cols(test |> select(grimm_pm10, date)) |>
    mutate(model = "XGBoost")
  
  # Obliczanie metryk
  xgb_metrics <- metrics_set(xgb_preds, truth = grimm_pm10, estimate = .pred) |>
    mutate(model = "XGBoost")
  
  wszystkie_predykcje <- bind_rows(wszystkie_predykcje, xgb_preds)
  wszystkie_metryki <- bind_rows(wszystkie_metryki, xgb_metrics)
  
} else {
  warning("Nie znaleziono zapisanego modelu XGB: ", xgb_model_path)
}

# Model 2 - SVR
svr_model_path <- "wyniki/final_svr_model.rds"
if (file.exists(svr_model_path)) {
  message("Wczytuję model SVR...")
  svr_final_fit <- readRDS(svr_model_path)
  
  # Generowanie predykcji na `test`
  svr_preds <- predict(svr_final_fit, new_data = test) |>
    bind_cols(test |> select(grimm_pm10, date)) |>
    mutate(model = "SVR")
  
  # Obliczanie metryk
  svr_metrics <- metrics_set(svr_preds, truth = grimm_pm10, estimate = .pred) |>
    mutate(model = "SVR")
  
  wszystkie_predykcje <- bind_rows(wszystkie_predykcje, svr_preds)
  wszystkie_metryki <- bind_rows(wszystkie_metryki, svr_metrics)
  
} else {
  warning("Nie znaleziono zapisanego modelu SVR: ", svr_model_path)
}

# Model 3 - LR
lr_model_path <- "wyniki/lm_final_model.rds"
if (file.exists(lr_model_path)) {
  message("Wczytuję model LR...")
  lr_final_model <- readRDS(lr_model_path)
  
  # WAŻNE: Model LR był trenowany na `train_tr`, więc predykcje robimy na `test_tr`
  lr_preds <- predict(lr_final_model, new_data = test_tr) |>
    bind_cols(test_tr |> select(grimm_pm10, date)) |> # `date` jest w `test_tr` jako "id"
    mutate(model = "Linear Regression")
  
  # Obliczanie metryk
  lr_metrics <- metrics_set(lr_preds, truth = grimm_pm10, estimate = .pred) |>
    mutate(model = "Linear Regression")
  
  wszystkie_predykcje <- bind_rows(wszystkie_predykcje, lr_preds)
  wszystkie_metryki <- bind_rows(wszystkie_metryki, lr_metrics)
  
} else {
  warning("Nie znaleziono zapisanego modelu LR: ", lr_model_path)
}

# Porównanie wyników

cat("\n\n--- PORÓWNANIE METRYK NA ZBIORZE TESTOWYM ---\n\n")

metryki_tabela <- wszystkie_metryki |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  arrange(rmse)

print(knitr::kable(metryki_tabela, digits = 4))


cat("\n\nGenerowanie wykresu porównawczego...\n")

if (nrow(wszystkie_predykcje) > 0) {
  
  diag_line <- geom_abline(
    color = "firebrick", 
    linetype = "dashed", 
    linewidth = 1
  )
  
  min_val <- min(c(wszystkie_predykcje$.pred, wszystkie_predykcje$grimm_pm10), na.rm = TRUE)
  max_val <- max(c(wszystkie_predykcje$.pred, wszystkie_predykcje$grimm_pm10), na.rm = TRUE)
  lims <- c(min_val, max_val)
  
  plot_comparison <- ggplot(wszystkie_predykcje, aes(x = .pred, y = grimm_pm10)) +
    geom_point(alpha = 0.3) +
    diag_line +
    facet_wrap(~ model) +
    coord_fixed(ratio = 1, xlim = lims, ylim = lims) +
    labs(
      title = "Porównanie predykcji modeli na zbiorze testowym",
      subtitle = "Wartości obserwowane vs. predykowane",
      x = "Wartości predykowane",
      y = "Wartości obserwowane (grimm_pm10)"
    ) +
    theme_minimal()
  
  print(plot_comparison)
  
  # Zapis wykresu
  # ggsave("wyniki/porownanie_predykcji_test.png", plot_comparison, width = 10, height = 5, dpi = 300)
  
} else {
  message("Brak danych predykcyjnych do wygenerowania wykresu.")
}



