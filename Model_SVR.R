# Wczytanie bibliotek -----------------------------------------------------------
library(tidyverse)
library(rsample)
library(recipes)
library(parsnip)
library(kernlab)
library(tidymodels)
library(tictoc)
library(stringr)

tidymodels_prefer()
set.seed(111)

# Wczytanie danych --------------------------------------------------------------
if (!exists("train") || !exists("test")) {
  if (file.exists("Kalibracja_quarto.qmd")) {
    message("Pobieram dane z Kalibracja_quarto.qmd")
    knitr::purl("Kalibracja_quarto.qmd", output = "dane_po_kalibracji.R", quiet = TRUE)
    source("dane_po_kalibracji.R")
  } else {
    stop("Brak obiektów 'train' i 'test'. Upewnij się, że skrypt EDA został uruchomiony.")
  }
} 

# Lista zmiennych do usunięcia
if (!exists("to_drop")) {
  to_drop <- character(0)
}

final_to_drop <- intersect(to_drop, names(train))
if ("poj_h" %in% names(train) && !("poj_h" %in% final_to_drop)) {
  final_to_drop <- c(final_to_drop, "poj_h")
}
message(paste("Zmienne do usunięcia:", paste(final_to_drop, collapse = ", ")))


# PRZEPISY DLA SVR ----------------------------------------------------

# RECEPTA 1: BAZOWA (bez interakcji)
rec_svr_base <- recipe(grimm_pm10 ~ ., data = train) |>
  update_role(date, new_role = "id") |>
  step_mutate(
    hour_s = sin(2*pi*as.numeric(hour)/24),
    hour_c = cos(2*pi*as.numeric(hour)/24),
    wday_s = sin(2*pi*as.numeric(wday)/7),
    wday_c = cos(2*pi*as.numeric(wday)/7)
  ) |>
  step_rm(hour, wday) |>
  step_rm(any_of(final_to_drop)) |>
  step_YeoJohnson(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_zv(all_predictors())


# RECEPTA 2: INTERAKCJE POGODOWE — FIX
rec_svr_weather <- recipe(grimm_pm10 ~ ., data = train) |>
  update_role(date, new_role = "id") |>
  step_mutate(
    hour_s = sin(2*pi*as.numeric(hour)/24),
    hour_c = cos(2*pi*as.numeric(hour)/24),
    wday_s = sin(2*pi*as.numeric(wday)/7),
    wday_c = cos(2*pi*as.numeric(wday)/7),
    high_rh   = if ("rh"   %in% names(train)) rh   > median(rh,   na.rm = TRUE) else NA,
    high_temp = if ("temp" %in% names(train)) temp > median(temp, na.rm = TRUE) else NA,
    weather_type = case_when(
      isTRUE(high_rh) & isTRUE(high_temp) ~ "warm_humid",
      isTRUE(high_rh) & !isTRUE(high_temp) ~ "cold_humid",
      !isTRUE(high_rh) & isTRUE(high_temp) ~ "warm_dry",
      TRUE ~ "cold_dry"
    ),
    # >>> WYMUSZAMY STAŁE POZIOMY <<<
    weather_type = factor(weather_type,
                          levels = c("warm_humid","cold_humid","warm_dry","cold_dry"))
  ) |>
  step_rm(hour, wday, high_rh, high_temp) |>
  step_rm(any_of(final_to_drop)) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  # >>> INTERAKCJE ZANIM COKOLWIEK USUNIEMY <<<
  step_interact(terms = ~ starts_with("weather_type_"):matches("^n_\\d+$")) |>
  step_interact(terms = ~ starts_with("weather_type_"):matches("^(ws|mws)$")) |>
  step_YeoJohnson(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  # >>> DOPIERO TERAZ PRUNING <<<
  step_zv(all_predictors())


# RECEPTA 3: INTERAKCJE CZASOWE
rec_svr_temporal <- recipe(grimm_pm10 ~ ., data = train) |>
  update_role(date, new_role = "id") |>
  step_mutate(
    hour_s = sin(2*pi*as.numeric(hour)/24),
    hour_c = cos(2*pi*as.numeric(hour)/24),
    wday_s = sin(2*pi*as.numeric(wday)/7),
    wday_c = cos(2*pi*as.numeric(wday)/7),
    # Pory dnia
    time_of_day = case_when(
      as.numeric(hour) %in% 0:5 ~ "night",
      as.numeric(hour) %in% 6:11 ~ "morning",
      as.numeric(hour) %in% 12:17 ~ "afternoon",
      TRUE ~ "evening"
    ),
    # Weekend
    weekend = as.integer(as.numeric(wday) %in% c(6, 7))
  ) |>
  step_rm(hour, wday) |>
  step_rm(any_of(final_to_drop)) |>
  step_string2factor(time_of_day) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  step_zv(all_predictors()) |>  
  # Interakcje pora dnia x zliczenia
  step_interact(terms = ~ starts_with("time_of_day_"):matches("^n_\\d+$")) |>
  # Interakcje weekend x zliczenia + wiatr
  step_interact(terms = ~ weekend:matches("^n_\\d+$|^(ws|mws)$")) |>
  step_YeoJohnson(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_zv(all_predictors())


# RECEPTA 4: KOMBINOWANE (pogoda + czas + temperatura)
rec_svr_combined <- recipe(grimm_pm10 ~ ., data = train) |>
  update_role(date, new_role = "id") |>
  step_mutate(
    hour_s = sin(2*pi*as.numeric(hour)/24),
    hour_c = cos(2*pi*as.numeric(hour)/24),
    wday_s = sin(2*pi*as.numeric(wday)/7),
    wday_c = cos(2*pi*as.numeric(wday)/7),
    temp_bin = if ("temp" %in% names(train)) cut(
      temp,
      breaks = quantile(temp, probs = seq(0, 1, 0.25), na.rm = TRUE),
      include.lowest = TRUE,
      labels = c("q1","q2","q3","q4")
    ) else NA_character_,
    high_rh = if ("rh" %in% names(train)) rh > median(rh, na.rm = TRUE) else NA,
    weather_humid = if_else(isTRUE(high_rh), "humid", "dry"),
    weekend = as.integer(as.numeric(wday) %in% c(6, 7)),
    # stałe poziomy
    temp_bin = factor(temp_bin, levels = c("q1","q2","q3","q4")),
    weather_humid = factor(weather_humid, levels = c("humid","dry"))
  ) |>
  step_rm(hour, wday, high_rh) |>
  step_rm(any_of(final_to_drop)) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  step_interact(terms = ~ starts_with("temp_bin_"):matches("^n_(0250|0500|1000)$")) |>
  step_interact(terms = ~ starts_with("weather_humid_"):matches("^(ws|mws)$")) |>
  step_interact(terms = ~ weekend:matches("^n_(0250|0500|1000)$")) |>
  step_YeoJohnson(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_zv(all_predictors())


# MODEL SVR I TUNING ------------------------------------------------------------

# Definicja modelu SVR 
svr_spec <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) |>
  set_engine("kernlab") |>
  set_mode("regression")

# Workflow
svr_workflows <- list(
  svr_base = workflow() |> add_recipe(rec_svr_base) |> add_model(svr_spec),
  svr_weather = workflow() |> add_recipe(rec_svr_weather) |> add_model(svr_spec),
  svr_temporal = workflow() |> add_recipe(rec_svr_temporal) |> add_model(svr_spec),
  svr_combined = workflow() |> add_recipe(rec_svr_combined) |> add_model(svr_spec)
)

# Walidacja krzyżowa
set.seed(42)
cv_folds <- vfold_cv(train, v = 5, strata = grimm_pm10)

# Siatka hiperparametrów 
svr_grid <- expand.grid(
  cost = 10^seq(-1, 2, length.out = 4),       
  rbf_sigma = 10^seq(-3, 0, length.out = 4)   
) |> as_tibble()

svr_metrics <- metric_set(rmse, rsq, mae)

cat(paste("\nLiczba kombinacji do przetestowania:", nrow(svr_grid), 
          "\nLiczba przepisów:", length(svr_workflows),
          "\nCałkowita liczba modeli:", nrow(svr_grid) * length(svr_workflows) * 5, "\n"))

# Tuning
cat("\n=== Rozpoczęcie tuningu SVR ===\n")
tic()

svr_tune_results <- imap(
  svr_workflows,
  ~ {
    cat(paste0("\n--- Tuning przepisu: ", .y, " (", which(names(svr_workflows) == .y), "/", 
               length(svr_workflows), ") ---\n"))
    set.seed(42)
    tune_grid(
      .x,
      resamples = cv_folds,
      grid = svr_grid,
      metrics = svr_metrics,
      control = control_grid(verbose = FALSE, save_pred = TRUE)
    )
  }
)

toc()

# WYNIKI I WYBÓR NAJLEPSZEGO MODELU ---------------------------------------------

best_models_summary <- map_dfr(
  svr_tune_results,
  ~ {
    best_rmse <- show_best(.x, metric = "rmse", n = 1)
    best_rsq <- show_best(.x, metric = "rsq", n = 1)
    best_mae <- show_best(.x, metric = "mae", n = 1)
    
    if (nrow(best_rmse) > 0) {
      tibble(
        mean_rmse = best_rmse$mean[1],
        std_err_rmse = best_rmse$std_err[1],
        mean_rsq = if(nrow(best_rsq) > 0) best_rsq$mean[1] else NA_real_,
        mean_mae = if(nrow(best_mae) > 0) best_mae$mean[1] else NA_real_,
        cost = best_rmse$cost[1],
        rbf_sigma = best_rmse$rbf_sigma[1]
      )
    } else {
      tibble(
        mean_rmse = NA_real_, 
        std_err_rmse = NA_real_,
        mean_rsq = NA_real_,
        mean_mae = NA_real_,
        cost = NA_real_, 
        rbf_sigma = NA_real_
      )
    }
  },
  .id = "recipe"
) |>
  mutate(recipe = str_remove(recipe, "svr_")) |>
  filter(!is.na(mean_rmse)) |>
  arrange(mean_rmse)

cat("\n=== PODSUMOWANIE NAJLEPSZYCH MODELI ===\n")
print(best_models_summary, n = Inf)

# Wybór najlepszego modelu
best_recipe_name <- best_models_summary$recipe[1]
best_full_name <- paste0("svr_", best_recipe_name)
best_results <- svr_tune_results[[best_full_name]]
best_svr_params <- select_best(best_results, metric = "rmse")

cat(paste0("\n✓ NAJLEPSZY MODEL: ", best_full_name, "\n"))
cat("\nParametry:\n")
print(best_svr_params)

# Finalizacja workflow
best_svr_wf <- svr_workflows[[best_full_name]] |>
  finalize_workflow(best_svr_params)

cat("\n=== TRENING FINALNEGO MODELU ===\n")
tic()
final_svr_fit <- fit(best_svr_wf, data = train)
toc()

# Utworzenie folderu i zapis wyników
if (!dir.exists("wyniki")) {
  dir.create("wyniki", recursive = TRUE)
}

saveRDS(svr_tune_results, "wyniki/svr_tune_results.rds")
saveRDS(final_svr_fit, "wyniki/final_svr_model.rds")
saveRDS(best_models_summary, "wyniki/best_models_summary.rds")

cat("\n✓ Zakończono proces modelowania SVR\n")
cat("✓ Wyniki zapisane w folderze 'wyniki/'\n")


#Najlepsze wyniki modelu daje przepis "combined" z interakcjami temperatury, wilgotności i weekendu.
#Hiperparametry dla najlepszego modelu przyjmują wartości cost = 10 i rbf_sigma = 0.01.



