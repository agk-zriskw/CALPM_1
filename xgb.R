# Model XGB ---------------------------------------------------------------

# Przepisy ----------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(lubridate)
library(knitr)
library(xgboost)
library(vip)
library(workflowsets)
library(purrr)
tidymodels_prefer()
set.seed(123)

# wczytanie danych 
if (!exists("train") || !exists("test")) {
  message("Pobieram dane z Kalibracja_quarto.qmd")
  knitr::purl("Kalibracja_quarto.qmd", output = "dane_po_kalibracji.R", quiet = TRUE)
  source("dane_po_kalibracji.R")
} else {
  message("Dane są już wczytane")
}

#print(head(train))
#print(head(test))

# 1 - przepis bazowy

rec_base_xgb <- recipe(grimm_pm10 ~ ., data = train) |> 
    update_role(date, new_role = "id") |> 
    step_mutate(
      hour_s = sin(2*pi*as.numeric(hour)/24),
      hour_c = cos(2*pi*as.numeric(hour)/24),
      wday_s = sin(2*pi*as.numeric(wday)/7),
      wday_c = cos(2*pi*as.numeric(wday)/7)) |> 
    step_rm(hour, wday) # niepotrzebne
  
# usuwam bardzo skoreloawne zmienne z listy
if (exists("to_drop", inherits = T) && length(intersect(to_drop, names(train))) > 0) {
  rec_base_xgb <- rec_base_xgb |>  step_rm(all_of(intersect(to_drop, names(train))))}
  
# transformacja, normalizacja, stale kolumny
rec_base_xgb <- rec_base_xgb |>
  step_YeoJohnson(all_numeric_predictors()) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_zv(all_predictors())

# 2 - przepis kategotie + interakcje automatyczne

cv <- names(train) |> keep(~ str_starts(.x, "n_"))
wv <- names(train) |> keep(~ .x %in% c("ws","mws"))

rec_time_weather_xgb <- recipe(grimm_pm10 ~ ., data = train) |> 
  update_role(date, new_role = "id") |> 
  step_mutate(
      
    # temp na kwartyle
    temp_bin = if("temp" %in% names(train)) cut(
      temp,
      breaks = quantile(temp, probs = seq(0,1,0.25), na.rm = T),
      include.lowest = T,
      labels = c("q1","q2","q3","q4"))
    else NA_character_,
      
    # jaka wilgotnosc i temp po medianie
    high_rh = if("rh" %in% names(train)) rh > median(rh, na.rm = T) else NA,
    high_temp = if ("temp" %in% names(train)) temp > median(temp, na.rm = T) else NA,
      
    # typy pogody
    weather_type = case_when(
      isTRUE(high_rh) & isTRUE(high_temp) ~ "warm_humid",
      isTRUE(high_rh) & !isTRUE(high_temp) ~ "cold_humid",
      !isTRUE(high_rh) & isTRUE(high_temp) ~ "warm_dry",
      T ~ "cold_dry"),
      
    # pory dnia
    time_of_day = case_when(
      as.numeric(hour) %in% 0:5 ~ "night",
      as.numeric(hour) %in% 6:11 ~ "morning",
      as.numeric(hour) %in% 12:17 ~ "afternoon",
      T ~ "evening"),
    
    weekend = as.numeric(wday) %in% c(6,7),
    hour_s = sin(2*pi*as.numeric(hour)/24),
    hour_c = cos(2*pi*as.numeric(hour)/24),
    wday_s = sin(2*pi*as.numeric(wday)/7),
    wday_c = cos(2*pi*as.numeric(wday)/7)) |> 
    step_rm(high_rh, high_temp, hour, wday) |> 
    step_string2factor(weather_type, time_of_day) |> 
    step_zv(all_nominal_predictors()) |> 
    step_dummy(all_nominal_predictors()) |> 
    step_mutate(weekend = as.integer(weekend)) |> 
    step_YeoJohnson(all_numeric_predictors()) |>
    step_normalize(all_numeric_predictors()) |>
    step_zv(all_predictors())  
# pozniej zeby progi liczyc na train: step_discretize(temp, num_breaks = 4, keep_original_cols = T)

# 3 - przepis z PCA (model ma mniej zmiennych do analizy,
#     zachowuje najwazniejsze "n_"
  
rec_pca_xgb <-  recipe(grimm_pm10 ~ ., data = train) |> 
  update_role(date, new_role = "id") |> 
  step_mutate(
    hour_s = sin(2*pi*as.numeric(hour)/24),
    hour_c = cos(2*pi*as.numeric(hour)/24),
    wday_s = sin(2*pi*as.numeric(wday)/7),
    wday_c = cos(2*pi*as.numeric(wday)/7)) |> 
  step_rm(hour, wday) |> 
  step_YeoJohnson(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_pca(all_of(cv), num_comp = 10) |> # 10 na test, trzeba dostroic
  step_zv(all_predictors())

# test 

cat("\n================ BASE XGB ================\n")
rec_base_prep <- prep(rec_base_xgb)
train_base <- juice(rec_base_prep)
test_base  <- bake(rec_base_prep, new_data = test)
cat("Wymiary train:", dim(train_base)[1], "x", dim(train_base)[2], "\n")
glimpse(train_base)

cat("\n=========== TIME-WEATHER XGB ===========\n")
rec_tw_prep <- prep(rec_time_weather_xgb)
train_tw <- juice(rec_tw_prep)
test_tw  <- bake(rec_tw_prep, new_data = test)
cat("Wymiary train:", dim(train_tw)[1], "x", dim(train_tw)[2], "\n")
glimpse(train_tw)

cat("\n================ PCA XGB =================\n")
rec_pca_prep <- prep(rec_pca_xgb)
train_pca <- juice(rec_pca_prep)
test_pca  <- bake(rec_pca_prep, new_data = test)
cat("Wymiary train:", dim(train_pca)[1], "x", dim(train_pca)[2], "\n")
glimpse(train_pca)


# Definicja modelu --------------------------------------------------------

xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  min_n = tune(),
  learn_rate = tune(),
  mtry = tune(),
  loss_reduction = tune()) |> 
  set_engine("xgboost") |> 
  set_mode("regression")


# Walidacja krzyżowa ------------------------------------------------------

set.seed(42)
cv_folds <- vfold_cv(train, v = 5)


# Workflow ----------------------------------------------------------------

recipe_list <- list(
  base = rec_base_xgb,
  features = rec_time_weather_xgb,
  pca = rec_pca_xgb)

wf_set <- workflow_set(
  preproc = recipe_list,
  models = list(xgb = xgb_spec),
  cross = T)

print(wf_set)

# Tuning ------------------------------------------------------------------

# ile predyktorów zostaje dla każdego recipe
count_predictors_after_prep <- function(rec, training_data) {
  rp <- tryCatch(prep(rec, training = training_data), error = function(e) e)
  if (inherits(rp, "error")) {
    warning("Prep failed for a recipe: ", conditionMessage(rp))
    return(0)
  }
  max(0, ncol(juice(rp)) - 1)  # -1 bo kolumna celu
}

pred_counts <- map_int(recipe_list, ~ count_predictors_after_prep(.x, training = train))
print(pred_counts)  

min_preds <- min(pred_counts[pred_counts > 0], na.rm = TRUE)
if (is.na(min_preds) || min_preds <= 0) {
  stop("Nie udało się policzyć liczby predyktorów dla żadnego recipe.")
}

# bezpieczny zakres mtry
mtry_upper <- min(5, as.integer(min_preds))
mtry_lower <- 1

message("Ustawiam mtry range: ", mtry_lower, " - ", mtry_upper)

set.seed(123)
xgb_grid <- grid_latin_hypercube(
  trees(range = c(250, 1500)),
  tree_depth(range = c(3, 10)),
  min_n(range = c(10, 50)),
  learn_rate(range = c(-2.5, -1)),  
  mtry(range = c(mtry_lower, mtry_upper)),
  loss_reduction(range = c(0,10)),
  size = 50
)

set.seed(123)
tune_results <- wf_set |> 
  workflow_map(
    "tune_grid",
    resamples = cv_folds,
    grid = xgb_grid,
    metrics = metric_set(rmse, mae),
    control = control_grid(save_pred = TRUE, verbose = TRUE)
  )

# Ewaluacja ---------------------------------------------------------------

print(autoplot(tune_results))
ranking <- rank_results(tune_results, rank_metric = "rmse", select_best = T)
print(ranking)

best_combo <- tune_results |>  #combo = przepis i parametry
  rank_results(rank_metric = "rmse") |> 
  filter(.metric == "rmse") |> 
  slice_head(n = 1)

cat("\n Najlepsza kombinacja \n")
print(best_combo)

best_wf_id <- best_combo$wflow_id[1]

best_wf <- tune_results |> 
  extract_workflow(best_wf_id)

best_params <- tune_results |> 
  extract_workflow_set_result(best_wf_id) |> 
  select_best(metric = "rmse")

cat("\n Najlepsze parametry \n")
print(best_params)

# Finalizacja -------------------------------------------------------------

final_xgb_wf <- finalize_workflow(best_wf, best_params)

cat("\n Finalny workflow \n")
print(final_xgb_wf)

final_fit <- last_fit(final_xgb_wf, split = split)

cat("\n Metryki na zbiorze testowym \n")
print(collect_metrics(final_fit))

test_pred <- collect_predictions(final_fit)

pred_plot <- ggplot(test_pred, aes(x = .pred, y= grimm_pm10)) +
  geom_point(alpha = 0.5, color = "pink2") +
  geom_abline(color = "purple", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Wartości obserwowane a predykcja na zbiorze testowym",
    x = "Predykcje",
    y = "Wartości obserwowane") +
  theme_minimal() +
  coord_fixed()

print(pred_plot)

wf_fitted <- final_fit$.workflow[[1]]

fit_parsnip <- tryCatch(
  extract_fit_parsnip(wf_fitted),
  error = function(e) {
    message("extract_fit_parsnip nie zadziałał, stosuję pull_workflow_fit()")
    pull_workflow_fit(wf_fitted)})

xgb_fit_engine <- fit_parsnip$fit

vip_plot <- vip(xgb_fit_engine, num_features = 20) +
  ggtitle("Najważniejsze zmienne w modelu XGB")

print(vip_plot)

# Wyniki ------------------------------------------------------------------

if (!dir.exists("wyniki_xgb")) {
  dir.create("wyniki_xgb", recursive = TRUE)}

metryki_testowe <- collect_metrics(final_fit)
write_csv(metryki_testowe, "wyniki_xgb/metryki_testowe.csv")
write_csv(best_params, "wyniki_xgb/najlepsze_hiperparametry.csv")

final_workflow_fitted <- extract_workflow(final_fit)
saveRDS(final_workflow_fitted, "wyniki_xgb/final_xgb_workflow.rds")


