# Model XGB ---------------------------------------------------------------

# Przepisy 

library(tidyverse)
library(tidymodels)
library(lubridate)
library(knitr)
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

# 2 - przepis kategotie + interakcje

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
  step_mutate(weekend = as.integer(weekend))
    
# pozniej zeby progi liczyc na train: step_discretize(temp, num_breaks = 4, keep_original_cols = T)

# warunkowe interakcje bo był problem z pustymi miejscami
has_wt   <- any(startsWith(colnames(train), "weather_type")) || T  
has_tod  <- any(startsWith(colnames(train), "time_of_day"))  || T
has_cv_wv <- (length(cv) + length(wv)) > 0
has_temp  <- "temp" %in% names(train)

if (has_cv_wv || has_temp) {
  rec_time_weather_xgb <- rec_time_weather_xgb |>
    step_interact(terms = ~ starts_with("weather_type"):any_of(c(cv, wv, "temp")))}

if (has_cv_wv) {
  rec_time_weather_xgb <- rec_time_weather_xgb |>
    step_interact(terms = ~ starts_with("time_of_day"):any_of(c(cv, wv))) |>
    step_interact(terms = ~ weekend:any_of(c(cv, wv)))}

rec_time_weather_xgb <- rec_time_weather_xgb |>
  step_YeoJohnson(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_zv(all_predictors())

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


