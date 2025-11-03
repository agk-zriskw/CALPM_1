## -------------------------------------------------------------------------------------------------------------------------------------
library(tidyverse)
library(rsample)
library(recipes)
library(corrplot)
library(e1071)


## -------------------------------------------------------------------------------------------------------------------------------------
load("dane/data_test.rdata")


## -------------------------------------------------------------------------------------------------------------------------------------
print("Brakujące wartości:")
print(colSums(is.na(ops_data)))
print("Podsumowanie:")
print(summary(ops_data))
glimpse(ops_data)


## -------------------------------------------------------------------------------------------------------------------------------------
set.seed(42)
split <- initial_time_split(ops_data %>% select(-poj_h), prop = 0.7)
train <- training(split)
test <- testing(split)


## -------------------------------------------------------------------------------------------------------------------------------------
# Identyfikacja wartości odstających za pomocą rozkładu międzykwartylowego
detect_outliers <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower <- q1 - 3 * iqr  
  upper <- q3 + 3 * iqr # Konserwatywne przyjęcie
  
  n_lower <- sum(x < lower, na.rm = TRUE)
  n_upper <- sum(x > upper, na.rm = TRUE)
  pct_outliers <- (n_lower + n_upper) / length(x) * 100
  
  tibble(lower = lower, upper = upper, 
         n_lower = n_lower, n_upper = n_upper,
         pct_outliers = round(pct_outliers, 2))
}


numeric_cols <- train %>% select(where(is.numeric), -wd) %>% names()

outlier_summary <- train %>%
  summarise(across(all_of(numeric_cols), detect_outliers)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "stats") %>%
  unnest(stats) %>%
  arrange(desc(pct_outliers))

print("Wartości odstające")
print(outlier_summary %>% filter(pct_outliers > 0))

vars_to_winsorize <- outlier_summary %>%
  pull(variable)




## -------------------------------------------------------------------------------------------------------------------------------------
numeric_cols <- train %>% select(where(is.numeric), -wd) %>% names()

skew_tbl <- train %>%
  summarise(across(all_of(numeric_cols), ~skewness(.x, na.rm = TRUE, type = 2))) %>%
  pivot_longer(everything(), names_to = "Zmienne", values_to = "Skosnosc") %>%
  arrange(desc(abs(Skosnosc)))

print(skew_tbl)




## -------------------------------------------------------------------------------------------------------------------------------------
zliczenia <- train %>% select(starts_with("n_")) %>% names()
target <- "grimm_pm10"
thr <- 0.9

# Zliczenia oraz zmienne dotyczące wiatru
preds <- c(zliczenia, "ws", "mws") %>% .[. %in% names(train)]

# Korelacja
cor_mat <- cor(train[, preds], use = "pairwise.complete.obs", method = "spearman")
tgt_cor <- abs(cor(train[, preds], train[[target]], use = "pairwise.complete.obs", method = "spearman"))

# Identyfikacja silnie skorelowanych zmiennych
pairs_df <- as.data.frame(as.table(cor_mat)) %>%
  as_tibble() %>%
  transmute(var1 = as.character(Var1), var2 = as.character(Var2), corr = Freq) %>%
  filter(var1 != var2) %>%
  mutate(pair = paste(pmin(var1, var2), pmax(var1, var2), sep = "__")) %>%
  distinct(pair, .keep_all = TRUE) %>%
  mutate(abs_corr = abs(corr),
         tgt1 = tgt_cor[var1, 1], 
         tgt2 = tgt_cor[var2, 1],
         drop_candidate = if_else(tgt1 <= tgt2, var1, var2)) %>%
  filter(abs_corr > thr) %>%
  arrange(desc(abs_corr))


to_drop <- unique(pairs_df$drop_candidate)

print(to_drop)
print(pairs_df)



## -------------------------------------------------------------------------------------------------------------------------------------

# Granice winsoryzacji
winsor_limits <- train %>%
  summarise(across(all_of(vars_to_winsorize), 
                   ~list(quantile(.x, c(0.01, 0.99), na.rm = TRUE)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "limits")

# Winsoryzacja
for (var in vars_to_winsorize) {
  limits <- quantile(train[[var]], c(0.01, 0.99), na.rm = TRUE)
  train[[var]] <- pmin(pmax(train[[var]], limits[1]), limits[2])
  test[[var]] <- pmin(pmax(test[[var]], limits[1]), limits[2])
}
  

rec <- recipe(grimm_pm10 ~ ., data = train) %>%
  update_role(date, new_role = "id") %>%
  step_mutate(
    hour_s = sin(2*pi*as.numeric(hour)/24),
    hour_c = cos(2*pi*as.numeric(hour)/24),
    wday_s = sin(2*pi*as.numeric(wday)/7),
    wday_c = cos(2*pi*as.numeric(wday)/7)
  ) %>%
  step_rm(hour, wday)

rec <- rec %>% step_rm(all_of(to_drop))

rec <- rec %>%
  step_YeoJohnson(all_numeric_predictors()) %>%  
  step_normalize(all_numeric_predictors()) %>%   
  step_zv(all_predictors())

rec_prep <- prep(rec, training = train)
train_tr <- bake(rec_prep, new_data = NULL)
test_tr <- bake(rec_prep, new_data = test)



## -------------------------------------------------------------------------------------------------------------------------------------
ggplot(train_tr, aes(grimm_pm10)) +
  geom_histogram(bins = 40, fill = "steelblue", alpha = 0.7) +
  labs(title = "Rozkład zmiennej objaśnianej", x = "grimm_pm10") +
  theme_minimal()


## -------------------------------------------------------------------------------------------------------------------------------------
cor_mat_tr <- train_tr %>%
  select(where(is.numeric)) %>%
  cor(use = "pairwise.complete.obs", method = "spearman")

corrplot(cor_mat_tr, type = "lower", tl.cex = 0.6, mar = c(0,0,1,0),
         addCoef.col = "black", number.cex = 0.6, number.digits = 2)


## -------------------------------------------------------------------------------------------------------------------------------------
num_pred <- train_tr %>% select(where(is.numeric), -grimm_pm10, -date) %>% names()
tgt_cor_tr <- abs(cor(train_tr[, num_pred], train_tr$grimm_pm10, use = "pairwise.complete.obs"))
top6 <- names(sort(tgt_cor_tr[,1], decreasing = TRUE)[1:6])

train_tr %>%
  select(grimm_pm10, all_of(top6)) %>%
  pivot_longer(all_of(top6), names_to = "feature", values_to = "value") %>%
  ggplot(aes(value, grimm_pm10)) +
  geom_point(alpha = 0.2) +
  geom_smooth(se = FALSE, color = "red") +
  facet_wrap(~ feature, scales = "free_x") +
  labs(title = "Target vs Top Correlated Features", x = NULL) +
  theme_minimal()


## -------------------------------------------------------------------------------------------------------------------------------------
train %>%
  ggplot(aes(date, grimm_pm10)) +
  geom_line(alpha = 0.7, color = "steelblue") +
  labs(title = "Zmienna objaśniana") +
  theme_minimal()


## -------------------------------------------------------------------------------------------------------------------------------------
train_int <- train %>%
  mutate(temp_bin = cut(temp, breaks = quantile(temp, probs = seq(0, 1, 0.25), na.rm = TRUE),
                        include.lowest = TRUE, labels = c("Q1", "Q2", "Q3", "Q4")))

ggplot(train_int, aes(ws, grimm_pm10, color = temp_bin)) +
  geom_point(alpha = 0.3) +
  geom_smooth(se = FALSE, method = "loess") +
  labs(title = "Prędkość wiatru a PM10 według kwartyli temperatury",
       subtitle = "Różne nachylenia sugerują interakcję temp:ws") +
  theme_minimal()


## -------------------------------------------------------------------------------------------------------------------------------------
train_weather <- train %>%
  mutate(
    high_rh = rh > median(rh, na.rm = TRUE),
    high_temp = temp > median(temp, na.rm = TRUE),
    weather_type = case_when(
      high_rh & high_temp ~ "Ciepło i Wilgotno",
      high_rh & !high_temp ~ "Zimno i Wilgotno",
      !high_rh & high_temp ~ "Ciepło i Sucho",
      TRUE ~ "Zimno i Sucho"
    )
  )

ggplot(train_weather, aes(n_0500, grimm_pm10, color = weather_type)) +
  geom_point(alpha = 0.3) +
  geom_smooth(se = FALSE, method = "lm") +
  labs(title = "Zliczenia cząsteczek vs PM10 z uwzględnieniem temperatury i wilgotności") +
  theme_minimal() +
  facet_wrap(~weather_type)


## -------------------------------------------------------------------------------------------------------------------------------------
# Check if relationships vary by time of day or day of week
train_time <- train %>%
  mutate(
    hour_cat = case_when(
      hour %in% 0:5 ~ "Noc",
      hour %in% 6:11 ~ "Poranek",
      hour %in% 12:17 ~ "Popołudnie",
      TRUE ~ "Wieczór"
    ),
    weekend = wday %in% c("Sat", "Sun")
  )

ggplot(train_time, aes(n_0500, grimm_pm10, color = hour_cat)) +
  geom_point(alpha = 0.2) +
  geom_smooth(se = FALSE) +
  labs(title = "Particle Count vs PM10 by Time of Day") +
  theme_minimal()


## -------------------------------------------------------------------------------------------------------------------------------------
rec <- recipe(grimm_pm10 ~ ., data = train) %>%
  update_role(date, new_role = "id") %>%
  step_mutate(
    # Obsługa dni tygodnia i godzin
    hour_s = sin(2*pi*as.numeric(hour)/24),
    hour_c = cos(2*pi*as.numeric(hour)/24),
    wday_s = sin(2*pi*as.numeric(wday)/7),
    wday_c = cos(2*pi*as.numeric(wday)/7),
    # Podział temperatury
    temp_bin = cut(temp, 
                   breaks = quantile(temp, probs = seq(0, 1, 0.25), na.rm = TRUE),
                   include.lowest = TRUE, labels = c("Q1", "Q2", "Q3", "Q4")),
    # Typ pogody
    high_rh = rh > median(rh, na.rm = TRUE),
    high_temp = temp > median(temp, na.rm = TRUE),
    weather_type = case_when(
      high_rh & high_temp ~ "warm_humid",
      high_rh & !high_temp ~ "cool_humid",
      !high_rh & high_temp ~ "warm_dry",
      TRUE ~ "cool_dry"
    ),
    # Podział dnia
    hour_cat = case_when(
      as.numeric(hour) %in% 0:5 ~ "night",
      as.numeric(hour) %in% 6:11 ~ "morning",
      as.numeric(hour) %in% 12:17 ~ "afternoon",
      TRUE ~ "evening"
    ),
    weekend = wday %in% c("Sat", "Sun")
  ) %>%
  step_string2factor(weather_type, hour_cat) %>%
  step_rm(hour, wday, high_rh, high_temp) 


  rec <- rec %>% step_rm(all_of(to_drop))


# Interakcje
rec <- rec %>%
  step_dummy(all_nominal_predictors()) %>%  
  step_interact(~ starts_with("weather_type"):matches("n_0500|n_0250|ws|temp")) %>%  
  step_interact(~ starts_with("temp_bin"):matches("n_0500|n_0250|ws")) %>%  
  step_interact(~ starts_with("hour_cat"):matches("n_0500|n_0250|ws")) %>%  
  step_interact(~ weekend:matches("n_0500|n_0250|ws")) %>%  
  step_YeoJohnson(all_numeric_predictors()) %>%  
  step_normalize(all_numeric_predictors()) %>%   
  step_zv(all_predictors())  

rec_prep <- prep(rec, training = train)
train_tr <- bake(rec_prep, new_data = NULL)
test_tr <- bake(rec_prep, new_data = test)



