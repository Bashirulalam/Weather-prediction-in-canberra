###import libraries

library(dplyr)
library(tidyr) 
library(ggplot2)
library(caret)

###load data

df_test <- read.csv("Weather Test Data.csv")
df_train <- read.csv("Weather Training Data.csv")

head(df_test)
head(df_train)


## Slelecting the data only for canberra city

canb_test <- df_test %>%
  filter(Location == "Canberra")

canb_train <- df_train %>%
  filter(Location == "Canberra")

#### feature selecting


tr_df <- canb_train %>%
  select(-row.ID, -Location, -WindGustDir, -WindGustSpeed, -WindDir9am, -WindDir3pm,
         -Temp9am, -Temp3pm)

ts_df <- canb_test %>%
  select(-row.ID, -Location, -WindGustDir, -WindGustSpeed, -WindDir9am, -WindDir3pm,
         -Temp9am, -Temp3pm)

# Now we will take the average of the features



train <- tr_df %>%
  mutate(
    TempAvg = rowMeans(select(., MaxTemp, MinTemp), na.rm = TRUE),
    WindSpeedAvg = rowMeans(select(., WindSpeed3pm, WindSpeed9am), na.rm = TRUE),
    HumidityAvg = rowMeans(select(., Humidity3pm, Humidity9am), na.rm = TRUE),
    PressureAvg = rowMeans(select(., Pressure3pm, Pressure9am), na.rm = TRUE),
    CloudAvg = rowMeans(select(., Cloud3pm, Cloud9am), na.rm = TRUE)
  )

test <- ts_df %>%
  mutate(
    TempAvg = rowMeans(select(., MaxTemp, MinTemp), na.rm = TRUE),
    WindSpeedAvg = rowMeans(select(., WindSpeed3pm, WindSpeed9am), na.rm = TRUE),
    HumidityAvg = rowMeans(select(., Humidity3pm, Humidity9am), na.rm = TRUE),
    PressureAvg = rowMeans(select(., Pressure3pm, Pressure9am), na.rm = TRUE),
    CloudAvg = rowMeans(select(., Cloud3pm, Cloud9am), na.rm = TRUE)
  )

train <- train %>%
  select(-MaxTemp, -MinTemp, -WindSpeed3pm, -WindSpeed9am, -Humidity3pm, -Humidity9am,
         -Pressure3pm, -Pressure9am, -Cloud3pm, -Cloud9am, -CloudAvg)

test <- test %>%
  select(-MaxTemp, -MinTemp, -WindSpeed3pm, -WindSpeed9am, -Humidity3pm, -Humidity9am,
         -Pressure3pm, -Pressure9am, -Cloud3pm, -Cloud9am, -CloudAvg)


dim(train)
dim(test)

### Explanatory data analysis

##Average temp when rain vs not rain.

##check for the missing value
colSums(is.na(test))


train <- train %>%
  select( -CloudAvg)

test <- test %>%
  select( -CloudAvg )




train <- train %>% 
  drop_na(WindSpeedAvg, PressureAvg, Rainfall, RainToday, HumidityAvg)
test  <- test %>% 
  drop_na(WindSpeedAvg, PressureAvg, Rainfall, RainToday)

dim(train)
dim(test)


num_train <- train %>% 
  select(where(is.numeric))

cor_mat <- cor(num_train, use = "pairwise.complete.obs")

heatmap(cor_mat)





# reshape selected features into long format
df_long <- train %>%
  select(RainTomorrow, TempAvg, WindSpeedAvg, HumidityAvg, PressureAvg) %>%
  pivot_longer(cols = c(TempAvg, WindSpeedAvg, HumidityAvg, PressureAvg),
               names_to = "Feature", values_to = "Value")

# plot with facets
ggplot(df_long, aes(x = RainTomorrow, y = Value, fill = RainTomorrow)) +
  geom_boxplot() +
  facet_wrap(~ Feature, scales = "free_y") +
  labs(title = "Boxplots of Features vs RainTomorrow",
       x = "Rain Tomorrow", y = "Value") +
  theme_bw()



### data distribution of the variables

# Select numeric variables only
num_vars <- train %>% select(where(is.numeric))

# Reshape to long format
df_long <- num_vars %>%
  pivot_longer(cols = everything(), names_to = "Feature", values_to = "Value")

# Plot histograms for all numeric features
ggplot(df_long, aes(x = Value)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  facet_wrap(~ Feature, scales = "free") +
  labs(title = "Distributions of Numeric Features", x = "Value", y = "Count") +
  theme_bw()

  

############################################################################################

##Modeling tempAvg (linear regression model)

model_temp <- lm(TempAvg ~ HumidityAvg + WindSpeedAvg + PressureAvg + Rainfall,
                 data = train)

summary(model_temp)

# Predictions
pred_temp <- predict(model_temp, newdata = test)


####### evalutaing the model

actual_temp <- test$TempAvg

# RMSE
rmse <- sqrt(mean((pred_temp - actual_temp)^2, na.rm = TRUE))

# MAE
mae <- mean(abs(pred_temp - actual_temp), na.rm = TRUE)

# R-squared
SSE <- sum((pred_temp - actual_temp)^2, na.rm = TRUE)
SST <- sum((actual_temp - mean(actual_temp, na.rm = TRUE))^2, na.rm = TRUE)
rsq <- 1 - SSE/SST

list(RMSE = rmse, MAE = mae, R2 = rsq)


########### modeling the rainfall today(logistic regression)

train$RainToday <- as.factor(train$RainToday)
test$RainToday  <- as.factor(test$RainToday)


# Fit logistic regression for RainToday
log_model_today <- glm(RainToday ~ TempAvg + WindSpeedAvg + HumidityAvg + PressureAvg,
                       data = train,
                       family = binomial)

summary(log_model_today)

# Predict probabilities
pred_prob <- predict(log_model_today, newdata = test, type = "response")

# Convert to Yes/No with threshold 0.5
pred_class <- ifelse(pred_prob > 0.5, "Yes", "No")

confusionMatrix(as.factor(pred_class), test$RainToday, positive = "Yes")