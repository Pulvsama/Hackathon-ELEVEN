# Hackathon-ELEVEN
ML project (ELEVEN Strategy) predicting short-term wait times in amusement parks

## Project Overview
This project predicts waiting times in amusement parks over the next 2 hours using the dataset ELEVEN Strategy provided us with. Developed as part of a hackathon, the project demonstrates data preprocessing, feature engineering, model selection, and evaluation. The project won the hackathon for its predictive performance and modeling approach.

## Data
- Dataset: Eleven Strategy original datasets of weather data and waiting times (for the training, the validation adn the test) and dataset added by ourselves to better predict waiting times
- Features: 
  - Eleven Dataset : 
    - Time-related features: Datetime, Time to events
    - Attraction metadata: Attraction Name, Capacity
    - Operational status: Downtime, Current Wait Time
    - Weather features: Temperature, Dew Point, Felt Temperature, Pressure, Humidity, Wind Speed, Rain, Snow, Cloud Coverage
    - Target feature : Wait Time in 2 hours
  - Our dataset :
    - Holidays period
    - Covid Periods
- Preprocessing: Encoded categorical variables (One Hot Encoding), Added features from our datasets, Added Cyclical time features (sin/cos transformation on Hour, Day of Week, and Month)

## Approach
- Models used: Random Forest, CatBoost, XGBoost  
- Data preprocessing and lagged features to improve model performance  
- Hyperparameter tuning with cross-validation using Optuna

## Results
- RMSE decreased from **13.4** (baseline) to **6.9** (final model)
- Final model achieved **winning performance in the hackathon**