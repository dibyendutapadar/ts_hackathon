import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1️⃣ Identify Drivers via Sensitivity & Correlation Analysis
def identify_drivers(df, target_col, max_lag=3):
    df = df.pivot(index="Month", columns="Item", values="Value").dropna()
    
    # Generate lag features
    for col in df.columns:
        if col != target_col:
            for lag in range(1, max_lag + 1):
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    df.dropna(inplace=True)

    # Compute correlation & mutual information scores
    correlations = df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
    mi_scores = mutual_info_regression(df.drop(columns=[target_col]), df[target_col])

    # Rank features
    driver_scores = pd.DataFrame({"Feature": correlations.index, "Correlation": correlations.values, "Mutual Info": mi_scores}).sort_values(by="Correlation", ascending=False)

    return df, driver_scores

# 2️⃣ Auto-Select Best Time-Series Model
def best_time_series_model(train, test, target_col):
    models = {
        "ARIMA": ARIMA(train[target_col], order=(1, 1, 1)).fit(),
        "Holt-Winters": ExponentialSmoothing(train[target_col], trend="add", seasonal="add", seasonal_periods=12).fit(),
        "SARIMA": SARIMAX(train[target_col], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
    }

    best_model = None
    best_rmse = float("inf")

    for name, model in models.items():
        forecast = model.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test[target_col], forecast))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    return best_model

# 3️⃣ Train ML Model for Residuals
def train_residual_model(X_train, y_train, X_test, method="RandomForest"):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    }

    model = models[method]

    if method == "XGBoost":
        param_grid = {"max_depth": [3, 5, 7], "n_estimators": [50, 100, 200]}
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_squared_error")
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)

    return model.predict(X_test)

# 4️⃣ Hybrid Forecast (Time-Series + ML Residuals)
def hybrid_forecast(df, target_col, top_n_drivers=3, ml_method="RandomForest"):
    full_data, driver_scores = identify_drivers(df, target_col)
    top_drivers = driver_scores.head(top_n_drivers)["Feature"].tolist()

    train = full_data.iloc[:-6]
    test = full_data.iloc[-6:]

    best_ts_model = best_time_series_model(train, test, target_col)
    ts_forecast = best_ts_model.forecast(steps=len(test))

    residuals = train[target_col] - best_ts_model.fittedvalues
    X_train = train[top_drivers]
    X_test = test[top_drivers]

    residual_preds = train_residual_model(X_train, residuals, X_test, ml_method)

    hybrid_forecast = ts_forecast + residual_preds

    metrics = {
        "Baseline MAE": mean_absolute_error(test[target_col], ts_forecast),
        "Hybrid MAE": mean_absolute_error(test[target_col], hybrid_forecast),
        "Baseline RMSE": np.sqrt(mean_squared_error(test[target_col], ts_forecast)),
        "Hybrid RMSE": np.sqrt(mean_squared_error(test[target_col], hybrid_forecast))
    }

    return train, test, ts_forecast, hybrid_forecast, metrics, driver_scores
