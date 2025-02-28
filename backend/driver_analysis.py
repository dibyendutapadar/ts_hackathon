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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ast
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
import shap
from scipy.stats import pearsonr
import openai
import os
import json
from pydantic import BaseModel, Field


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 1️⃣ Create Lagged Features & Moving Averages
def create_lagged_features(df, target_col, include_items, max_lag=3, ma_windows=[3, 6]):
    """
    Creates lagged features and moving averages for selected drivers.
    """
    df = df.pivot(index="Month", columns="Item", values="Value")
    df.columns = df.columns.str.strip()
    target_col = target_col.strip()
    include_items = [item.strip() for item in include_items]
    
    df = df[[col for col in df.columns if col in include_items or col == target_col]]
    
    feature_cols = []
    for col in df.columns:
        if col != target_col:
            # Add lag features
            for lag in range(1, max_lag + 1):
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
                feature_cols.append(f"{col}_lag{lag}")

            # Add moving averages
            for window in ma_windows:
                df[f"{col}_ma{window}"] = df[col].rolling(window=window).mean()
                feature_cols.append(f"{col}_ma{window}")
    
    df.dropna(inplace=True)
    return df, feature_cols

# 2️⃣ Cross-Correlation Function (CCF) for Lag Selection
def cross_correlation_lag(df, target_col, max_lag=6):
    """
    Determines the best lag for each feature using cross-correlation.
    """
    best_lags = {}
    for col in df.columns:
        if col != target_col:
            best_corr = 0
            best_lag = 0
            for lag in range(1, max_lag + 1):
                shifted = df[col].shift(lag).dropna()
                target_shifted = df[target_col].iloc[len(df) - len(shifted):]
                if len(shifted) > 10:
                    corr, _ = pearsonr(shifted, target_shifted)
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
            if best_lag > 0:
                best_lags[col] = best_lag
    return best_lags

# 3️⃣ Feature Selection using Mutual Information & Lasso
def select_features(df, target_col, feature_cols):
    """
    Selects important features using Mutual Information and Lasso regression.
    """
    X = df[feature_cols]
    y = df[target_col]

    print("----+++++----")
    print(X)
    print(y)
    print("----+++++----")

    mi_scores = mutual_info_regression(X, y)
    lasso = LassoCV(cv=5).fit(X, y)
    lasso_selected = X.columns[lasso.coef_ != 0]
    selected_features = list(set(lasso_selected) | set(X.columns[np.argsort(mi_scores)[-10:]]))
    return selected_features

# 4️⃣ Compute SHAP Values using XGBoost
def shap_feature_importance(df, target_col, selected_features):
    """
    Trains an XGBoost model and computes SHAP values to identify key drivers.
    """
    X = df[selected_features]
    y = df[target_col]
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap_importance = pd.DataFrame({
        "Feature": X.columns,
        "SHAP Importance": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by="SHAP Importance", ascending=False)
    return shap_importance


# 5️⃣ Main Driver Identification Function
def identify_drivers(df, target_col, include_items, max_lag=3, ma_windows=[3, 6]):
    """
    Identifies key drivers of the target variable considering lag effects.
    """
    df, feature_cols = create_lagged_features(df, target_col, include_items, max_lag, ma_windows)
    best_lags = cross_correlation_lag(df, target_col, max_lag)
    selected_features = select_features(df, target_col, feature_cols)
    shap_importance = shap_feature_importance(df, target_col, selected_features)
    return df, shap_importance, best_lags





# ====================================
# Auto-Select Best Time-Series Model
# ====================================




def best_time_series_model(train, test, target_col, arima_order, sarima_order, hw_trend, hw_seasonal):
    arima_order = ast.literal_eval(arima_order)
    sarima_order = ast.literal_eval(sarima_order)

    models = {
        # "ARIMA": ARIMA(train[target_col], order=arima_order).fit(),
        "Holt-Winters": ExponentialSmoothing(train[target_col], trend=hw_trend, seasonal=hw_seasonal, seasonal_periods=12).fit(),
        "SARIMA": SARIMAX(train[target_col], order=(1, 1, 1), seasonal_order=sarima_order).fit(),
        # "Prophet": Prophet().fit(train.rename(columns={"Month": "ds", "Value": "y"})),

    }

    best_model = None
    best_rmse = float("inf")
    best_model_name = ""

    for name, model in models.items():
        forecast = model.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test[target_col], forecast))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name

    return best_model, best_model_name

# 3️⃣ Train ML Model for Residuals
def train_residual_model(X_train, y_train, X_test, method="RandomForest"):
    models = {
        # "LinearRegression": LinearRegression(),
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
def hybrid_forecast(df, target_col, top_n_drivers, ml_method, include_items, arima_order, sarima_order, hw_trend, hw_seasonal):
    # ✅ Filter the dataset to include only selected features + target
    # ✅ Strip spaces from column names

    df.columns = df.columns.str.strip()
    df["Item"] = df["Item"].str.strip() 
    df["Item"] = df["Item"].apply(lambda x: x.encode("ascii", "ignore").decode()) 
    df = df.drop_duplicates(subset=["Item", "Month"], keep="first")

    # ✅ Ensure target column and selected features are stripped of spaces
    target_col = target_col.strip()
   
    include_items = [item.strip() for item in include_items]

    # ✅ Filter the dataset to include only selected features + target
    selected_features = [target_col] + include_items  # Ensure the target variable is always included

    
    df = df[df["Item"].isin(selected_features)]


    # ✅ Call identify_drivers() with included features only
    full_data, driver_scores, best_lags = identify_drivers(df, target_col, include_items)

    top_drivers = driver_scores.head(top_n_drivers)["Feature"].tolist()

    train = full_data.iloc[:-12]
    test = full_data.iloc[-12:]

    best_ts_model, best_model_name = best_time_series_model(train, test, target_col, arima_order, sarima_order, hw_trend, hw_seasonal)
    ts_forecast = best_ts_model.forecast(steps=len(test))

    residuals = train[target_col] - best_ts_model.fittedvalues
    X_train = train[top_drivers]
    X_test = test[top_drivers]

    residual_preds = train_residual_model(X_train, residuals, X_test, ml_method)
    hybrid_forecast = ts_forecast + residual_preds

    # ✅ Compute Performance Metrics
    mape_baseline = np.mean(np.abs((test[target_col] - ts_forecast) / test[target_col])) * 100
    mape_hybrid = np.mean(np.abs((test[target_col] - hybrid_forecast) / test[target_col])) * 100
    r2_baseline = r2_score(test[target_col], ts_forecast)
    r2_hybrid = r2_score(test[target_col], hybrid_forecast)

    metrics = {
        "Baseline": {
            "MAE": mean_absolute_error(test[target_col], ts_forecast),
            "RMSE": np.sqrt(mean_squared_error(test[target_col], ts_forecast)),
            "MAPE": mape_baseline,
            "R²": r2_baseline
        },
        "Hybrid": {
            "MAE": mean_absolute_error(test[target_col], hybrid_forecast),
            "RMSE": np.sqrt(mean_squared_error(test[target_col], hybrid_forecast)),
            "MAPE": mape_hybrid,
            "R²": r2_hybrid
        }
    }

    forecast_data = {
        "base_forecast": list(ts_forecast),
        "residual_forecast": list(residual_preds),
        "final_forecast": list(hybrid_forecast),
        "target": target_col,
        "drivers": top_drivers,
        "best_lags": best_lags  # ✅ Include lag information for reference
    }

    return train, test, ts_forecast, hybrid_forecast, metrics, driver_scores, best_model_name, forecast_data



def forecast_with_openai(train, forecast_periods, target_item, driver_data):

    """
    Uses OpenAI to generate a natural-language forecast based on target variable and driver data.
    """
    history_text = "\n".join(
    f"In {index.strftime('%b-%y')}, {target_item} was {row[target_item]:.2f} and {', '.join([f'{driver}: {row[driver]:.2f}' for driver in driver_data['drivers']])}."
    for index, row in train.iterrows()
    )   

    class ForecastEntry(BaseModel):
        month: str
        value: float

    class ForecastResponse(BaseModel):
        forecast: list[ForecastEntry]
        summary: str


    prompt = f"""
    Here is a time-series of financial data along with key drivers:
    {history_text}
    - Predict the next {forecast_periods} months.
    - Analyze the trend and seasonality.
    - Explain how the drivers impact the forecast.

    the explanation should be in a lucid language, bulleted format, easy to understand

    Return a JSON with:
    - 'forecast': List of month-value pairs for the next {forecast_periods} months.
    - 'summary': Markdown bullet points summarizing trends, seasonality, and driver impacts.
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "forecast_schema",  # Add a descriptive name for your schema
                "schema": {
                    "type": "object",
                    "properties": {
                        "forecast": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "month": {"type": "string"},
                                    "value": {"type": "number"}
                                },
                                "required": ["month", "value"]
                            }
                        },
                        "summary": {"type": "string"}
                    },
                    "required": ["forecast", "summary"],
                    "additionalProperties": False
                }
            }
        },
        store=True
    )

    response_data = json.loads(response.choices[0].message.content)
    print("==== response ====")
    print(response_data)
    parsed_forecast = ForecastResponse(**response_data)
    return parsed_forecast.forecast, parsed_forecast.summary