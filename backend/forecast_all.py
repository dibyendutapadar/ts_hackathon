import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from prophet import Prophet
from statsmodels.tsa.forecasting.theta import ThetaModel
from sklearn.model_selection import train_test_split
import openai
from pydantic import BaseModel, Field
import re
import json
from pmdarima import auto_arima
from darts.models import NaiveSeasonal, NaiveMovingAverage, RandomForest
from darts import TimeSeries
from typing import List, Optional, Dict
from backend.PredictV3EngineConfig import HoltWinterConfig, PredictEngineWeightageConfig, PredictSameMonthOnMonthConfig, PredictV3EngineConfig, ProjectionViaHoltWinterConfig, SSAForecastConfig, StatisticalDistributionConfig, StatisticalDistributionRiskBoundary
from backend.V3Engine import GetPredictions



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


class ForecastEntry(BaseModel):
    month: str
    value: float

class ForecastResponse(BaseModel):
    forecast: list[ForecastEntry]
    summary: str


def forecast_with_openai(train, forecast_periods, target_item):
    history_text = "\n".join(
        f"In {row.Month.strftime('%b-%y')}, the value was {row.Value}." for _, row in train.iterrows()
    )

    prompt = f"""
    Here is a time-series of financial data:
    {history_text}
    Based on the above pattern of a organizations {target_item}, predict the next {forecast_periods} months and provide a summary explanation of the forecast.
    The summary should be a little detailed. How is the trend and seasonality, how they are affecting the months, etc. Keep it within 100 words, but well markdown formatted in bullet points for easily understandable.
    Don't use any model or code, use natural reasoning ability for forecasting.
    Return the response as a JSON object with keys 'forecast' and 'summary'.
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



def naivePrediction(train, periods, algorithm):
    time_series = TimeSeries.from_dataframe(train, time_col="Month")
    print(time_series)
    if algorithm == "Naive":
        model = NaiveSeasonal(12)
    elif algorithm == "MovingAvg":
        model = NaiveMovingAverage(12)
    elif algorithm == "RandomForest":
        model = RandomForest(lags=12, output_chunk_length=3)
    else:
        model = NaiveMovingAverage(3)
    model.fit(time_series)
    forecast = model.predict(periods)
    forecast = forecast.values().flatten().tolist()
    return forecast

def accuracyOutput(naive_forecast, test):
    return {
        "Forecast": naive_forecast,
        "MAE": round(mean_absolute_error(test, naive_forecast), 3),
        "RMSE": round(np.sqrt(mean_squared_error(test, naive_forecast)), 3),
        "MAPE": round(mean_absolute_percentage_error(test, naive_forecast), 3),
        "Accuracy": 100- round(mean_absolute_percentage_error(test, naive_forecast), 3),
        "R2": round(r2_score(test, naive_forecast), 3)
    }

def v3Prediction(str_list):
    floatList = list(map(float, str_list))
    input_bucketized = transform_to_bucketize_series(floatList, 12)
    prediction_config = PredictV3EngineConfig(False, 6, True, PredictSameMonthOnMonthConfig(0.1, 0.2, 0.3),
                                          PredictEngineWeightageConfig(0.25, 0.25, 0.5), HoltWinterConfig(0.8, 0.9, 0.01),
                                          SSAForecastConfig(5, 10, 100), ProjectionViaHoltWinterConfig(0.7, 0.9, 
                                          StatisticalDistributionConfig(StatisticalDistributionRiskBoundary(0.3, 0.8), StatisticalDistributionRiskBoundary(0.15, 0.95), StatisticalDistributionRiskBoundary(0.1, 0.99), 36, 2.5, 0.05, 0.1, 0.2, 50.0, True), 0.0, True, 0.8, 1.2))

    predictions = GetPredictions(input_bucketized, 1, prediction_config)
    result = [prediction.predictedValue for prediction in predictions]
    return result

def transform_to_bucketize_series(flat_time_series: List[Optional[float]], period_count: int) -> List[List[Optional[float]]]:
        result = []
        for i in range(len(flat_time_series)):
            if i < period_count:
                result.append([flat_time_series[i]])
            else:
                index = i % period_count
                result[index].append(flat_time_series[i])
        return result

# Forecasting Function
def forecast_time_series_all(df, target_item, forecast_periods=12):
    data = df[df["Item"] == target_item][["Month", "Value"]].sort_values("Month")
    data = data.drop_duplicates(subset=["Month"], keep="first")
    train, test = train_test_split(data, test_size=forecast_periods, shuffle=False)

    models = {
        "Holt-Winters": ExponentialSmoothing(train["Value"], trend="add", seasonal="add", seasonal_periods=12).fit(),
        "SARIMA": SARIMAX(train["Value"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(),
        "Prophet": Prophet().fit(train.rename(columns={"Month": "ds", "Value": "y"})),
        "ARIMA": auto_arima(train["Value"], seasonal=False, stepwise=True, trace=True),
        # "NaiveSeasonal": NaiveSeasonal(12).fit(TimeSeries.from_dataframe(train, time_col="Month")),
        # "NaiveMovingAvg": NaiveMovingAverage(12).fit(TimeSeries.from_dataframe(train, time_col="Month")),
        # "RandomForest": RandomForest(lags=12, output_chunk_length=3).fit(TimeSeries.from_dataframe(train, time_col="Month"))
    }

    results = {}
    test_df = pd.DataFrame({"ds": test["Month"]})

    for name, model in models.items():
        if name == "Prophet":
            forecast = model.predict(test_df)["yhat"]
        elif name in ["ARIMA", "NaiveSeasonal", "NaiveMovingAvg", "RandomForest"]:
            forecast = model.predict(n_periods=len(test))
        else:
            forecast = model.forecast(steps=len(test))

        results[name] = {
            "Forecast": forecast.values,
            "MAE": round(mean_absolute_error(test["Value"], forecast), 3),
            "RMSE": round(np.sqrt(mean_squared_error(test["Value"], forecast)),3),
            "MAPE": round(mean_absolute_percentage_error(test["Value"], forecast), 3),
            "Accuracy" : 100 - round(mean_absolute_percentage_error(test["Value"], forecast), 3),
            "R2": round(r2_score(test["Value"], forecast),3)
            
        }




# OpenAI Forecasting

    openai_forecast, forecast_summary = forecast_with_openai(train, forecast_periods, target_item)
    forecast_values = np.array([entry.value for entry in openai_forecast])

    

    results["OpenAI"] = {
        "Forecast": openai_forecast,
        "MAE": round(mean_absolute_error(test["Value"], forecast_values), 3),
        "RMSE": round(np.sqrt(mean_squared_error(test["Value"], forecast_values)), 3),
        "MAPE": round(mean_absolute_percentage_error(test["Value"], forecast_values), 3),
        "Accuracy" : 100 - round(mean_absolute_percentage_error(test["Value"], forecast_values), 3),
        "R2": round(r2_score(test["Value"], forecast_values),3),
        "Summary": forecast_summary
    }



    naive_forecast = naivePrediction(train, 12, "Naive")
    results["Naive"] = accuracyOutput(naive_forecast, test["Value"])
    ma_forecast = naivePrediction(train, 12, "MovingAvg")
    results["Moving Avg"] = accuracyOutput(ma_forecast, test["Value"])
    maq_forecast = naivePrediction(train, 12, "MovingAvgQuarter")
    results["Moving Avg Quarter"] = accuracyOutput(maq_forecast, test["Value"])
    RandomForest = naivePrediction(train, 12, "RandomForest")
    results["Random Forest"] = accuracyOutput(RandomForest, test["Value"])


    predictv3_forecast = v3Prediction(train["Value"])
    results["PredictV3"] = {
        "Forecast": predictv3_forecast,
        "MAE": round(mean_absolute_error(test["Value"], predictv3_forecast), 3),
        "RMSE": round(np.sqrt(mean_squared_error(test["Value"], predictv3_forecast)), 3),
        "MAPE": round(mean_absolute_percentage_error(test["Value"], predictv3_forecast), 3),
        "Accuracy" : 100 - round(mean_absolute_percentage_error(test["Value"], forecast_values), 3),
        "R2": round(r2_score(test["Value"], predictv3_forecast),3)
    }


    return train, test, results