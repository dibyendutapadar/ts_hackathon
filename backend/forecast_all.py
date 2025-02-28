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



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


class ForecastEntry(BaseModel):
    month: str
    value: float

class ForecastResponse(BaseModel):
    forecast: list[ForecastEntry]
    summary: str


def forecast_with_openai(train, forecast_periods):
    history_text = "\n".join(
        f"In {row.Month.strftime('%b-%y')}, the value was {row.Value}." for _, row in train.iterrows()
    )

    prompt = f"""
    Here is a time-series of financial data:
    {history_text}
    Based on the above pattern, predict the next {forecast_periods} months and provide a summary explanation of the forecast.
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




# Forecasting Function
def forecast_time_series(df, target_item, forecast_periods=12):
    data = df[df["Item"] == target_item][["Month", "Value"]].sort_values("Month")
    train, test = train_test_split(data, test_size=forecast_periods, shuffle=False)

    models = {
        "ARIMA": ARIMA(train["Value"], order=(1, 1, 1)).fit(),
        "Holt-Winters": ExponentialSmoothing(train["Value"], trend="add", seasonal="add", seasonal_periods=12).fit(),
        "SARIMA": SARIMAX(train["Value"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(),
        "Prophet": Prophet().fit(train.rename(columns={"Month": "ds", "Value": "y"})),
        # "Theta": ThetaModel(train["Value"]).fit(),
        # "Dynamic Factor": DynamicFactor(train["Value"], k_factors=1).fit()
    }

    results = {}
    test_df = pd.DataFrame({"ds": test["Month"]})

    for name, model in models.items():
        if name == "Prophet":
            forecast = model.predict(test_df)["yhat"]
        elif name == "Dynamic Factor":
            forecast = model.predict(start=len(train), end=len(train) + len(test) - 1)
        else:
            forecast = model.forecast(steps=len(test))

        results[name] = {
            "Forecast": forecast.values,
            "MAE": round(mean_absolute_error(test["Value"], forecast), 3),
            "RMSE": round(np.sqrt(mean_squared_error(test["Value"], forecast)),3),
            "MAPE": round(mean_absolute_percentage_error(test["Value"], forecast), 3),
            "R2": round(r2_score(test["Value"], forecast),3)
            
        }



# OpenAI Forecasting

    openai_forecast, forecast_summary = forecast_with_openai(train, forecast_periods)
    forecast_values = np.array([entry.value for entry in openai_forecast])

    

    results["OpenAI"] = {
        "Forecast": openai_forecast,
        "MAE": round(mean_absolute_error(test["Value"], forecast_values), 3),
        "RMSE": round(np.sqrt(mean_squared_error(test["Value"], forecast_values)), 3),
        "MAPE": round(mean_absolute_percentage_error(test["Value"], forecast_values), 3),
        "R2": round(r2_score(test["Value"], forecast_values),3),
        "Summary": forecast_summary
    }

    return train, test, results