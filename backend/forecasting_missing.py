import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
import openai
from pydantic import BaseModel, Field
import re
import json




load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


def mask_data(df, mask_percentage=20):
    masked_df = df.copy()
    rng = np.random.default_rng()

    if "Value" in masked_df.columns and not masked_df.empty:
        mask_size = int(len(masked_df) * mask_percentage / 100)
        
        if mask_size > 0:  # Ensure we don't try to sample an empty set
            mask_indices = rng.choice(masked_df.index, size=mask_size, replace=False)
            masked_df.loc[mask_indices, "Value"] = np.nan

    return masked_df


class ForecastEntry(BaseModel):
    month: str
    value: float

class ForecastResponse(BaseModel):
    forecast: list[ForecastEntry]
    summary: str


def forecast_with_openai_missing(train, forecast_periods):
    history_text = "\n".join(
        f"In {row.Month.strftime('%b-%y')}, the value was {row.Value if not pd.isna(row.Value) else 'MISSING'}." for _, row in train.iterrows()
    )

    prompt = f"""
    Here is a time-series of financial data:
    {history_text}
    Based on the above pattern, predict the next {forecast_periods} months and provide a summary explanation of the forecast.
    The summary should be a little detailed. How is the trend and seasonality, how they are affecting the months, etc. Keep it within 100 words, but well markdown formatted in bullet points for easily understandable.
    There are some missing data, provide an explanation for how the assumption was done for them.
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
    return parsed_forecast.forecast, parsed_forecast.summary, prompt




# Forecasting Function
def forecast_for_missing_data(actual_df, target_item, masked_df,forecast_periods=12):
    data = actual_df[actual_df["Item"] == target_item][["Month", "Value"]].sort_values("Month")
    train, test = train_test_split(data, test_size=forecast_periods, shuffle=False)
    masked_data= masked_df.sort_values("Month")
    train1, _ =train_test_split(masked_data, test_size=forecast_periods, shuffle=False)

    

    results = {}




# OpenAI Forecasting

    openai_forecast, forecast_summary, prompt = forecast_with_openai_missing(train1, forecast_periods)
    forecast_values = np.array([entry.value for entry in openai_forecast])

    

    results["OpenAI"] = {
        "Forecast": openai_forecast,
        "MAE": round(mean_absolute_error(test["Value"], forecast_values), 3),
        "RMSE": round(np.sqrt(mean_squared_error(test["Value"], forecast_values)), 3),
        "MAPE": round(mean_absolute_percentage_error(test["Value"], forecast_values), 3),
        "R2": round(r2_score(test["Value"], forecast_values),3),
        "Summary": forecast_summary
    }

    return train, test, results , prompt