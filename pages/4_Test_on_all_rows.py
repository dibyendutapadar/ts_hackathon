import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import streamlit as st
from backend.data_processing import load_and_transform
from backend.forecasting import forecast_time_series
from st_aggrid import AgGrid, GridOptionsBuilder
from pydantic import BaseModel, Field
import openai
import json

# Streamlit UI Setup
st.title("📊 Multi-Model Time Series Forecasting")

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# File Upload
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file:
    df_view = pd.read_excel(uploaded_file)
    st.write("### Data Preview")
    df_view = df_view.round(2)

    # Load and Transform
    df = load_and_transform(uploaded_file)
    
    # Select Targets
    target_items = st.multiselect("Select Items for Forecasting", df["Item"].unique())
    
    # Forecast Button
    if st.button("Run Forecast for Selected Items") and target_items:
        with st.spinner("Running forecasts on selected items..."):
            all_results = {}
            for item in target_items:
                train, test, results = forecast_time_series(df, item)
                all_results[item] = results

        # Prepare a unified DataFrame for all metrics
        metrics_data = []
        for item, result in all_results.items():
            for model, res in result.items():
                metrics_data.append([item, model, res["MAE"], res["RMSE"], res["MAPE"], res["R2"]])
        
        metrics_df = pd.DataFrame(metrics_data, columns=["Item", "Model", "MAE", "RMSE", "MAPE", "R2"])
        
        # Display Results Table
        st.write("### Model Performance Metrics for Selected Items")
        AgGrid(metrics_df)
        
        # Rank Models using OpenAI
        ranking_prompt = f"""
        Given the following model performance metrics table, rank the models. Take into consideration both MAPE and R2 for an informed decision. Explain the response in a short summarization.
        {metrics_df.to_csv(index=False)}
        Return the output as a JSON.
        """
        
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[{"role": "user", "content": ranking_prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ranking_schema",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "rankings": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "item": {"type": "string"},
                                            "models": {"type": "array", "items": {"type": "string"}}
                                        },
                                        "required": ["item", "models"]
                                    }
                                },
                                "summary":{"type": "string"}
                            },
                            "required": ["rankings"],
                            "additionalProperties": False
                        }
                    }
                },
                store=True
            )
            response_data = json.loads(response.choices[0].message.content)
            ranking_data = response_data.get("rankings", [])
            ranking_summary = response_data.get("summary",[])
        except Exception as e:
            st.error(f"Error processing ranking data: {str(e)}")
            ranking_data = []

        
        st.write(response_data)
        st.write(ranking_data)
        st.write(ranking_summary)
        
        if ranking_data:
            # Convert ranking data into DataFrame with variable column count
            max_models = max(len(entry["models"]) for entry in ranking_data)
            column_names = ["Item"] + [f"Rank {i+1}" for i in range(max_models)]
            ranking_rows = [[entry["item"]] + entry["models"] + ["-"] * (max_models - len(entry["models"])) for entry in ranking_data]
            ranking_df = pd.DataFrame(ranking_rows, columns=column_names)
            
            st.write("### Model Ranking Based on Performance")
            AgGrid(ranking_df)
        else:
            st.write("No ranking data available.")
        
        # Plot Forecast Results for Each Selected Item
        for item, result in all_results.items():
            st.write(f"### Forecast for {item}")
            plt.figure(figsize=(12, 6))
            train, test, _ = forecast_time_series(df, item)
            plt.plot(train["Month"], train["Value"], label="Train Data", color="blue")
            plt.plot(test["Month"], test["Value"], label="Test Data", color="black", linestyle="dashed")
            
            for name, res in result.items():
                forecast_values = np.array([entry.value for entry in res["Forecast"]], dtype=float) if name == "OpenAI" else np.array(res["Forecast"], dtype=float)
                plt.plot(test["Month"], forecast_values, label=name)

            plt.legend()
            plt.xlabel("Month")
            plt.ylabel("Value")
            plt.title(f"Forecasting Comparison for {item}")
            plt.grid()
            st.pyplot(plt)

            # Show Forecast Summary
            if "OpenAI" in result:
                st.write("### Explanation of Forecast")
                st.markdown(result["OpenAI"]["Summary"])