import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import streamlit as st
from backend.data_processing import load_and_transform
from backend.forecasting import forecast_time_series
from backend.forecast_all import forecast_time_series_all
from st_aggrid import AgGrid, GridOptionsBuilder
from pydantic import BaseModel, Field
import openai
import json
import plotly.graph_objects as go

# Streamlit UI Setup
st.title("📊 Multi-Model Time Series Forecasting")

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# File Upload
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file:
    df_view = pd.read_excel(uploaded_file)
    
    df_view = df_view.round(2)
    st.write("### Data Preview")
    st.dataframe(df_view, height=500, width=1000)

    # Load and Transform
    df = load_and_transform(uploaded_file)
    
    # Select Targets
    target_items = st.multiselect("Select Items for Forecasting", df["Item"].unique())
    
    # Forecast Button
    if st.button("Run Forecast for Selected Items") and target_items:
        with st.spinner("Running forecasts on selected items..."):
            all_results = {}
            for item in target_items:
                train, test, results = forecast_time_series_all(df, item)
                all_results[item] = results

        # Prepare a unified DataFrame for all metrics
        metrics_data = []
        for item, result in all_results.items():
            for model, res in result.items():
                metrics_data.append([item, model, res["MAE"], res["RMSE"], res["MAPE"], res["Accuracy"],res["R2"]])
        
        metrics_df = pd.DataFrame(metrics_data, columns=["Item", "Model", "MAE", "RMSE", "MAPE", "Accuracy", "R2"])
        
        # Display Results Table
        st.write("### Model Performance Metrics for Selected Items")
        st.dataframe(metrics_df, height=500, width=1000)
        
        # Rank Models using OpenAI
        ranking_prompt = f"""
        Given the following model performance metrics table, rank the models. Take into consideration both MAPE and R2 for an informed decision. 
        Explain the response in a short summarization. Bullet wise which is the best model for which Target.
        And also include that If I have to do an ensembel model for the entire dataset what %age weightage should I assign to each model.

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

        if ranking_data:
            # Convert ranking data into DataFrame with variable column count
            max_models = max(len(entry["models"]) for entry in ranking_data)
            column_names = ["Item"] + [f"Rank {i+1}" for i in range(max_models)]
            ranking_rows = [[entry["item"]] + entry["models"] + ["-"] * (max_models - len(entry["models"])) for entry in ranking_data]
            ranking_df = pd.DataFrame(ranking_rows, columns=column_names)
            
            st.write("### Model Ranking Based on Performance")
            st.dataframe(ranking_df, height=500, width=1000)
            st.write(ranking_summary)
        else:
            st.write("No ranking data available.")
        
        # Plot Forecast Results for Each Selected Item
        for item, result in all_results.items():
            st.write(f"### Forecast for {item}")
            # plt.figure(figsize=(12, 6))
            train, test, _ = forecast_time_series(df, item)
            # plt.plot(train["Month"], train["Value"], label="Train Data", color="blue")
            # plt.plot(test["Month"], test["Value"], label="Test Data", color="black", linestyle="dashed")
            
            # for name, res in result.items():
            #     forecast_values = np.array([entry.value for entry in res["Forecast"]], dtype=float) if name == "OpenAI" else np.array(res["Forecast"], dtype=float)
            #     plt.plot(test["Month"], forecast_values, label=name)

            # plt.legend()
            # plt.xlabel("Month")
            # plt.ylabel("Value")
            # plt.title(f"Forecasting Comparison for {item}")
            # plt.grid()
            # st.pyplot(plt)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train["Month"], y=train["Value"], mode='lines', name="Train Data", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=test["Month"], y=test["Value"], mode='lines', name="Test Data", line=dict(color="blue")))
            for name, res in results.items():
                if name == "OpenAI":
                    # Extract values from ForecastEntry objects
                    forecast_values = np.array([entry.value for entry in res["Forecast"]], dtype=float)
                else:
                    forecast_values = np.array(res["Forecast"], dtype=float)
                fig.add_trace(go.Scatter(x=test["Month"], y=forecast_values, mode='lines', name=name))

        # Update layout
            fig.update_layout(
                title="Forecasting Comparison",
                xaxis_title="Month",
                yaxis_title="Value",
                hovermode="x",
                template="plotly_white"
            )

# Render the interactive chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)


            # Show Forecast Summary
            if "OpenAI" in result:
                st.write("### Explanation of Forecast")
                st.markdown(result["OpenAI"]["Summary"])