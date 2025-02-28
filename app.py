import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backend.data_processing import load_and_transform
from backend.forecasting import forecast_time_series

# Streamlit UI Setup
st.title("📊 Multi-Model Time Series Forecasting")

# File Upload
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file is not None:
    df = load_and_transform(uploaded_file)
    st.write("Preview of Data:")
    st.dataframe(df.head())

    # Select Target Column
    target_item = st.selectbox("Select an Item for Forecasting", df["Item"].unique())

    # Forecast Button
    if st.button("Run Forecast"):
        with st.spinner("Running forecasts..."):
            train, test, results = forecast_time_series(df, target_item)

        # Plot Forecast Results
        plt.figure(figsize=(12, 6))
        plt.plot(train["Month"], train["Value"], label="Train Data", color="blue")
        plt.plot(test["Month"], test["Value"], label="Test Data", color="black", linestyle="dashed")

        for name, res in results.items():
            if name == "OpenAI":
                # Extract values from ForecastEntry objects
                forecast_values = np.array([entry.value for entry in res["Forecast"]], dtype=float)
            else:
                forecast_values = np.array(res["Forecast"], dtype=float)
            plt.plot(test["Month"], forecast_values, label=name)

        plt.legend()
        plt.xlabel("Month")
        plt.ylabel("Value")
        plt.title("Forecasting Comparison")
        plt.grid()
        st.pyplot(plt)

        # Show Metrics
        metrics_df = pd.DataFrame({name: [res["MAE"], res["RMSE"], res["MAPE"],res["R2"]] for name, res in results.items()},
                              index=["MAE", "RMSE","MAPE","R2"]).T
        st.write("Model Performance Metrics:")
        st.dataframe(metrics_df)
        st.write("Explanation of Forecast")
        st.markdown(results["OpenAI"]["Summary"])
