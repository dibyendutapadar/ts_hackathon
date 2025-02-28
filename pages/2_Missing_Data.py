import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from backend.data_processing import load_and_transform
from backend.forecasting_missing import forecast_for_missing_data, mask_data

st.title("🤖 OpenAI Forecasting with Missing Data")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file:
    df_view = pd.read_excel(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df_view, height=500, width=1000)
    df = load_and_transform(uploaded_file)
    

    target_item = st.selectbox("Select an Item", df["Item"].unique())

    if st.button("Mask Data and Run OpenAI Forecast"):
        masked_df = mask_data(df[df["Item"] == target_item])

        

        train, test, results, prompt = forecast_for_missing_data(df, target_item, masked_df, forecast_periods=12)

        st.write("====Send the following prompt to openAI=======")
        st.write(prompt)
        st.write("Waiting for response")

        plt.figure(figsize=(12, 6))
        plt.plot(masked_df["Month"], masked_df["Value"], label="Masked Data", color="red")
        # plt.plot(df["Month"], df["Value"], label="Original Data", color="blue", alpha=0.5)
        plt.plot(test["Month"], test["Value"], label="Test Data", color="black", linestyle="dashed")


        for name, res in results.items():
            if name == "OpenAI":
                # Extract values from ForecastEntry objects
                forecast_values = np.array([entry.value for entry in res["Forecast"]], dtype=float)
            print(forecast_values)
            plt.plot(test["Month"], forecast_values, label=name)
        
        plt.legend()
        plt.xlabel("Month")
        plt.ylabel("Value")
        plt.title("Forecasting Comparison")
        plt.grid()
        st.pyplot(plt)


        metrics_df = pd.DataFrame({name: [res["MAE"], res["RMSE"], res["MAPE"],res["R2"]] for name, res in results.items()},
                              index=["MAE", "RMSE","MAPE","R2"]).T
        st.write("Model Performance Metrics:")
        st.dataframe(metrics_df)
        st.write("Explanation of Forecast")
        st.markdown(results["OpenAI"]["Summary"])
