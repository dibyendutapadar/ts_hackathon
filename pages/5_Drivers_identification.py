import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from backend.data_processing import load_and_transform
from backend.driver_analysis import hybrid_forecast

st.title("📊 Driver Analysis & Hybrid Forecasting")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file:

    
    df_view = pd.read_excel(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df_view, height=500, width=1000)


    df = load_and_transform(uploaded_file)
    target_item = st.selectbox("Select Target Variable", df["Item"].unique())
    top_n_drivers = st.slider("Select Number of Top Drivers", 1, 5, 3)
    ml_method = st.selectbox("Select Residual Prediction Model", ["LinearRegression", "RandomForest", "XGBoost"])

    if st.button("Run Analysis & Forecast"):
        train, test, ts_forecast, hybrid_forecast, metrics, driver_scores = hybrid_forecast(df, target_item, top_n_drivers, ml_method)

        st.write("### Key Drivers Identified")
        st.dataframe(driver_scores.head(top_n_drivers))

        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train[target_item], label="Train Data", color="blue")
        plt.plot(test.index, test[target_item], label="Test Data", linestyle="dashed", color="black")
        plt.plot(test.index, ts_forecast, label="Baseline (Best Time-Series)", color="red")
        plt.plot(test.index, hybrid_forecast, label="Hybrid Model (With Drivers)", color="green")
        plt.legend()
        plt.xlabel("Month")
        plt.ylabel("Value")
        plt.title("Forecast Comparison: Baseline vs Hybrid")
        plt.grid()
        st.pyplot(plt)

        st.write("### Model Performance Metrics")
        st.dataframe(pd.DataFrame(metrics, index=["Value"]).T)
