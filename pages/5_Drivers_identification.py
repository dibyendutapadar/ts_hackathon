import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import openai
import seaborn as sns
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import statsmodels.api as sm
from backend.data_processing import load_and_transform
from backend.driver_analysis import hybrid_forecast, forecast_with_openai

st.title("📊 Driver Analysis & Hybrid Forecasting")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file:

    df_view = pd.read_excel(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df_view, height=500, width=1000)

    df = load_and_transform(uploaded_file)
    # 2️⃣ Target Selection
    target_item = st.selectbox("Select Target Variable", df["Item"].str.strip().unique())

    # 3️⃣ Allow User to Include Specific Items
    include_items = st.multiselect("Select Items to Include in Analysis", df["Item"].str.strip().unique())

    # 4️⃣ Forecast Configuration
    top_n_drivers = st.slider("Select Number of Top Drivers", 1, 10, 3)
    ml_method = st.selectbox("Select Residual Prediction Model", ["RandomForest", "XGBoost"])

    # Model Parameter Selection
    st.write("### Model Parameter Configuration")
    arima_order = st.text_input("ARIMA Order (p,d,q)", "(1,1,1)")
    sarima_order = st.text_input("SARIMA Seasonal Order (P,D,Q,S)", "(1,1,1,12)")
    hw_trend = st.selectbox("Holt-Winters Trend", ["add", "mul"])
    hw_seasonal = st.selectbox("Holt-Winters Seasonal", ["add", "mul"])

    if st.button("Run Analysis & Forecast"):
        st.write(df)
        train, test, ts_forecast, hybrid_forecast, metrics, driver_scores, best_model_name, forecast_data = hybrid_forecast(
            df, target_item, top_n_drivers, ml_method, include_items, arima_order, sarima_order, hw_trend, hw_seasonal
        )

        # 6️⃣ Time Series Decomposition
        st.write("### Time Series Decomposition")
        decomposition = sm.tsa.seasonal_decompose(train[target_item], period=12)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        axes[0].plot(decomposition.trend)
        axes[0].set_title("Trend")
        axes[1].plot(decomposition.seasonal)
        axes[1].set_title("Seasonality")
        axes[2].plot(decomposition.resid)
        axes[2].set_title("Residuals")
        plt.tight_layout()
        st.pyplot(fig)


        # 7️⃣ Display Key Drivers
        st.write("### Key Drivers Identified")
        st.dataframe(driver_scores.head(top_n_drivers))


        # 8️⃣ Bar Graph of Driver Importance (SHAP Values)
        st.write("### Driver Importance (SHAP Values)")
        plt.figure(figsize=(10, 5))
        sns.barplot(x=driver_scores["SHAP Importance"].head(top_n_drivers), y=driver_scores["Feature"].head(top_n_drivers), palette="viridis")
        plt.xlabel("SHAP Importance Score")
        plt.ylabel("Feature")
        plt.title("Top Drivers of Target Variable")
        st.pyplot(plt)

        st.write(f"### Best Selected Base Model: {best_model_name}")

        # OpenAI Forecast
        if OPENAI_API_KEY:

            openai_forecast, openai_summary = forecast_with_openai(train, forecast_periods=len(test), target_item=target_item, driver_data=forecast_data)
            forecast_values = np.array([entry.value for entry in openai_forecast])


            metrics["OpenAI"] = {
                # "Forecast": forecast_values,
                "MAE": round(mean_absolute_error(test[target_item], forecast_values), 3),
                "RMSE": round(np.sqrt(mean_squared_error(test[target_item], forecast_values)), 3),
                "MAPE": round(mean_absolute_percentage_error(test[target_item], forecast_values), 3),
                "R²": round(r2_score(test[target_item], forecast_values),3),
                }

            

        # Forecast Comparison Plot
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train[target_item], label="Train Data", color="blue")
            plt.plot(test.index, test[target_item], label="Test Data", linestyle="dashed", color="black")
            plt.plot(test.index, ts_forecast, label="Baseline (Best Time-Series)", color="red")
            plt.plot(test.index, hybrid_forecast, label="Hybrid Model (With Drivers)", color="green")
            plt.plot(test.index, forecast_values, label="OpenAI Forecast", color="purple", linestyle="dotted")
            plt.legend()
            plt.xlabel("Month")
            plt.ylabel("Value")
            plt.title("Forecast Comparison: Baseline vs Hybrid vs OpenAI")
            plt.grid()
            st.pyplot(plt)


            st.write("### Model Performance Metrics")
            st.write(metrics)
            metrics_df = pd.DataFrame(metrics).T
            metrics_df.columns = ["MAE", "RMSE", "MAPE", "R²"]
            st.dataframe(metrics_df)

            st.write("### OpenAI-Generated Summary")
            st.markdown(openai_summary)