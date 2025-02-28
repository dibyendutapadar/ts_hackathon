import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backend.data_processing import load_and_transform
from backend.forecasting import forecast_time_series
from st_aggrid import AgGrid, GridOptionsBuilder
import plotly.graph_objects as go

# Streamlit UI Setup
st.title("📊 Multi-Model Time Series Forecasting")

# File Upload
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file:
    df_view = pd.read_excel(uploaded_file)

    st.write("### Data Preview")

    # Round numbers to 2 decimal places
    df_view = df_view.round(2)

    # Apply conditional formatting using a new column
    def apply_color(val):
        """Returns color style based on value (red for negative, blue for positive)"""
        if isinstance(val, (int, float)):
            return f"color: {'red' if val < 0 else 'blue' if val > 0 else 'black'}"
        return ""

    # Convert numeric columns to styled strings for AgGrid
    for col in df_view.select_dtypes(include=['number']).columns:
        df_view[col] = df_view[col].apply(lambda x: f"{x:.2f}")  # Format to 2 decimal places

    # Create Ag-Grid options builder
    gb = GridOptionsBuilder.from_dataframe(df_view)

    # Enable pagination and scrolling
    gb.configure_pagination(paginationAutoPageSize=False)  # Keep scrollable table
    gb.configure_grid_options(domLayout="autoHeight")  # Auto adjust height
    gb.configure_default_column(resizable=True, suppressSizeToFit=True)  # Prevent squeezing
    gb.configure_side_bar()  # Enable column filtering

    # Freeze first two columns
    gb.configure_columns(df_view.columns[:1], pinned="left")

    # Apply custom cell styling for negative/positive values
    for col in df_view.columns:
        if df_view[col].dtype == "object":  # All strings in black
            gb.configure_column(col, cellStyle={"color": "black"})
        elif df_view[col].dtype in ["float64", "int64"]:  # Check for numeric columns
            gb.configure_column(
            col, 
            cellStyle=lambda params: {"color": "red"} if float(params["value"]) < 0 else {"color": "blue"}
        )

    # Build grid options
    grid_options = gb.build()

    # Render DataFrame using AgGrid with frozen columns and scrollable view
    AgGrid(
        df_view, 
        gridOptions=grid_options, 
        height=500,  # Fixed height for vertical scroll
        fit_columns_on_grid_load=False,  # Prevent auto-fit
        enable_enterprise_modules=False  # Avoid heavy rendering
    )










    # Load and Transform
    df = load_and_transform(uploaded_file)

    # Select Target Column
    target_item = st.selectbox("Select an Item for Forecasting", df["Item"].unique())

    # Forecast Button
    if st.button("Run Forecast"):
        with st.spinner("Running forecasts..."):
            train, test, results = forecast_time_series(df, target_item)

        # # Plot Forecast Results
        # plt.figure(figsize=(12, 6))
        # plt.plot(train["Month"], train["Value"], label="Train Data", color="blue")
        # plt.plot(test["Month"], test["Value"], label="Test Data", color="black", linestyle="dashed")

        # Create the interactive Plotly figure
        fig = go.Figure()

        # Add train data
        fig.add_trace(go.Scatter(x=train["Month"], y=train["Value"], mode='lines', name="Train Data", line=dict(color="blue")))

        # Add test data
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


        # Show Metrics
        metrics_df = pd.DataFrame({name: [res["MAE"], res["RMSE"], res["MAPE"],res["R2"]] for name, res in results.items()},
                              index=["MAE", "RMSE","MAPE","R2"]).T
        st.write("Model Performance Metrics:")
        st.dataframe(metrics_df)
        st.write("Explanation of Forecast")
        st.markdown(results["OpenAI"]["Summary"])
