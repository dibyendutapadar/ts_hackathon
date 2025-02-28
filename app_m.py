import streamlit as st

st.set_page_config(page_title="Time Series Forecasting", layout="wide")

st.title("📊 Multi-Page Forecasting App")

st.sidebar.success("Select a page from the navigation.")
st.write("""
### Welcome to the Forecasting App!
- Compare multiple forecasting models on various targets
- Use OpenAI to forecast for missing data and get explanations.
- Improve Forecast using drivers
""")