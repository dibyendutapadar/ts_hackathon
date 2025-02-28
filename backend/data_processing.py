import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def load_and_transform(file):
    df = pd.read_excel(file, sheet_name=0)
    df = df.melt(id_vars=[df.columns[0]], var_name="Month", value_name="Value")
    df.columns = ["Item", "Month", "Value"]
    df["Month"] = pd.to_datetime(df["Month"], format="%b-%Y")  # Adjust date format if needed
    return df