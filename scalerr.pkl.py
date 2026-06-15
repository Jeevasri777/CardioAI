import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler


# Load dataset

df = pd.read_csv(r"C:\Users\jeeva\OneDrive\Desktop\internship\datasetss.csv")


# Remove target column

X = df.drop(
    "Heart Disease",
    axis=1
)


# Create scaler

scaler = StandardScaler()


# Fit scaler

scaler.fit(
    X
)


# Save

joblib.dump(
    scaler,
    "scalerr.pkl"
)


print("scaler.pkl created")

