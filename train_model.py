import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Create Models Folder
os.makedirs("models", exist_ok=True)

# Load Dataset
df = pd.read_csv("dataset/solar_data.csv")

# Data Cleaning
df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

# Features & Target
X = df.drop("generated_power_kw", axis=1)
y = df["generated_power_kw"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluation
print("\n===== MODEL PERFORMANCE =====")
print("MAE :", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R² :", r2_score(y_test, pred))

# Save Model
joblib.dump(
    model,
    "models/solar_model.pkl"
)

print("\nModel Saved Successfully")