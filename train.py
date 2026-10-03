
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/food_waste_data.csv"
MODEL_PATH = "models/linear_regression_model.joblib"
REPORT_PATH = "reports/model_results.csv"

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH).drop_duplicates()

# Feature engineering
df["unsold_meals"] = (df["meals_prepared"] - df["meals_sold"]).clip(lower=0)
df["sales_ratio"] = df["meals_sold"] / df["meals_prepared"].replace(0, 1)
df["temperature_above_25"] = (df["temperature_c"] - 25).clip(lower=0)

target = "food_waste_kg"

features = [
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "special_event",
    "temperature_c",
    "temperature_above_25",
    "rainfall_mm",
    "customers",
    "meals_prepared",
    "meals_sold",
    "unsold_meals",
    "previous_waste_kg",
    "avg_meal_price",
    "staff_count",
    "occupancy_rate",
    "food_category",
]

X = df[features].copy()
y = df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

categorical_features = ["day_of_week", "food_category"]
numeric_features = [c for c in features if c not in categorical_features]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

results = pd.DataFrame(
    [{
        "model": "Linear Regression",
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }]
)

results.to_csv(REPORT_PATH, index=False)
joblib.dump(model, MODEL_PATH)

print("\n===== Linear Regression Results =====")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))
print("MAE :", round(mae, 3), "kg")
print("RMSE:", round(rmse, 3), "kg")
print("R²  :", round(r2, 3))
print("\nModel saved successfully:")
print(MODEL_PATH)
