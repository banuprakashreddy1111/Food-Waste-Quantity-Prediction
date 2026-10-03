
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATA_PATH = "data/food_waste_data.csv"
REPORT_PATH = Path("reports")
REPORT_PATH.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH).drop_duplicates()

print("===== Dataset Information =====")
print("Shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary:")
print(df.describe(include="all").T)

df["unsold_meals"] = (df["meals_prepared"] - df["meals_sold"]).clip(lower=0)

plt.figure(figsize=(8, 5))
plt.hist(df["food_waste_kg"], bins=30)
plt.title("Distribution of Food Waste")
plt.xlabel("Food waste (kg)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(REPORT_PATH / "food_waste_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df["unsold_meals"], df["food_waste_kg"], alpha=0.5)
plt.title("Unsold Meals vs Food Waste")
plt.xlabel("Unsold meals")
plt.ylabel("Food waste (kg)")
plt.tight_layout()
plt.savefig(REPORT_PATH / "unsold_meals_vs_waste.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df["customers"], df["food_waste_kg"], alpha=0.5)
plt.title("Customers vs Food Waste")
plt.xlabel("Customers")
plt.ylabel("Food waste (kg)")
plt.tight_layout()
plt.savefig(REPORT_PATH / "customers_vs_waste.png", dpi=300)
plt.close()

numeric_columns = [
    "is_weekend",
    "is_holiday",
    "special_event",
    "temperature_c",
    "rainfall_mm",
    "customers",
    "meals_prepared",
    "meals_sold",
    "previous_waste_kg",
    "avg_meal_price",
    "staff_count",
    "occupancy_rate",
    "food_waste_kg",
]

corr = df[numeric_columns].corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(REPORT_PATH / "correlation_matrix.png", dpi=300)
plt.close()

print("\nEDA completed successfully.")
print("Charts saved in reports/.")
