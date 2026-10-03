
import joblib
import pandas as pd

MODEL_PATH = "models/linear_regression_model.joblib"
model = joblib.load(MODEL_PATH)

new_data = pd.DataFrame([{
    "day_of_week": "Friday",
    "is_weekend": 0,
    "is_holiday": 0,
    "special_event": 1,
    "temperature_c": 29,
    "temperature_above_25": max(29 - 25, 0),
    "rainfall_mm": 2,
    "customers": 520,
    "meals_prepared": 620,
    "meals_sold": 530,
    "unsold_meals": max(620 - 530, 0),
    "previous_waste_kg": 14,
    "avg_meal_price": 180,
    "staff_count": 12,
    "occupancy_rate": 84,
    "food_category": "Indian",
}])

prediction = model.predict(new_data)[0]

print("\n===== Food Waste Prediction =====")
print("Predicted food waste:", round(prediction, 2), "kg")
