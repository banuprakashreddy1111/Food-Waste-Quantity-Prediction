
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/linear_regression_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.set_page_config(
    page_title="Food Waste Quantity Prediction",
    page_icon="🍽️",
    layout="centered",
)

st.title("🍽️ Food Waste Quantity Prediction")
st.write("Predict expected food waste in kilograms using Linear Regression.")

st.subheader("Prediction Details")

day = st.selectbox(
    "Day of week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
)
is_weekend = int(day in ["Saturday", "Sunday"])

holiday = st.selectbox("Holiday?", ["No", "Yes"])
special_event = st.selectbox("Special event?", ["No", "Yes"])

temperature = st.number_input(
    "Temperature (°C)", min_value=10.0, max_value=45.0, value=28.0
)
rainfall = st.number_input(
    "Rainfall (mm)", min_value=0.0, max_value=200.0, value=5.0
)
customers = st.number_input(
    "Customers", min_value=50, max_value=1000, value=450
)
meals_prepared = st.number_input(
    "Meals prepared", min_value=50, max_value=1200, value=550
)
meals_sold = st.number_input(
    "Meals sold", min_value=0, max_value=1200, value=470
)
previous_waste = st.number_input(
    "Previous waste (kg)", min_value=0.0, max_value=100.0, value=12.0
)
avg_meal_price = st.number_input(
    "Average meal price", min_value=20.0, max_value=1000.0, value=180.0
)
staff_count = st.number_input(
    "Staff count", min_value=1, max_value=100, value=12
)
occupancy_rate = st.slider("Occupancy rate (%)", 0, 100, 80)

food_category = st.selectbox(
    "Food category",
    ["Indian", "Chinese", "Continental", "Fast_Food", "Bakery"],
)

if st.button("Predict Food Waste"):
    unsold_meals = max(meals_prepared - meals_sold, 0)
    sales_ratio = meals_sold / meals_prepared if meals_prepared else 0
    temperature_above_25 = max(temperature - 25, 0)

    input_data = pd.DataFrame([{
        "day_of_week": day,
        "is_weekend": is_weekend,
        "is_holiday": int(holiday == "Yes"),
        "special_event": int(special_event == "Yes"),
        "temperature_c": temperature,
        "temperature_above_25": temperature_above_25,
        "rainfall_mm": rainfall,
        "customers": customers,
        "meals_prepared": meals_prepared,
        "meals_sold": meals_sold,
        "unsold_meals": unsold_meals,
        "previous_waste_kg": previous_waste,
        "avg_meal_price": avg_meal_price,
        "staff_count": staff_count,
        "occupancy_rate": occupancy_rate,
        "food_category": food_category,
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated food waste: {prediction:.2f} kg")
