import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path('data/food_waste_data.csv')
rng = np.random.default_rng(42)
n = 2500
dates = pd.date_range('2023-01-01', periods=n, freq='D')
is_weekend = (dates.dayofweek >= 5).astype(int)
is_holiday = rng.binomial(1, 0.08, n)
special_event = rng.binomial(1, 0.12, n)
temperature = np.clip(rng.normal(27, 4.5, n), 16, 39)
rainfall = np.maximum(0, rng.gamma(1.2, 6, n) - 3)
customers = np.clip(330 + 80*is_weekend + 45*special_event - 25*is_holiday + rng.normal(0,35,n), 120, 650).round().astype(int)
meals_prepared = np.clip(customers*rng.uniform(1.03,1.25,n) + 35*special_event + rng.normal(0,25,n),150,850).round().astype(int)
meals_sold = np.minimum(meals_prepared, np.clip(customers*rng.uniform(.88,1.04,n)+rng.normal(0,18,n),100,800)).round().astype(int)
previous_waste = np.zeros(n); previous_waste[0] = 10
for i in range(1,n):
    previous_waste[i] = .45*previous_waste[i-1] + rng.normal(7,2.2) + .012*max(meals_prepared[i-1]-meals_sold[i-1],0)
previous_waste = np.clip(previous_waste,2,40)
avg_meal_price = np.clip(rng.normal(180,35,n),90,320)
staff_count = np.clip(customers/45+rng.normal(0,1.0,n),5,25).round().astype(int)
occupancy_rate = np.clip(customers/rng.uniform(500,650,n)*100+rng.normal(0,5,n),25,100).round(1)
food_category = rng.choice(['Indian','Chinese','Continental','Fast_Food','Bakery'], n, p=[.30,.18,.18,.20,.14])
unsold = np.maximum(meals_prepared-meals_sold,0)
cat = pd.Series(food_category).map({'Indian':1.0,'Chinese':.8,'Continental':1.3,'Fast_Food':.5,'Bakery':.9}).to_numpy()
food_waste = 2.5+.045*unsold+.012*customers+.18*previous_waste+.035*rainfall+.018*np.maximum(temperature-25,0)+1.8*special_event+.7*is_holiday+cat+rng.normal(0,1.0,n)
food_waste = np.clip(food_waste,1,65).round(2)
df = pd.DataFrame({'date':dates,'day_of_week':dates.day_name(),'is_weekend':is_weekend,'is_holiday':is_holiday,'special_event':special_event,'temperature_c':temperature.round(2),'rainfall_mm':rainfall.round(2),'customers':customers,'meals_prepared':meals_prepared,'meals_sold':meals_sold,'previous_waste_kg':previous_waste.round(2),'avg_meal_price':avg_meal_price.round(2),'staff_count':staff_count,'occupancy_rate':occupancy_rate,'food_category':food_category,'food_waste_kg':food_waste})
OUT.parent.mkdir(exist_ok=True, parents=True); df.to_csv(OUT,index=False)
print(f'Saved {len(df)} rows to {OUT}')
