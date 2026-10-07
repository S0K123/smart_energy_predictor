import numpy as np, pandas as pd
#rng = pd.date_range("2025-08-01", "2025-09-30 23:00", freq="H")  
# 60 days hourly
rng = pd.date_range("2025-08-01", "2025-09-30 23:00", freq="h")
np.random.seed(42)

# base temperature profile (warmer afternoons)
base_temp = 28 + 5*np.sin(2*np.pi*(rng.hour-14)/24) + np.random.normal(0,0.8,len(rng))

# daily pattern (low night, peaks morning/evening)
hour_factor = np.array([0.35,0.32,0.30,0.28,0.28,0.35,0.55,0.70,0.60,0.50,0.45,0.42,
                        0.40,0.42,0.48,0.55,0.70,0.90,1.00,0.95,0.85,0.70,0.55,0.45])
hour_factor = hour_factor / hour_factor.mean()  # normalize

# weekend effect
is_weekend = rng.weekday>=5
wknd_boost = np.where(is_weekend, 1.10, 1.00)

# base daily kWh (12±2)
daily_base = 12 + np.random.normal(0,2,len(rng))  # noisy baseline per hour row

kwh = (daily_base * hour_factor[rng.hour]) * wknd_boost * (1 + 0.015*(base_temp-28))
kwh = np.clip(kwh, 0.2, None)

df = pd.DataFrame({
    "timestamp": rng,
    "temp_c": np.round(base_temp,1),
    "kwh": np.round(kwh,3)
})
#df.to_csv("data/hourly_energy.csv", index=False)
df.to_csv("hourly_energy.csv", index=False)
print("Wrote data/hourly_energy.csv", df.shape)
