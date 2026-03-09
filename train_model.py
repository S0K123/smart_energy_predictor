import pandas as pd, numpy as np, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/hourly_energy.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp")

# feature engineering
df["hour"] = df["timestamp"].dt.hour
df["dow"]  = df["timestamp"].dt.dayofweek
df["is_wknd"] = (df["dow"]>=5).astype(int)
df["lag1"] = df["kwh"].shift(1)
df["lag24"] = df["kwh"].shift(24)
df["roll24"] = df["kwh"].rolling(24, min_periods=1).mean()

df = df.dropna()  # due to lags

features = ["temp_c","hour","dow","is_wknd","lag1","lag24","roll24"]
target = "kwh"

# time-based split (last 20% as test)
split_idx = int(len(df)*0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]

X_train, y_train = train[features], train[target]
X_test,  y_test  = test[features],  test[target]

model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, pred),3))
print("R2 :", round(r2_score(y_test, pred),3))

joblib.dump({"model":model, "features":features}, "models/energy_model.pkl")
print("Saved models/energy_model.pkl")
