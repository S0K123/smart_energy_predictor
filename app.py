import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import matplotlib.pyplot as plt
from datetime import timedelta


st.set_page_config(page_title="EcoElectro ⚡ Smart Energy Predictor", layout="wide")


st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: #f5f5f5;
            font-family: 'Segoe UI', sans-serif;
        }

        h1, h2, h3, h4 {
            color: #f39c12 !important;
            font-weight: 700;
        }

        section[data-testid="stSidebar"] {
            background-color: #2c2c2c !important;
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: #ffffff !important;
        }

        .stButton>button {
            background-color: #e67e22;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.6em 1.2em;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #d35400;
            color: white;
        }

        .stTabs [role="tablist"] button {
            color: #f5f5f5 !important;
            background: #333333;
            border-radius: 10px;
            border: 1px solid #555555;
            margin-right: 8px;
        }
        .stTabs [role="tablist"] button:hover {
            background-color: #f39c12 !important;
            color: black !important;
        }
        .stTabs [role="tablist"] button[aria-selected="true"] {
            background-color: #e67e22 !important;
            color: white !important;
        }

        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            color: #f5f5f5 !important;
        }
    </style>
""", unsafe_allow_html=True)


st.title("🌿 EcoElectro: Smart Energy Usage Predictor")

bundle = joblib.load("energy_model.pkl")
#bundle = joblib.load("models/energy_model.pkl")
model = bundle["model"]
FEATURES = bundle["features"]

hist = pd.read_csv("hourly_energy.csv", parse_dates=["timestamp"]).sort_values("timestamp")
#hist = pd.read_csv("data/hourly_energy.csv", parse_dates=["timestamp"]).sort_values("timestamp")

# SIDEBAR 
st.sidebar.header("🔧 Prediction Inputs")
base_temp = st.sidebar.slider("Expected Avg Temperature (°C)", 20, 40, int(hist["temp_c"].tail(24).mean()))
start_from = hist["timestamp"].max() + pd.Timedelta(hours=1)
st.sidebar.write("📅 Forecast starts:", f"**{start_from:%Y-%m-%d %H:%M}**")

# TABS
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Predict Next 24h", "📤 Upload CSV"])


# TAB 1: OVERVIEW
with tab1:
    st.subheader("Recent Usage (Last 7 Days)")
    h7 = hist[hist["timestamp"] >= hist["timestamp"].max() - pd.Timedelta(days=7)]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(h7["timestamp"], h7["kwh"], color="#e67e22", linewidth=2)
    ax.set_ylabel("Energy (kWh)")
    ax.set_xlabel("Date (Month & Day)")
    ax.set_title("Actual Hourly Consumption (Last 7 Days)")
    ax.set_xticks(h7["timestamp"][::12])
    ax.set_xticklabels([ts.strftime('%a %d %b') for ts in h7["timestamp"][::12]], rotation=30, ha='right')
    st.pyplot(fig)

    # Peak hour calculation
    peak_ts = h7.loc[h7["kwh"].idxmax(), "timestamp"]
    peak_range = f"{(peak_ts.hour)}:00 - {(peak_ts.hour + 2) % 24}:00"

    st.metric("📈 Avg Hourly kWh (7 days)", f"{h7['kwh'].mean():.2f}")
    st.metric("🌞 Peak Hours (7 days)", peak_range)

    st.subheader("Temperature vs Energy Usage")
    fig2, ax2 = plt.subplots()
    ax2.scatter(hist["temp_c"], hist["kwh"], alpha=0.6, color="#f39c12")
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Energy Usage (kWh)")
    ax2.set_title("Effect of Temperature on Energy Usage")
    st.pyplot(fig2)

# TAB 2: NEXT 24H PREDICTION

with tab2:
    st.subheader("🔮 Predict Next 24 Hours Energy Usage")

    last = hist.tail(24).copy()
    rows = []
    for i in range(24):
        ts = start_from + timedelta(hours=i)
        hour, dow = ts.hour, ts.weekday()
        is_wknd = 1 if dow >= 5 else 0
        temp = base_temp + 3 * np.sin(2 * np.pi * (hour - 14) / 24)

        lag1 = last["kwh"].iloc[-1] if len(rows) == 0 else rows[-1]["pred"]
        lag24 = last["kwh"].iloc[i] if i < len(last) else rows[i - 24]["pred"]
        roll24 = pd.Series([*(last["kwh"].tolist()), *(r.get("pred", 0) for r in rows)]).tail(24).mean()

        X_temp = pd.DataFrame([{
            "temp_c": temp, "hour": hour, "dow": dow, "is_wknd": is_wknd,
            "lag1": lag1, "lag24": lag24, "roll24": roll24
        }])

        pred = model.predict(X_temp[FEATURES])[0]
        rows.append({
            "timestamp": ts, "temp_c": temp, "hour": hour, "dow": dow,
            "is_wknd": is_wknd, "lag1": lag1, "lag24": lag24, "roll24": roll24,
            "pred": pred
        })

    df_pred = pd.DataFrame(rows)

    # Plot predictions with month & day labels
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(df_pred["timestamp"], df_pred["pred"], marker="o", color="#e67e22", linewidth=2)
    ax3.set_ylabel("Predicted Energy (kWh)")
    ax3.set_xlabel("Day & Hour (with Month)")
    ax3.set_title("Predicted Hourly Energy Consumption (Next 24 Hours)")
    ax3.set_xticks(df_pred["timestamp"][::3])
    ax3.set_xticklabels([ts.strftime('%a %d %b %H:%M') for ts in df_pred["timestamp"][::3]], rotation=30, ha='right')
    st.pyplot(fig3)

    # Suggest optimal hours (lowest predicted load)
    k = st.slider("How many low-load hours to suggest?", 3, 8, 3)
    best = df_pred.nsmallest(k, "pred")[["timestamp", "pred"]].sort_values("timestamp")

    st.subheader("🌙 Suggested Low-Load Hours (Ideal for Shiftable Appliances)")
    for _, r in best.iterrows():
        st.write(f"• {r['timestamp'].strftime('%a %d %b %H:%M')} → ~{r['pred']:.2f} kWh")

    # Save to SQLite
    if st.button("💾 Save forecast to database"):
        conn = sqlite3.connect("energy_logs.db")
        df_pred[["timestamp", "pred", "temp_c"]].to_sql("forecast", conn, if_exists="append", index=False)
        conn.close()
        st.success("✅ Forecast saved successfully to energy_logs.db")

# TAB 3: UPLOAD CUSTOM CSV
with tab3:
    st.subheader("📤 Upload Your Own Hourly CSV")
    st.caption("Columns: timestamp,temp_c,kwh  (timestamp in ISO, hourly readings)")

    up = st.file_uploader("Choose CSV file", type=["csv"])
    if up:
        user = pd.read_csv(up, parse_dates=["timestamp"]).sort_values("timestamp")
        st.write(user.head())
        st.write("Rows:", len(user))

        fig4, ax4 = plt.subplots(figsize=(10, 3))
        ax4.plot(user["timestamp"], user["kwh"], color="#d35400", linewidth=2)
        ax4.set_ylabel("kWh")
        ax4.set_xlabel("Date & Time (with Month)")
        ax4.set_title("User Uploaded Data")
        ax4.set_xticks(user["timestamp"][::6])
        ax4.set_xticklabels([ts.strftime('%a %d %b %H:%M') for ts in user["timestamp"][::6]], rotation=30, ha='right')
        st.pyplot(fig4)
