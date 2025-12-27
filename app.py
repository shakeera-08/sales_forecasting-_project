import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

st.set_page_config(page_title="Sales Forecasting", layout="centered")

st.title("📈 Sales Forecasting Dashboard")
st.write("Rossmann Store Sales – 7 Day Forecast")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("train.csv", low_memory=False)
    data = data[data["Open"] == 1]
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
    return data

train = load_data()

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
train["Year"] = train["Date"].dt.year
train["Month"] = train["Date"].dt.month
train["Day"] = train["Date"].dt.day
train["WeekOfYear"] = train["Date"].dt.isocalendar().week.astype(int)
train["IsWeekend"] = train["DayOfWeek"].isin([6, 7]).astype(int)

train = train.sort_values(["Store", "Date"])
train["Sales_Lag_1"] = train.groupby("Store")["Sales"].shift(1)
train["Sales_Rolling_7"] = (
    train.groupby("Store")["Sales"].shift(1).rolling(7).mean()
)

train = train.dropna()

# Sample for speed
train = train.sample(n=200000, random_state=42)

features = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday",
    "Year", "Month", "Day", "WeekOfYear",
    "IsWeekend", "Sales_Lag_1", "Sales_Rolling_7"
]

X = train[features]
y = train["Sales"]

# -------------------------------
# TRAIN MODEL
# -------------------------------
@st.cache_resource
def train_model():
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X, y)
    return model

model = train_model()

# -------------------------------
# UI INPUT
# -------------------------------
store_id = st.selectbox(
    "Select Store ID",
    sorted(train["Store"].unique())
)

if st.button("Predict Next 7 Days"):
    store_data = train[train["Store"] == store_id].sort_values("Date")
    last_row = store_data.iloc[-1]

    last_date = last_row["Date"]
    last_sales = last_row["Sales"]
    last_roll = last_row["Sales_Rolling_7"]

    future_dates = []
    future_sales = []

    for i in range(1, 8):
        next_date = last_date + pd.Timedelta(days=i)

        row = pd.DataFrame([{
            "Store": store_id,
            "DayOfWeek": next_date.dayofweek + 1,
            "Promo": 0,
            "SchoolHoliday": 0,
            "Year": next_date.year,
            "Month": next_date.month,
            "Day": next_date.day,
            "WeekOfYear": next_date.isocalendar().week,
            "IsWeekend": 1 if next_date.dayofweek >= 5 else 0,
            "Sales_Lag_1": last_sales,
            "Sales_Rolling_7": last_roll
        }])

        prediction = model.predict(row)[0]

        future_dates.append(next_date.date())
        future_sales.append(int(prediction))

        last_roll = (last_roll * 6 + prediction) / 7
        last_sales = prediction

    result_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Sales": future_sales
    })

    st.subheader("📊 7-Day Sales Forecast")
    st.dataframe(result_df)

    st.line_chart(result_df.set_index("Date"))
