import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# 1. LOAD DATA
# -------------------------------
print("Loading data...")
train = pd.read_csv("train.csv", low_memory=False)
print("Data loaded:", train.shape)

# -------------------------------
# 2. BASIC CLEANING
# -------------------------------
# Keep only open stores
train = train[train["Open"] == 1]

# Convert Date
train["Date"] = pd.to_datetime(train["Date"], dayfirst=True)

# -------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------
train["Year"] = train["Date"].dt.year
train["Month"] = train["Date"].dt.month
train["Day"] = train["Date"].dt.day
train["WeekOfYear"] = train["Date"].dt.isocalendar().week.astype(int)
train["IsWeekend"] = train["DayOfWeek"].isin([6, 7]).astype(int)

# Sort for lag features
train = train.sort_values(["Store", "Date"])

# Lag & rolling features
train["Sales_Lag_1"] = train.groupby("Store")["Sales"].shift(1)
train["Sales_Rolling_7"] = (
    train.groupby("Store")["Sales"]
    .shift(1)
    .rolling(7)
    .mean()
)

# Remove NaNs
train = train.dropna()

# SPEED FIX: sample data
train = train.sample(n=200000, random_state=42)
print("Using sample data:", train.shape)

# -------------------------------
# 4. FEATURE SELECTION
# -------------------------------
features = [
    "Store",
    "DayOfWeek",
    "Promo",
    "SchoolHoliday",
    "Year",
    "Month",
    "Day",
    "WeekOfYear",
    "IsWeekend",
    "Sales_Lag_1",
    "Sales_Rolling_7"
]

X = train[features]
y = train["Sales"]

# -------------------------------
# 5. TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# -------------------------------
# 6. XGBOOST MODEL
# -------------------------------
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

print("Training XGBoost model...")
model.fit(X_train, y_train)
print("Training completed")

# -------------------------------
# 7. PREDICTION
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 8. EVALUATION
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))

# -------------------------------
# 9. VISUALIZATION
# -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.title("XGBoost: Actual vs Predicted Sales")
plt.xlabel("Samples")
plt.ylabel("Sales")
plt.legend()
plt.show()

# -------------------------------
# 10. NEXT 7 DAYS SALES FORECAST
# -------------------------------
store_id = 1

store_data = train[train["Store"] == store_id].sort_values("Date")
last_row = store_data.iloc[-1]

last_date = last_row["Date"]
last_sales = last_row["Sales"]
last_roll = last_row["Sales_Rolling_7"]

print("\nNext 7 Days Sales Forecast for Store", store_id)

for i in range(1, 8):
    next_date = last_date + pd.Timedelta(days=i)

    input_row = pd.DataFrame([{
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

    prediction = model.predict(input_row)[0]
    print(f"{next_date.date()} → Predicted Sales: {int(prediction)}")

    # Update for next day (recursive forecasting)
    last_roll = (last_roll * 6 + prediction) / 7
    last_sales = prediction
