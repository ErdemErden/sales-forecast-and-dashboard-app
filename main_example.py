import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime

# Read the CSV file ------------------- STEP 1 - Load & Clean Data
test_df = pd.read_csv("sample_sales_data.csv")
# Clean columns
test_df.columns = test_df.columns.str.strip()
test_df["Date"] = pd.to_datetime(test_df["Date"])

# Show the first 5 rows --------------- STEP 2 - Basic Info
print("First 5 rows:")
print(test_df.head())
# Show total number of rows
print(f"\nTotal number of rows: {len(test_df)}")
# Show column names
print("\nColumn names:", test_df.columns.tolist())

#Plot Sales Quantity over Date -------- STEP 3 - Plot Historical Sales
plt.figure(figsize=(10,5))
plt.plot(test_df["Date"], test_df["Sales_Quantity"], marker='o')
plt.title("Sales Quantity Over Time")
plt.xlabel("Date")
plt.ylabel("Sales Quantity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Convert Date to ordinal -------------- STEP 4 - Linear Regression Forecast (7 Days)
test_df["Date_Ordinal"] = test_df["Date"].map(lambda x: x.toordinal())

#Train Linear Regression Model
X = test_df["Date_Ordinal"].values.reshape(-1, 1)
y = test_df["Sales_Quantity"].values

model = LinearRegression()
model.fit(X, y)

#Predict for the next 7 days
future_dates = [test_df["Date"].max() + pd.Timedelta(days=i) for i in range(1, 8)]
future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
predicted_sales = model.predict(future_ordinals)

#Show forecast
print("\n Linear Regression Forecast (7 Days):")
for date, pred in zip(future_dates, predicted_sales):
    print(f"{date.date()} --> Predicted Sales: {int(pred)}")

#Plot Forecast
plt.figure(figsize=(12,5))
plt.plot(test_df["Date"],test_df["Sales_Quantity"], marker='o', label="Historical Sales")
plt.plot(future_dates,predicted_sales,marker='x',linestyle='--',label="Forecast (7 Days")
plt.xlabel("Date")
plt.ylabel("Sales Quantity")
plt.title("Sales Forecast")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

#----------Moving Average Forecast-------- STEP 5
#Set "Date" as index (for time series operations)
test_df.set_index("Date", inplace=True)

#Calculate moving average
test_df["MA_7"] = test_df["Sales_Quantity"].rolling(window=7).mean()
test_df["MA_30"] = test_df["Sales_Quantity"].rolling(window=30).mean()
test_df["MA_90"] = test_df["Sales_Quantity"].rolling(window=90).mean()

#Show latest moving average values
print("\nLatest_7-Day Moving Average Forecast:", int(test_df["MA_7"].dropna().iloc[-1]))
print("Latest 30-Day Moving Average Forecast:", int(test_df["MA_30"].dropna().iloc[-1]))
print("Latest 90-Day Moving Average Forecast", "Not enough data" if test_df["MA_90"].isna().all() else int(test_df["MA_90"].dropna().iloc[-1]))

#Plot Moving Averages
plt.figure(figsize=(12,5))
plt.plot(test_df["Sales_Quantity"], label= "Actual Sales", marker='o')
plt.plot(test_df["MA_7"], label="7-Day MA", linestyle="--")
plt.plot(test_df["MA_30"], label="30-Day MA", linestyle="--")
plt.plot(test_df["MA_90"], label="90-Day MA", linestyle="--")
plt.title("Sales Quantity & Moving Averages")
plt.xlabel("Date")
plt.ylabel("Sales Quantity")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#------------------ STEP 6 - Export CSV ---------------------
# Reset index to get Date back as column
test_df.reset_index(inplace=True)

#Save forecast output
forecast_test_df = pd.DataFrame({
        "Date": test_df["Date"].iloc[-7:].dt.date,
        "7-Day MA Forecast": test_df["MA_7"].iloc[-7:].tolist(),
        "30-Day MA Forecast": test_df["MA_30"].iloc[-7:].tolist()
    })
forecast_test_df.to_csv("forecast_output.csv", index=False)
print("✅ Forecast exported to forecast_output.csv")

